"""Fixed-threshold diagnostics distinct from COCO AP integration."""

import contextlib
import io
from collections import Counter

import numpy as np

from dashboard.new_data.ingest import NAMES
from src.evaluation.error_analysis import iou_matrix

CONFIDENCE = 0.34


def harmonic(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def fixed(record, threshold=CONFIDENCE):
    pb = np.asarray(record["boxes"]).reshape(-1, 4)
    gb = np.asarray(record["gt_boxes"]).reshape(-1, 4)
    overlap = iou_matrix(pb, gb)
    used = set()
    events = []
    tp = np.zeros(14, int)
    fp = tp.copy()
    fn = tp.copy()
    order = sorted(range(len(pb)), key=lambda i: (-record["scores"][i], i))
    for pi in order:
        score = record["scores"][pi]
        c = record["classes"][pi]
        same = [g for g in range(len(gb)) if record["gt_classes"][g] == c]
        eligible = [g for g in same if g not in used and overlap[pi, g] >= 0.5]
        if score < threshold:
            if eligible:
                events.append(
                    {
                        "kind": "low-confidence correct candidate",
                        "prediction": pi,
                        "gt": max(eligible, key=lambda g: overlap[pi, g]),
                        "class_id": c,
                        "confidence": score,
                        "iou": float(max(overlap[pi, g] for g in eligible)),
                    }
                )
            continue
        gi = int(np.argmax(overlap[pi])) if len(gb) else None
        best = float(overlap[pi, gi]) if gi is not None else 0.0
        if eligible:
            gi = max(eligible, key=lambda g: overlap[pi, g])
            used.add(gi)
            tp[c] += 1
            kind = "correct detection"
            best = float(overlap[pi, gi])
        else:
            fp[c] += 1
            if any(overlap[pi, g] >= 0.5 for g in same):
                kind = "duplicate detection"
            elif best >= 0.5:
                kind = "classification error"
            elif best >= 0.1:
                kind = "localization error"
            else:
                kind = "background false positive"
        events.append(
            {
                "kind": kind,
                "prediction": pi,
                "gt": gi,
                "class_id": c,
                "confidence": score,
                "iou": best,
                "annotation_review_candidate": kind == "background false positive"
                and score >= 0.8,
            }
        )
    for gi, c in enumerate(record["gt_classes"]):
        if gi not in used:
            fn[c] += 1
            events.append(
                {
                    "kind": "missed object",
                    "prediction": None,
                    "gt": gi,
                    "class_id": c,
                    "confidence": None,
                    "iou": None,
                }
            )
    return tp, fp, fn, events


def summary_counts(tp, fp, fn):
    p = np.divide(tp, tp + fp, out=np.zeros(14), where=tp + fp > 0)
    r = np.divide(tp, tp + fn, out=np.zeros(14), where=tp + fn > 0)
    present = tp + fn > 0
    P = float(p[present].mean()) if present.any() else 0.0
    R = float(r[present].mean()) if present.any() else 0.0
    mp = float(tp.sum() / max(1, (tp + fp).sum()))
    mr = float(tp.sum() / max(1, (tp + fn).sum()))
    return {
        "precision": P,
        "recall": R,
        "f1": harmonic(P, R),
        "micro_precision": mp,
        "micro_recall": mr,
        "micro_f1": harmonic(mp, mr),
    }


def labeled(records, bootstrap=100):
    from src.evaluation.common_metrics import measure

    with contextlib.redirect_stdout(io.StringIO()):
        overall, classes, confusion, grid, _ = measure(
            records, NAMES, confidence=CONFIDENCE
        )
    support_counts = Counter(c for r in records for c in r["gt_classes"])
    events = []
    per_image = []
    counts = []
    strata = []
    for rec in records:
        tp, fp, fn, errs = fixed(rec)
        counts.append((tp, fp, fn))
        for e in errs:
            e.update(image_id=rec["image_id"], name=rec["name"])
        events.extend(errs)
        per_image.append(
            dict(
                image_id=rec["image_id"],
                name=rec["name"],
                tp=int(tp.sum()),
                fp=int(fp.sum()),
                fn=int(fn.sum()),
                gt=len(rec["gt_boxes"]),
                predictions=int((tp + fp).sum()),
                **summary_counts(tp, fp, fn),
            )
        )
        matched = {e["gt"] for e in errs if e["kind"] == "correct detection"}
        for gi, (b, c) in enumerate(zip(rec["gt_boxes"], rec["gt_classes"])):
            area = (b[2] - b[0]) * (b[3] - b[1])
            cx = (b[0] + b[2]) / 2 / rec["width"]
            cy = (b[1] + b[3]) / 2 / rec["height"]
            strata.append(
                {
                    "class_id": c,
                    "class_frequency": "low support"
                    if support_counts[c] < 20
                    else "represented",
                    "size": "small"
                    if area < 1024
                    else "medium"
                    if area < 9216
                    else "large",
                    "density": "sparse"
                    if len(rec["gt_boxes"]) <= 5
                    else "moderate"
                    if len(rec["gt_boxes"]) <= 15
                    else "dense",
                    "aspect_ratio": "portrait"
                    if rec["width"] / rec["height"] < 0.8
                    else "landscape"
                    if rec["width"] / rec["height"] > 1.25
                    else "square",
                    "position": "edge"
                    if min(cx, cy, 1 - cx, 1 - cy) < 0.2
                    else "center",
                    "matched": int(gi in matched),
                }
            )
    curves = []
    for threshold in np.linspace(0, 1, 21):
        totals = np.sum([fixed(r, float(threshold))[:3] for r in records], axis=0)
        curves.append(dict(confidence=float(threshold), **summary_counts(*totals)))
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(bootstrap):
        indices = rng.integers(0, len(records), len(records))
        total = np.sum([counts[i] for i in indices], axis=0)
        samples.append(summary_counts(*total))
    ci = (
        {
            k: [float(x) for x in np.percentile([s[k] for s in samples], [2.5, 97.5])]
            for k in samples[0]
        }
        if samples
        else {}
    )
    valid = lambda a: float(a[a >= 0].mean()) if (a >= 0).any() else None
    totals = np.sum(counts, axis=0)
    overall.update(
        {k: v for k, v in summary_counts(*totals).items() if k.startswith("micro")}
    )
    return {
        "overall": overall,
        "per_class": classes,
        "per_image": per_image,
        "errors": events,
        "confusion": confusion.tolist(),
        "curves": curves,
        "ap_by_iou": [
            {"iou": round(0.5 + 0.05 * i, 2), "ap": valid(grid[i])} for i in range(10)
        ],
        "pr": [{"recall": i / 100, "precision": valid(grid[0, i])} for i in range(101)],
        "strata": strata,
        "bootstrap": {
            "seed": 42,
            "replicates": bootstrap,
            "unit": "image",
            "interval": "95% percentile; fixed-threshold metrics only; no significance test; AP CI unavailable",
            "intervals": ci,
        },
        "error_counts": dict(Counter(e["kind"] for e in events)),
    }


def unlabeled(records):
    kept = [
        (c, s)
        for r in records
        for c, s in zip(r["classes"], r["scores"])
        if s >= CONFIDENCE
    ]
    return {
        "predictions": len(kept),
        "average_detections": len(kept) / len(records),
        "mean_confidence": float(np.mean([s for c, s in kept])) if kept else None,
        "class_coverage": len({c for c, s in kept}),
        "class_counts": dict(Counter(NAMES[c] for c, s in kept)),
    }


def latency(rows):
    result = {}
    for key in [
        "preprocessing_ms",
        "inference_ms",
        "postprocessing_ms",
        "end_to_end_ms",
    ]:
        values = [r[key] for r in rows]
        result[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
        }
    result["still_images_per_second"] = 1000 / result["end_to_end_ms"]["mean"]
    result["inference_images_per_second"] = 1000 / result["inference_ms"]["mean"]
    return result
