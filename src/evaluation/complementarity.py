"""Calibration-only threshold curves and paired object-recovery diagnostics."""

import numpy as np
import torch
from torchvision.ops import box_iou

from src.evaluation.comparison_protocol import GRID, NAMES


def assignments(record, threshold=0.001):
    boxes = np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4)
    gt = np.asarray(record["gt_boxes"], dtype=np.float32).reshape(-1, 4)
    iou = box_iou(torch.from_numpy(boxes), torch.from_numpy(gt)).numpy()
    matched = {}
    used = set()
    for p in np.argsort(-np.asarray(record["scores"]), kind="stable"):
        if record["scores"][p] < threshold:
            continue
        candidates = [
            g
            for g, c in enumerate(record["gt_classes"])
            if g not in used and c == record["classes"][p] and iou[p, g] >= 0.5
        ]
        if candidates:
            g = max(candidates, key=lambda g: iou[p, g])
            used.add(g)
            matched[g] = {
                "prediction": int(p),
                "score": record["scores"][p],
                "iou": float(iou[p, g]),
            }
    return matched


def select_threshold(curve):
    # Scores equal to 12 decimal places tie; choose the higher threshold.
    return max(curve, key=lambda r: (round(r["macro_class_f1"], 12), r["threshold"]))[
        "threshold"
    ]


def threshold_curve(records, grid=GRID):
    supports = np.bincount([c for r in records for c in r["gt_classes"]], minlength=14)
    matches = [assignments(r) for r in records]
    scores = np.array([s for r in records for s in r["scores"]])
    classes = np.array([c for r in records for c in r["classes"]], dtype=int)
    tp_scores = np.array([v["score"] for match in matches for v in match.values()])
    tp_classes = np.array(
        [r["gt_classes"][g] for r, match in zip(records, matches) for g in match],
        dtype=int,
    )
    out = []
    present = supports > 0
    for threshold in grid:
        predictions = np.bincount(classes[scores >= threshold], minlength=14)
        tp = np.bincount(tp_classes[tp_scores >= threshold], minlength=14)
        p = np.divide(tp, predictions, out=np.zeros(14), where=predictions > 0)
        rec = np.divide(tp, supports, out=np.zeros(14), where=present)
        f = np.divide(2 * p * rec, p + rec, out=np.zeros(14), where=p + rec > 0)
        mp, mr = float(p[present].mean()), float(rec[present].mean())
        out.append(
            {
                "threshold": threshold,
                "precision": mp,
                "recall": mr,
                "harmonic_aggregate_f1": 2 * mp * mr / (mp + mr) if mp + mr else 0.0,
                "macro_class_f1": float(f[present].mean()),
                "predictions": int(predictions.sum()),
                "tp": int(tp.sum()),
            }
        )
    return out


def paired_analysis(left, right, thresholds):
    if [r["image_id"] for r in left] != [r["image_id"] for r in right]:
        raise ValueError("Unpaired images")
    counts = np.zeros((14, 4), dtype=int)
    confidence = []
    ious = []
    overlap = []
    per_image = []
    for a, b in zip(left, right):
        if a["gt_boxes"] != b["gt_boxes"] or a["gt_classes"] != b["gt_classes"]:
            raise ValueError("Unpaired ground truth")
        am, bm = assignments(a, thresholds["E1"]), assignments(b, thresholds["E3"])
        n = np.zeros(4, dtype=int)
        for g, c in enumerate(a["gt_classes"]):
            category = (
                0 if g in am and g in bm else 1 if g in am else 2 if g in bm else 3
            )
            counts[c, category] += 1
            n[category] += 1
            if category == 0:
                confidence.append([am[g]["score"], bm[g]["score"]])
                ious.append([am[g]["iou"], bm[g]["iou"]])
        aa = [p for p, s in enumerate(a["scores"]) if s >= thresholds["E1"]]
        bb = [p for p, s in enumerate(b["scores"]) if s >= thresholds["E3"]]
        matrix = box_iou(
            torch.tensor([a["boxes"][p] for p in aa], dtype=torch.float32).reshape(
                -1, 4
            ),
            torch.tensor([b["boxes"][p] for p in bb], dtype=torch.float32).reshape(
                -1, 4
            ),
        ).numpy()
        for i, p in enumerate(aa):
            candidates = [
                j for j, q in enumerate(bb) if b["classes"][q] == a["classes"][p]
            ]
            overlap.append(float(max(matrix[i, candidates], default=0.0)))
        per_image.append(
            {
                "image_id": a["image_id"],
                **dict(zip(["both", "E1_only", "E3_only", "neither"], n.tolist())),
            }
        )
    confidence = np.array(confidence)
    ious = np.array(ious)
    overlap = np.array(overlap)
    return {
        "thresholds": thresholds,
        "matching_iou": 0.5,
        "counts": dict(
            zip(["both", "E1_only", "E3_only", "neither"], counts.sum(0).tolist())
        ),
        "per_class": [
            {
                "class_id": i,
                "name": name,
                "gt_support": int(counts[i].sum()),
                "meaningful_support": int(counts[i].sum()) >= 50,
                **dict(
                    zip(["both", "E1_only", "E3_only", "neither"], counts[i].tolist())
                ),
            }
            for i, name in enumerate(NAMES)
        ],
        "confidence_correlation": {
            "method": "Pearson on scores for GT objects correctly detected by both; conditional descriptive statistic, not calibrated probability agreement",
            "objects": len(confidence),
            "pearson": float(np.corrcoef(confidence.T)[0, 1])
            if len(confidence) > 1 and all(confidence.std(0) > 0)
            else None,
        },
        "localization": {
            "objects": len(ious),
            "E1_mean_iou": float(ious[:, 0].mean()) if len(ious) else None,
            "E3_mean_iou": float(ious[:, 1].mean()) if len(ious) else None,
            "mean_E3_minus_E1": float((ious[:, 1] - ious[:, 0]).mean())
            if len(ious)
            else None,
        },
        "prediction_overlap": {
            "definition": "For every E1 operating-threshold prediction, maximum same-class IoU with any E3 operating prediction in same image; missing candidate gives zero, many-to-one",
            "E1_predictions": len(overlap),
            "mean_max_iou": float(overlap.mean()) if len(overlap) else None,
            "fraction_iou_ge_50": float((overlap >= 0.5).mean())
            if len(overlap)
            else None,
            "fraction_iou_ge_75": float((overlap >= 0.75).mean())
            if len(overlap)
            else None,
        },
        "limitations": "Not a fusion result: oracle unique recoveries do not quantify attainable fusion AP. Rare classes <50 GT are flagged; no significance tests on this calibration analysis.",
    }, per_image
