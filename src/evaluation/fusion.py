"""Bounded, deterministic E1-primary fusion; no model or dataset access."""

import random

import numpy as np

IOU_GRID = [0.45, 0.55, 0.65]
WEIGHTS = [(0.70, 0.30), (0.85, 0.15)]
ADDITION_GRID = [0.35, 0.45, 0.55]
OPERATING_GRID = [round(0.20 + i * 0.05, 2) for i in range(9)]


def overlaps(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    lo = np.maximum(box[:2], boxes[:, :2])
    hi = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(hi - lo, 0).prod(1)
    area = (boxes[:, 2:] - boxes[:, :2]).prod(1)
    own = np.prod(np.asarray(box[2:]) - box[:2])
    return intersection / np.maximum(own + area - intersection, 1e-12)


def check(record):
    n = len(record["scores"])
    if len(record["boxes"]) != n or len(record["classes"]) != n:
        raise ValueError("Unaligned prediction arrays")
    for box, cls, score in zip(record["boxes"], record["classes"], record["scores"]):
        if (
            len(box) != 4
            or not np.isfinite([*box, score]).all()
            or not 0.001 <= score <= 1
        ):
            raise ValueError("Invalid/non-finite prediction")
        if int(cls) != cls or not 0 <= cls < 14 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Invalid class/box")


def candidates():
    output = [{"id": "baseline", "algorithm": "baseline"}]
    for iou in IOU_GRID:
        output.append({"id": f"union_{iou:.2f}", "algorithm": "union", "iou": iou})
        for a, b in WEIGHTS:
            output.append(
                {
                    "id": f"wbf_{iou:.2f}_{a:.2f}",
                    "algorithm": "wbf",
                    "iou": iou,
                    "weights": [a, b],
                }
            )
        for threshold in ADDITION_GRID:
            output.append(
                {
                    "id": f"recovery_{iou:.2f}_{threshold:.2f}",
                    "algorithm": "recovery",
                    "iou": iou,
                    "addition_threshold": threshold,
                }
            )
    return output


def class_nms(boxes, scores, classes, threshold):
    order = np.argsort(-np.asarray(scores), kind="stable").tolist()
    kept = []
    while order:
        first = order.pop(0)
        kept.append(first)
        if not order:
            break
        iou = overlaps(np.asarray(boxes[first]), [boxes[i] for i in order])
        order = [
            i
            for j, i in enumerate(order)
            if classes[i] != classes[first] or iou[j] <= threshold
        ]
    return kept


def fuse(a, b, configuration, context=None):
    check(a)
    check(b)
    if a["image_id"] != b["image_id"] or (a["width"], a["height"]) != (
        b["width"],
        b["height"],
    ):
        raise ValueError("Unpaired images")
    cfg = configuration
    kind = cfg["algorithm"]
    if cfg not in candidates():
        raise ValueError("Outside frozen candidate grid")
    result = {k: v for k, v in a.items() if k not in ["boxes", "scores", "classes"]}
    boxes = [list(x) for x in a["boxes"]]
    scores = list(a["scores"])
    classes = list(a["classes"])
    if kind == "union":
        boxes += [list(x) for x in b["boxes"]]
        scores += list(b["scores"])
        classes += list(b["classes"])
        keep = class_nms(boxes, scores, classes, cfg["iou"])[:300]
        boxes, scores, classes = (
            [values[i] for i in keep] for values in [boxes, scores, classes]
        )
    elif kind == "wbf":
        used = set()
        w1, w3 = cfg["weights"]
        for i in np.argsort(-np.asarray(scores), kind="stable"):
            available = [
                j
                for j, c in enumerate(b["classes"])
                if c == classes[i] and j not in used
            ]
            if not available:
                continue
            ious = overlaps(np.asarray(boxes[i]), [b["boxes"][j] for j in available])
            j = available[int(np.argmax(ious))]
            if max(ious) < cfg["iou"]:
                continue
            used.add(j)
            s1, s3 = scores[i], b["scores"][j]
            boxes[i] = (
                (w1 * s1 * np.array(boxes[i]) + w3 * s3 * np.array(b["boxes"][j]))
                / (w1 * s1 + w3 * s3)
            ).tolist()
            scores[i] = w1 * s1 + w3 * s3
        # E1-anchored two-model WBF: no unmatched E3 additions, no E1 removal.
    elif kind == "recovery":
        if context is None:
            raise ValueError("Training-fold support context required")
        eligible = set(context["eligible_classes"])
        weak = set(context["weak_classes"])
        anchors = [
            box
            for box, score in zip(a["boxes"], a["scores"])
            if score >= OPERATING_GRID[0]
        ]
        added = []
        for j in np.argsort(-np.asarray(b["scores"]), kind="stable"):
            cls = b["classes"][j]
            score = b["scores"][j]
            box = b["boxes"][j]
            threshold = max(
                cfg["addition_threshold"],
                0.55 if cls in weak else cfg["addition_threshold"],
            )
            if cls not in eligible or score < threshold:
                continue
            if anchors and max(overlaps(np.asarray(box), anchors)) >= cfg["iou"]:
                continue
            if (
                added
                and max(overlaps(np.asarray(box), [b["boxes"][k] for k in added]))
                >= cfg["iou"]
            ):
                continue
            if len(boxes) >= 300:
                break  # Never evict an existing E1 prediction.
            added.append(j)
            boxes.append(list(box))
            scores.append(score)
            classes.append(cls)
    elif kind != "baseline":
        raise ValueError("Unknown fusion algorithm")
    # Stable descending order is shared by every export. E1 values remain unchanged in recovery.
    order = np.argsort(-np.asarray(scores), kind="stable")[:300]
    result.update(
        boxes=[boxes[i] for i in order],
        scores=[scores[i] for i in order],
        classes=[classes[i] for i in order],
    )
    check(result)
    return result


def make_folds(image_ids, seed=42):
    if len(image_ids) != len(set(image_ids)) or len(image_ids) % 5:
        raise ValueError("Need unique IDs divisible into five folds")
    ids = sorted(image_ids)
    random.Random(seed).shuffle(ids)
    return [sorted(ids[i::5]) for i in range(5)]


def split_fold(rows, folds, index):
    ids = [i for fold in folds for i in fold]
    if len(ids) != len(set(ids)) or set(ids) != {r["image_id"] for r in rows}:
        raise ValueError("Fold leakage, duplicate or missing IDs")
    held = set(folds[index])
    train = [r for r in rows if r["image_id"] not in held]
    test = [r for r in rows if r["image_id"] in held]
    return train, test


def improvement_gate(
    baseline, other, base_classes, classes, base_errors, errors, reproducible=True
):
    keys = ["map50_95", "precision", "recall", "macro_class_f1"]
    delta = {k: other[k] - baseline[k] for k in keys}
    gains = {
        k: np.array(
            [float(c[k] or 0) - float(b[k] or 0) for b, c in zip(base_classes, classes)]
        )
        for k in ["f1", "ap50_95"]
    }
    supports = np.array([c["gt_count"] for c in classes])
    frequent = int(np.argmax(supports))
    distributed = any(
        (
            delta[metric] >= limit
            and sum((gain >= 0.01) & (supports >= 100)) >= 2
            and np.delete(gain, frequent).mean() > 0
        )
        for key, metric, limit in [
            ("f1", "macro_class_f1", 0.02),
            ("ap50_95", "map50_95", 0.01),
        ]
        for gain in [gains[key]]
    )
    bd, od = base_errors["duplicates"], errors["duplicates"]
    br = bd / max(baseline["predictions_fixed"], 1)
    orr = od / max(other["predictions_fixed"], 1)
    checks = {
        "ap_noninferiority": delta["map50_95"] >= -0.005,
        "precision_noninferiority": delta["precision"] >= -0.01,
        "recall_noninferiority": delta["recall"] >= -0.01,
        "minimum_improvement": delta["macro_class_f1"] >= 0.02
        or delta["map50_95"] >= 0.01,
        "distributed_class_gain": bool(distributed),
        "duplicate_safety": od - bd <= max(10, 0.25 * bd) and orr - br <= 0.02,
        "reproducible": bool(reproducible),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "deltas": delta,
        "duplicates_baseline": bd,
        "duplicates_candidate": od,
        "both_precision_recall_70": other["precision"] >= 0.70
        and other["recall"] >= 0.70,
    }
