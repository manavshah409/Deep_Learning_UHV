"""Common architecture-independent COCO AP and fixed-threshold diagnostics v1."""

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou

from src.evaluation.error_analysis import iou_matrix

CONFIDENCE = 0.25
MATCH_IOU = 0.5


def harmonic(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def measure(records, names, confidence=0.25):
    if not 0 <= confidence <= 1:
        raise ValueError("Invalid confidence threshold")
    gt = {
        "info": {},
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)],
    }
    predictions = []
    tp, fp, fn, support, raw_counts, kept_counts = [
        np.zeros(14, dtype=int) for _ in range(6)
    ]
    confusion = np.zeros((15, 15), dtype=int)
    image_errors = []
    for record in records:
        iid = record["image_id"]
        gb = np.asarray(record["gt_boxes"]).reshape(-1, 4)
        gc = np.asarray(record["gt_classes"], dtype=int)
        pb = np.asarray(record["boxes"]).reshape(-1, 4)
        pc = np.asarray(record["classes"], dtype=int)
        scores = np.asarray(record["scores"])
        gt["images"].append(
            {"id": iid, "height": record["height"], "width": record["width"]}
        )
        for box, cls in zip(gb, gc):
            x1, y1, x2, y2 = box.tolist()
            gt["annotations"].append(
                {
                    "id": len(gt["annotations"]) + 1,
                    "image_id": iid,
                    "category_id": int(cls) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
        for box, cls, score in zip(pb, pc, scores):
            x1, y1, x2, y2 = box.tolist()
            predictions.append(
                {
                    "image_id": iid,
                    "category_id": int(cls) + 1,
                    "score": float(score),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                }
            )
        raw_counts += np.bincount(pc, minlength=14)
        keep = scores >= confidence
        pb, pc, scores = pb[keep], pc[keep], scores[keep]
        support += np.bincount(gc, minlength=14)
        kept_counts += np.bincount(pc, minlength=14)
        overlaps = iou_matrix(pb, gb)
        order = np.argsort(-scores, kind="stable")
        used = set()
        correct = []
        # Stable score order; equal scores retain original prediction index.
        fixed_order = np.argsort(-scores, kind="stable")
        fixed_iou = box_iou(
            torch.tensor(pb, dtype=torch.float32), torch.tensor(gb, dtype=torch.float32)
        ).numpy()
        for pi in fixed_order:
            candidates = [
                g
                for g in range(len(gc))
                if gc[g] == pc[pi] and g not in used and fixed_iou[pi, g] >= MATCH_IOU
            ]
            if candidates:
                gi = max(candidates, key=lambda g: fixed_iou[pi, g])
                used.add(gi)
                tp[pc[pi]] += 1
                correct.append((int(pi), gi))
            else:
                fp[pc[pi]] += 1
        missed = [g for g in range(len(gc)) if g not in used]
        for gi in missed:
            fn[gc[gi]] += 1
        # Confusion matrix is a separate class-agnostic score-ordered matching.
        matched_gt = set()
        matches = []
        background_predictions = []
        for pi in order:
            candidates = [
                g
                for g in range(len(gc))
                if g not in matched_gt and overlaps[pi, g] >= MATCH_IOU
            ]
            if candidates:
                gi = max(candidates, key=lambda g: overlaps[pi, g])
                matched_gt.add(gi)
                confusion[gc[gi], pc[pi]] += 1
                matches.append(
                    {
                        "gt": gi,
                        "prediction": int(pi),
                        "iou": float(overlaps[pi, gi]),
                        "correct_class": bool(gc[gi] == pc[pi]),
                    }
                )
            else:
                confusion[14, pc[pi]] += 1
                background_predictions.append(int(pi))
        for gi in set(range(len(gc))) - matched_gt:
            confusion[gc[gi], 14] += 1
        duplicates, localization, background = [], [], []
        for pi in background_predictions:
            same = np.where(gc == pc[pi])[0]
            best = float(overlaps[pi, same].max()) if len(same) else 0.0
            if best >= MATCH_IOU:
                duplicates.append(pi)
            elif best >= 0.1:
                localization.append(pi)
            else:
                background.append(pi)
        scale = min(
            480 / min(record["width"], record["height"]),
            640 / max(record["width"], record["height"]),
        )
        area = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
        small = [int(g) for g in missed if area[g] < 32**2]
        resized_small = [int(g) for g in missed if area[g] * scale**2 < 16**2]
        image_errors.append(
            {
                "image_id": iid,
                "gt_count": len(gc),
                "predictions_fixed": len(pc),
                "correct": len(correct),
                "class_aware_fp": len(pc) - len(correct),
                "class_aware_fn": len(missed),
                "missed_gt": missed,
                "missed_original_small": small,
                "missed_resized_small": resized_small,
                "confusion_matches": matches,
                "duplicate_predictions": duplicates,
                "localization_candidates": localization,
                "background_candidates": background,
            }
        )
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    if predictions:
        result = coco.loadRes(predictions)
    else:
        result = COCO()
        result.dataset = {**gt, "annotations": []}
        result.createIndex()
    evaluator = COCOeval(coco, result, "bbox")
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.evaluate()
    evaluator.accumulate()
    precision_grid = evaluator.eval["precision"][:, :, :, 0, -1]
    avg = lambda x: float(x[x > -1].mean()) if (x > -1).any() else None
    precision = np.divide(tp, tp + fp, out=np.zeros(14), where=tp + fp > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(14), where=tp + fn > 0)
    per_class = [
        {
            "class_id": i,
            "name": names[i],
            "gt_count": int(support[i]),
            "prediction_count_raw": int(raw_counts[i]),
            "prediction_count_fixed": int(kept_counts[i]),
            "tp": int(tp[i]),
            "fp": int(fp[i]),
            "fn": int(fn[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": harmonic(precision[i], recall[i]),
            "ap50": avg(precision_grid[0, :, i]),
            "ap50_95": avg(precision_grid[:, :, i]),
        }
        for i in range(14)
    ]
    present = support > 0
    p = float(precision[present].mean()) if present.any() else 0.0
    r = float(recall[present].mean()) if present.any() else 0.0
    summary = {
        "precision": p,
        "recall": r,
        "harmonic_aggregate_f1": harmonic(p, r),
        "macro_class_f1": float(np.mean([c["f1"] for c in per_class if c["gt_count"]]))
        if present.any()
        else 0.0,
        "map50": avg(precision_grid[0]),
        "map50_95": avg(precision_grid),
        "images": len(records),
        "objects": int(support.sum()),
        "predictions_raw": int(raw_counts.sum()),
        "predictions_fixed": int(kept_counts.sum()),
        "confidence": confidence,
        "matching_iou": MATCH_IOU,
        "ap_score_floor": 0.001,
        "nms_iou": "detector-specific; see inference configuration",
        "max_detections": 300,
        "fixed_method": "Same-class score-descending greedy IoU>=.5; macro over represented GT classes",
        "confusion_method": "Rows GT, columns prediction, last row/column background; separate class-agnostic score-descending IoU>=.5 matching",
    }
    return summary, per_class, confusion, precision_grid, image_errors
