"""Match predictions to labels at IoU .5, producing evidence for manual analysis."""

import argparse
from pathlib import Path
import json
import random
import numpy as np
from src.data.common import ROOT, save_json


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    left = np.maximum(a[:, None, :2], b[None, :, :2])
    right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(right - left, 0).prod(2)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)
    bb = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-12)


def match_predictions(gt_boxes, gt_classes, pred_boxes, pred_classes, threshold=0.5):
    ious = iou_matrix(
        np.array(gt_boxes).reshape(-1, 4), np.array(pred_boxes).reshape(-1, 4)
    )
    candidates = sorted(
        zip(*np.where(ious >= threshold)),
        key=lambda pair: (-ious[pair], pair[0], pair[1]),
    )
    used_gt = set()
    used_pred = set()
    matches = []
    for g, p in candidates:
        if g in used_gt or p in used_pred:
            continue
        used_gt.add(g)
        used_pred.add(p)
        matches.append(
            {
                "gt": int(g),
                "prediction": int(p),
                "iou": float(ious[g, p]),
                "correct_class": int(gt_classes[g]) == int(pred_classes[p]),
            }
        )
    return (
        matches,
        sorted(set(range(len(gt_boxes))) - used_gt),
        sorted(set(range(len(pred_boxes))) - used_pred),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--count", type=int, default=100)
    a = p.parse_args()
    from ultralytics import YOLO

    model = YOLO(a.weights)
    root = Path(a.dataset_root)
    rows = [r for r in json.loads(Path(a.manifest).read_text()) if r["split"] == "val"]
    rng = random.Random(42)
    rng.shuffle(rows)
    # Random sample plus extremes and class coverage; this is diagnostic, not overall evaluation.
    chosen = rows[: a.count] + [
        max(rows, key=lambda r: r["objects"]),
        min(rows, key=lambda r: r["objects"]),
    ]
    covered = set()
    for r in rows:
        cs = {int(l.split()[0]) for l in (root / r["label"]).read_text().splitlines()}
        if cs - covered:
            chosen.append(r)
            covered |= cs
    chosen = list({r["image_id"]: r for r in chosen}.values())
    out = ROOT / "reports/predictions/error_analysis"
    if out.exists():
        raise ValueError("Diagnostic output already exists; preserve prior evidence")
    out.mkdir(parents=True)
    records = []
    for row in chosen:
        result = model.predict(
            str(root / row["image"]),
            conf=0.10,
            device=a.device,
            imgsz=640,
            iou=0.7,
            max_det=300,
            verbose=False,
        )[0]
        h, w = result.orig_shape
        gt = []
        classes = []
        for line in (root / row["label"]).read_text().splitlines():
            c, x, y, bw, bh = map(float, line.split())
            classes.append(int(c))
            gt.append(
                [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
            )
        pb = result.boxes.xyxy.cpu().numpy()
        pc = result.boxes.cls.cpu().numpy()
        matches, fn, fp = match_predictions(gt, classes, pb, pc)
        filename = f"val_{row['image_id']}.jpg"
        result.save(filename=str(out / filename))
        records.append(
            {
                "image_id": row["image_id"],
                "source": row["source"],
                "prediction_image": filename,
                "gt_count": len(gt),
                "gt_boxes": gt,
                "gt_classes": classes,
                "pred_boxes": pb.tolist(),
                "pred_classes": pc.astype(int).tolist(),
                "pred_confidences": result.boxes.conf.cpu().tolist(),
                "prediction_count": len(pb),
                "correct_matches": sum(m["correct_class"] for m in matches),
                "class_confusions": [
                    {
                        "truth": model.names[classes[m["gt"]]],
                        "predicted": model.names[int(pc[m["prediction"]])],
                    }
                    for m in matches
                    if not m["correct_class"]
                ],
                "unmatched_gt": fn,
                "unmatched_predictions": fp,
                "matches": matches,
                "confidence_threshold": 0.10,
                "iou_threshold": 0.5,
                "low_confidence_predictions": int(
                    (result.boxes.conf.cpu().numpy() < 0.25).sum()
                ),
                "note": "Class-agnostic greedy IoU matching for diagnostic evidence; not COCO AP matching. Occlusion requires manual review.",
            }
        )
    save_json(out / "diagnostic_cases_full.json", records)
    detailed = {
        "gt_boxes",
        "gt_classes",
        "pred_boxes",
        "pred_classes",
        "pred_confidences",
    }
    save_json(
        ROOT / "reports/error_analysis/diagnostic_cases.json",
        [{k: v for k, v in row.items() if k not in detailed} for row in records],
    )
    print("Diagnostic cases:", len(records))


if __name__ == "__main__":
    main()
