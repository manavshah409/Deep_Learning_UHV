"""COCOeval area AP from original-image frozen YOLO boxes, maxDets=300."""

import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image


def area_groups(area):
    # Exact COCOeval inclusive endpoints; a boundary object may enter two bins.
    return [
        name
        for name, lo, hi in [
            ("small", 0, 32**2),
            ("medium", 32**2, 96**2),
            ("large", 96**2, 1e10),
        ]
        if lo <= area <= hi
    ]


def evaluate_sizes(root, predictions, names, prediction_class_map=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    rows = json.loads((root / "val_manifest.json").read_text())
    gt = {
        "info": {},
        "images": [],
        "annotations": [],
        "categories": [{"id": int(i), "name": n} for i, n in names.items()],
    }
    ids = {}
    counts = {
        name: np.zeros(len(names), dtype=int) for name in ["small", "medium", "large"]
    }
    for row in rows:
        filename = Path(row["image"]).name
        assert filename not in ids
        ids[filename] = row["image_id"]
        with Image.open(root / row["image"]) as im:
            w, h = im.size
        gt["images"].append({"id": row["image_id"], "width": w, "height": h})
        for line in (root / row["label"]).read_text().splitlines():
            c, x, y, bw, bh = map(float, line.split())
            bw *= w
            bh *= h
            area = bw * bh
            gt["annotations"].append(
                dict(
                    id=len(gt["annotations"]) + 1,
                    image_id=row["image_id"],
                    category_id=int(c),
                    bbox=[x * w - bw / 2, y * h - bh / 2, bw, bh],
                    area=area,
                    iscrowd=0,
                )
            )
            for group in area_groups(area):
                counts[group][int(c)] += 1
    preds = copy.deepcopy(predictions)
    mapping = (
        list(range(len(names)))
        if prediction_class_map is None
        else prediction_class_map
    )
    if len(set(mapping)) != len(names):
        raise ValueError("Prediction class map must cover frozen classes exactly")
    reverse = {source: target for target, source in enumerate(mapping)}
    for p in preds:
        p["image_id"] = ids[p.pop("file_name")]
        if p["category_id"] not in reverse:
            raise ValueError("Unmapped prediction category")
        p["category_id"] = reverse[p["category_id"]]
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    det = coco.loadRes(preds)
    ev = COCOeval(coco, det, "bbox")
    ev.params.imgIds = sorted(ids.values())
    ev.params.catIds = list(range(len(names)))
    ev.params.maxDets = [1, 10, 300]
    ev.evaluate()
    ev.accumulate()
    precision = ev.eval["precision"]

    def ap(values):
        values = values[values > -1]
        return float(values.mean()) if values.size else None

    result = {
        "prediction_class_map": mapping,
        "method": "pycocotools COCOeval original-image bbox area; GT reconstructed from frozen YOLO labels. Standard COCO ignored-GT/out-of-area-unmatched-prediction logic. maxDets [1,10,300], IoU .50:.05:.95, 101 recall points. Not identical AP integration to Ultralytics overall AP.",
        "area_bounds_pixels_squared": dict(
            zip(ev.params.areaRngLbl, ev.params.areaRng)
        ),
        "boundary_note": "COCOeval includes endpoints; exact 1024/9216-area boxes can occur in adjacent bins.",
        "groups": {},
    }
    for index, name in enumerate(ev.params.areaRngLbl):
        counts_i = (
            np.array(
                [
                    sum(a["category_id"] == c for a in gt["annotations"])
                    for c in range(len(names))
                ]
            )
            if name == "all"
            else counts[name]
        )
        result["groups"][name] = {
            "objects": int(counts_i.sum()),
            "ap50_95": ap(precision[:, :, :, index, -1]),
            "ap50": ap(precision[0, :, :, index, -1]),
            "per_class": [
                dict(
                    class_id=c,
                    name=names[c],
                    objects=int(counts_i[c]),
                    ap50_95=ap(precision[:, :, c, index, -1]),
                    ap50=ap(precision[0, :, c, index, -1]),
                    status="measured"
                    if counts_i[c]
                    else "not_applicable_no_ground_truth",
                )
                for c in range(len(names))
            ],
        }
    return result
