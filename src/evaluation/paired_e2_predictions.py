"""Same-image GT/E0/E1 diagnostic review; all imagery remains local and ignored."""

import argparse
import json
from PIL import Image, ImageDraw
from src.data.common import ROOT, save_json, sha256
from src.evaluation.error_analysis import match_predictions

IDS = [4232, 10518, 1364, 21621, 4711, 954]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    a = p.parse_args()
    from ultralytics import YOLO
    import torch

    assert torch.backends.mps.is_available()
    data = ROOT / "data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2"
    rows = {
        r["image_id"]: r for r in json.loads((data / "val_manifest.json").read_text())
    }
    out = ROOT / "reports/predictions/E2_640_vs_960_paired_v1"
    if out.exists():
        raise ValueError("Paired output exists; preserve it")
    out.mkdir(parents=True)
    models = [YOLO(a.weights), YOLO(a.weights)]
    assert models[0].names == models[1].names
    details = []
    summary = []
    for cid in IDS:
        row = rows[cid]
        original = Image.open(data / row["image"]).convert("RGB")
        w, h = original.size
        gt = []
        classes = []
        for line in (data / row["label"]).read_text().splitlines():
            c, x, y, bw, bh = map(float, line.split())
            classes.append(int(c))
            gt.append(
                [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
            )
        panels = []
        case = dict(image_id=cid, gt_count=len(gt))
        detail = dict(image_id=cid, gt_boxes=gt, gt_classes=classes)
        for label, model in [("GT", None), *zip(["640", "960"], models)]:
            panel = original.copy()
            draw = ImageDraw.Draw(panel)
            if model is None:
                boxes = gt
                cs = classes
                confs = [None] * len(gt)
            else:
                result = model.predict(
                    str(data / row["image"]),
                    device="mps",
                    imgsz=int(label),
                    batch=1,
                    conf=0.10,
                    iou=0.7,
                    max_det=300,
                    verbose=False,
                    save=False,
                )[0]
                boxes = result.boxes.xyxy.cpu().numpy()
                cs = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                matches, fn, fp = match_predictions(gt, classes, boxes, cs)
                case[label] = dict(
                    predictions=len(boxes),
                    correct_matches=sum(m["correct_class"] for m in matches),
                    unmatched_gt=len(fn),
                    unmatched_predictions=len(fp),
                    class_confusions=[
                        dict(
                            truth=model.names[classes[m["gt"]]],
                            predicted=model.names[int(cs[m["prediction"]])],
                        )
                        for m in matches
                        if not m["correct_class"]
                    ],
                )
                detail[label] = dict(
                    boxes=boxes.tolist(),
                    classes=cs.tolist(),
                    confidences=confs.tolist(),
                    matches=matches,
                    unmatched_gt=fn,
                    unmatched_predictions=fp,
                )
            for i, (box, c, conf) in enumerate(zip(boxes, cs, confs)):
                color = (
                    "lime"
                    if label == "GT"
                    else ("cyan" if label == "640" else "orange")
                )
                draw.rectangle(list(box), outline=color, width=2)
                name = models[0].names[int(c)]
                txt = f"{i}:{name}" if conf is None else f"{i}:{name} {float(conf):.2f}"
                draw.text(
                    (float(box[0]), max(0, float(box[1]) - 12)),
                    txt,
                    fill=color,
                    stroke_width=1,
                    stroke_fill="black",
                )
            panels.append(panel)
        width = 1280
        ph = round(h * width / w)
        sheet = Image.new("RGB", (width, 3 * (ph + 25)), "white")
        d = ImageDraw.Draw(sheet)
        for i, (label, panel) in enumerate(
            zip(["GT", "E1 YOLOv8s at 640", "E1 YOLOv8s at 960"], panels)
        ):
            y = i * (ph + 25)
            d.text(
                (8, y + 6),
                f"Validation {cid} | {label} | prediction conf .10, NMS .7",
                fill="black",
            )
            sheet.paste(panel.resize((width, ph)), (0, y + 25))
        sheet.save(out / f"val_{cid}_GT_640_960.jpg", quality=95)
        details.append(detail)
        summary.append(case)
    save_json(out / "detailed_predictions.json", details)
    save_json(
        ROOT / "reports/comparisons/E2_paired_summary.json",
        dict(
            status="awaiting_manual_review",
            image_ids=IDS,
            validation_manifest_sha256=sha256(data / "val_manifest.json"),
            weights_sha256=sha256(a.weights),
            confidence=0.10,
            nms_iou=0.7,
            matching_iou=0.5,
            method="Same six Phase 1 diagnostic validation scenes; class-agnostic greedy IoU matching. Not an unbiased error-rate estimate. Source-label limitations are assessed separately.",
            cases=summary,
        ),
    )
    print("Generated six GT/640/960 review images")


if __name__ == "__main__":
    main()
