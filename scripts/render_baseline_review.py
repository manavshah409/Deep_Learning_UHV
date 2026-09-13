"""Render paired ground-truth/prediction images from saved diagnostic evidence."""

import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2"
OUT = ROOT / "reports/predictions/error_analysis"


def main():
    records = json.loads((OUT / "diagnostic_cases_full.json").read_text())
    manifest = {
        r["image_id"]: r for r in json.loads((DATA / "val_manifest.json").read_text())
    }
    import yaml

    names = yaml.safe_load((DATA / "dataset.yaml").read_text())["names"]
    missed = []
    for r in records:
        with Image.open(DATA / manifest[r["image_id"]]["image"]) as im:
            scale = 640 / max(im.size)
        for i in r["unmatched_gt"]:
            b = r["gt_boxes"][i]
            missed.append(((b[2] - b[0]) * (b[3] - b[1]) * scale**2, r["image_id"], i))
    print("Smallest missed boxes at 640 scale:", sorted(missed)[:8])
    simple = max(
        (r for r in records if r["gt_count"] <= 10),
        key=lambda r: r["correct_matches"] - len(r["unmatched_predictions"]),
    )
    ids = list(
        dict.fromkeys(
            [1364, 21621, 4711, 954, sorted(missed)[0][1], simple["image_id"]]
        )
    )
    for r in records:
        if r["image_id"] not in ids:
            continue
        gt = Image.open(DATA / manifest[r["image_id"]]["image"]).convert("RGB")
        draw = ImageDraw.Draw(gt)
        for i, (b, c) in enumerate(zip(r["gt_boxes"], r["gt_classes"])):
            color = "red" if i in r["unmatched_gt"] else "lime"
            draw.rectangle(b, outline=color, width=3)
            draw.text(
                (b[0], max(0, b[1] - 12)),
                f"{i} {names[c]}",
                fill=color,
                stroke_width=1,
                stroke_fill="black",
            )
        pred = Image.open(OUT / r["prediction_image"]).convert("RGB")
        width = 1200
        height = round(gt.height * width / gt.width)
        pair = Image.new("RGB", (width, height * 2 + 64), "white")
        d = ImageDraw.Draw(pair)
        d.text(
            (12, 8),
            f"VAL {r['image_id']} GROUND TRUTH | red=unmatched GT at IoU .5",
            fill="black",
        )
        pair.paste(gt.resize((width, height)), (0, 28))
        d.text(
            (12, height + 36),
            f"PREDICTIONS conf .10 | GT {r['gt_count']} correct-class matches {r['correct_matches']} unmatched GT {len(r['unmatched_gt'])} unmatched predictions {len(r['unmatched_predictions'])}",
            fill="black",
        )
        pair.paste(pred.resize((width, height)), (0, height + 64))
        pair.save(OUT / f"review_pair_{r['image_id']}.jpg", quality=93)
    print("Paired cases", ids)


if __name__ == "__main__":
    main()
