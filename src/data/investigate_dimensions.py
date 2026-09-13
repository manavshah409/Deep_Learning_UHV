"""Diagnostic hypotheses only; never repair source images or annotations."""

import json
from PIL import Image, ImageDraw
from .common import ROOT, paths, annotation_paths, load_coco, save_json, sha256
from .render import draw_box


def main():
    cfg = paths()
    selection = json.loads(
        (
            ROOT
            / "data/interim/subset_plans/uvh26_mv_baseline_subset_v1/selection.json"
        ).read_text()
    )
    row = next(r for r in selection if r["split"] == "train" and r["image_id"] == 21818)
    data = load_coco(annotation_paths(cfg["raw"])["train"])
    meta = next(r for r in data["images"] if r["id"] == 21818)
    anns = [a for a in data["annotations"] if a["image_id"] == 21818]
    names = {c["id"]: c["name"] for c in data["categories"]}
    source = cfg["raw"] / row["source"]
    with Image.open(source) as im:
        raw = im.convert("RGB")
    out = ROOT / "reports/annotation_samples/dimension_investigation"
    out.mkdir(parents=True, exist_ok=True)
    existing = list((ROOT / "data/processed").rglob(source.stem + ".txt"))
    outputs = []
    for mode in ["raw_coco", "scaled_width", "yolo_metadata_roundtrip_hypothesis"]:
        im = raw.copy()
        draw = ImageDraw.Draw(im)
        for a in anns:
            x, y, w, h = a["bbox"]
            if mode != "raw_coco":
                factor = raw.width / meta["width"]
                x *= factor
                w *= factor
            draw_box(
                draw,
                (x, y, x + w, y + h),
                f"{a['id']} {names[a['category_id']]}",
                a["category_id"] - 1,
            )
        draw.rectangle((0, 0, im.width, 32), fill="black")
        draw.text(
            (8, 8),
            mode
            + (
                " (no existing label; simulated metadata normalization)"
                if mode.startswith("yolo")
                else ""
            ),
            fill="white",
        )
        p = out / (mode + ".jpg")
        im.save(p, quality=94)
        outputs.append(str(p.relative_to(ROOT)))
    save_json(
        ROOT / "reports/audit/803489_dimension_investigation.json",
        dict(
            image_id=21818,
            split="train",
            source=row["source"],
            selected=True,
            source_sha256=sha256(source),
            annotation_dimensions=[meta["width"], meta["height"]],
            actual_dimensions=list(raw.size),
            boxes=[
                {
                    "annotation_id": a["id"],
                    "category": names[a["category_id"]],
                    "bbox": a["bbox"],
                    "right_boundary": a["bbox"][0] + a["bbox"][2],
                    "exceeds_actual_width": a["bbox"][0] + a["bbox"][2] > raw.width,
                }
                for a in anns
            ],
            existing_yolo_labels=[str(p.relative_to(ROOT)) for p in existing],
            outputs=outputs,
            third_overlay="No existing converted label for this image. Explicitly simulated metadata-normalized YOLO roundtrip; equivalent to width scaling, not independent evidence.",
            decision="pending_visual_inspection",
        ),
    )
    print(
        "boxes",
        len(anns),
        "rightmost",
        max(a["bbox"][0] + a["bbox"][2] for a in anns),
        "existing_labels",
        existing,
    )
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
