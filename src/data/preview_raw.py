"""Early visual QA on available images; does not certify full acquisition/conversion."""

from collections import defaultdict
import random
import yaml
from PIL import Image, ImageDraw
from .common import (
    paths,
    annotation_paths,
    load_coco,
    convert_box,
    category_mapping,
    save_json,
    ROOT,
)
from .render import draw_box


def main():
    cfg = paths()
    candidates = []
    annotations = {}
    categories = None
    for split, path in annotation_paths(cfg["raw"]).items():
        d = load_coco(path)
        cats = category_mapping(d["categories"])
        if categories is None:
            categories = cats
        byname = {
            p.name: p
            for p in (cfg["raw"] / f"UVH-26-{split.title()}" / "data").rglob("*.png")
        }
        byid = defaultdict(list)
        for ann in d["annotations"]:
            byid[ann["image_id"]].append(ann)
        for im in d["images"]:
            if im["file_name"] in byname:
                key = (split, im["id"])
                annotations[key] = byid[im["id"]]
                candidates.append(
                    {"split": split, **im, "path": byname[im["file_name"]]}
                )
    rng = random.Random(42)
    candidates.sort(key=lambda r: (r["split"], r["id"]))
    chosen = rng.sample(candidates, min(20, len(candidates)))
    covered = set()
    reasons = {(r["split"], r["id"]): "seed42 random available image" for r in chosen}
    for r in candidates:
        key = (r["split"], r["id"])
        cs = {ann["category_id"] for ann in annotations[key]}
        if cs - covered:
            chosen.append(r)
            reasons[key] = "class coverage"
            covered |= cs
    for r, reason in [
        (
            max(candidates, key=lambda r: len(annotations[(r["split"], r["id"])])),
            "dense available",
        ),
        (
            min(candidates, key=lambda r: len(annotations[(r["split"], r["id"])])),
            "sparse available",
        ),
    ]:
        chosen.append(r)
        reasons[(r["split"], r["id"])] = reason
    names = {c["original_id"]: c["name"] for c in categories}
    chosen = list({(r["split"], r["id"]): r for r in chosen}.values())
    out = cfg["reports"] / "annotation_samples/early"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for r in chosen:
        key = (r["split"], r["id"])
        with Image.open(r["path"]) as image:
            im = image.convert("RGB")
        if im.size != (r["width"], r["height"]):
            raise ValueError("Image dimensions mismatch")
        draw = ImageDraw.Draw(im)
        for ann in annotations[key]:
            cx, cy, w, h = convert_box(ann["bbox"], r["width"], r["height"])
            box = (
                (cx - w / 2) * im.width,
                (cy - h / 2) * im.height,
                (cx + w / 2) * im.width,
                (cy + h / 2) * im.height,
            )
            draw_box(
                draw, box, names[ann["category_id"]], ann["category_id"] - 1, len(names)
            )
        im.thumbnail((1280, 720))
        name = f"{r['split']}_{r['id']}.jpg"
        im.save(out / name, quality=90)
        records.append(
            {
                "split": r["split"],
                "image_id": r["id"],
                "filename": name,
                "reason": reasons[key],
                "classes": sorted(
                    {names[ann["category_id"]] for ann in annotations[key]}
                ),
                "objects": len(annotations[key]),
            }
        )
    save_json(
        out / "index.json",
        {
            "available_images": len(candidates),
            "review_samples": records,
            "note": "Early subset of downloaded images, not final converted-dataset review.",
        },
    )
    import pandas as pd

    pd.DataFrame(categories).to_csv(
        cfg["reports"] / "tables/class_mapping.csv", index=False
    )
    (ROOT / "configs/class_mapping.yaml").write_text(
        yaml.safe_dump(categories, sort_keys=False)
    )
    print(
        "available", len(candidates), "rendered", len(chosen), "classes", len(covered)
    )


if __name__ == "__main__":
    main()
