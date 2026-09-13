"""Render seeded and targeted original/converted annotation samples locally."""

import argparse
import json
import random
from PIL import Image, ImageDraw
import yaml
from .common import paths, save_json
from .render import draw_box


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paths.local.yaml")
    p.add_argument("--dataset-version", default="uvh26_mv_yolo_v1")
    a = p.parse_args()
    cfg = paths(a.config)
    root = cfg["processed"] / a.dataset_version
    rows = json.loads((root / "manifest.json").read_text())
    names = yaml.safe_load((root / "dataset.yaml").read_text())["names"]
    annotated = [r for r in rows if r["objects"]]
    rng = random.Random(42)
    chosen = rng.sample(annotated, min(20, len(annotated)))
    reasons = {(r["split"], r["image_id"]): "seed42 random" for r in chosen}
    for r, reason in [
        (max(annotated, key=lambda r: r["objects"]), "dense"),
        (min(annotated, key=lambda r: r["objects"]), "sparse"),
    ]:
        chosen.append(r)
        reasons[(r["split"], r["image_id"])] = reason
    covered = set()
    smallest = (float("inf"), None)
    for r in annotated:
        labels = [
            list(map(float, l.split()))
            for l in (root / r["label"]).read_text().splitlines()
        ]
        cs = {int(l[0]) for l in labels}
        new = cs - covered
        if new:
            chosen.append(r)
            reasons[(r["split"], r["image_id"])] = f"class coverage {sorted(new)}"
            covered |= cs
        area = min(l[3] * l[4] for l in labels)
        if area < smallest[0]:
            smallest = (area, r)
    if smallest[1]:
        chosen.append(smallest[1])
        reasons[(smallest[1]["split"], smallest[1]["image_id"])] = (
            "smallest normalized box"
        )
    out = cfg["reports"] / "annotation_samples"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    unique = {(r["split"], r["image_id"]): r for r in chosen}
    for key, r in unique.items():
        with Image.open(root / r["image"]) as raw:
            im = raw.convert("RGB")
        draw = ImageDraw.Draw(im)
        for line in (root / r["label"]).read_text().splitlines():
            cid, cx, cy, w, h = map(float, line.split())
            x0 = (cx - w / 2) * im.width
            y0 = (cy - h / 2) * im.height
            x1 = (cx + w / 2) * im.width
            y1 = (cy + h / 2) * im.height
            draw_box(draw, (x0, y0, x1, y1), names[int(cid)], int(cid), len(names))
        im.thumbnail((1280, 720))
        filename = f"{r['split']}_{r['image_id']}.jpg"
        im.save(out / filename, quality=88)
        records.append(
            {
                "split": r["split"],
                "image_id": r["image_id"],
                "file": filename,
                "reason": reasons[key],
            }
        )
    save_json(out / "index.json", records)
    print(len(records), "samples in", out)


if __name__ == "__main__":
    main()
