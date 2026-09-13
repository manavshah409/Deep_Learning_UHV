"""Generate frozen-manifest review evidence; passing requires separate visual review."""

import json
import random
from PIL import Image, ImageDraw
import yaml
from .common import ROOT, save_json, sha256
from .render import draw_box


def main():
    import argparse

    global DEST
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root")
    p.add_argument("--recovery", default="reports/audit/baseline_subset_recovery.json")
    a = p.parse_args()
    if a.dataset_root:
        DEST = (ROOT / a.dataset_root).resolve()
    rows = json.loads((DEST / "manifest.json").read_text())
    names = yaml.safe_load((DEST / "dataset.yaml").read_text())["names"]
    labels = {}
    for row in rows:
        labels[(row["split"], row["image_id"])] = [
            list(map(float, l.split()))
            for l in (DEST / row["label"]).read_text().splitlines()
        ]
    selected = {}

    def add(row, reason):
        key = (row["split"], row["image_id"])
        if key not in selected:
            selected[key] = dict(row=row, reasons=[])
        selected[key]["reasons"].append(reason)

    for s in ["train", "val"]:
        split = [r for r in rows if r["split"] == s]
        for r in random.Random(42).sample(split, 16):
            add(r, "seed42 random " + s)
        add(max(split, key=lambda r: r["objects"]), "densest " + s)
        add(min(split, key=lambda r: r["objects"]), "sparsest " + s)

    def get(r):
        return labels[(r["split"], r["image_id"])]

    add(
        min(rows, key=lambda r: min((l[3] * l[4] for l in get(r)), default=1)),
        "smallest normalized object",
    )
    add(
        min(
            rows,
            key=lambda r: min(
                (
                    min(
                        l[1] - l[3] / 2,
                        l[2] - l[4] / 2,
                        1 - l[1] - l[3] / 2,
                        1 - l[2] - l[4] / 2,
                    )
                    for l in get(r)
                ),
                default=1,
            ),
        ),
        "nearest coordinate boundary",
    )
    covered = {int(l[0]) for v in selected.values() for l in get(v["row"])}
    for c in names:
        if c not in covered:
            r = next(r for r in rows if any(int(l[0]) == c for l in get(r)))
            add(r, f"class coverage {names[c]}")
            covered |= {int(l[0]) for l in get(r)}
    for c in [9, 10, 13]:
        add(
            max(rows, key=lambda r: sum(int(l[0]) == c for l in get(r))),
            f"rare class {names[c]} exposure",
        )
    recovery = json.loads((ROOT / a.recovery).read_text())
    for decision in recovery["replacements"]:
        rep = decision["replacement"]
        if not any(
            r["split"] == rep["split"] and r["image_id"] == rep["image_id"]
            for r in rows
        ):
            continue
        add(
            next(
                r
                for r in rows
                if r["split"] == rep["split"] and r["image_id"] == rep["image_id"]
            ),
            "audited replacement",
        )
    out = ROOT / f"reports/annotation_samples/final_{DEST.name}"
    if out.exists():
        raise ValueError("Review images already exist")
    out.mkdir(parents=True)
    records = []
    for key, item in selected.items():
        r = item["row"]
        with Image.open(DEST / r["image"]) as original:
            im = original.convert("RGB")
        draw = ImageDraw.Draw(im)
        for c, x, y, w, h in get(r):
            draw_box(
                draw,
                (
                    (x - w / 2) * im.width,
                    (y - h / 2) * im.height,
                    (x + w / 2) * im.width,
                    (y + h / 2) * im.height,
                ),
                names[int(c)],
                int(c),
            )
        filename = f"{r['split']}_{r['image_id']}.jpg"
        im.save(out / filename, quality=92)
        records.append(
            dict(
                split=r["split"],
                image_id=r["image_id"],
                file=filename,
                reasons=item["reasons"],
                classes=sorted({int(l[0]) for l in get(r)}),
            )
        )
    sheets = []
    for start in range(0, len(records), 4):
        canvas = Image.new("RGB", (1920, 1160), "white")
        draw = ImageDraw.Draw(canvas)
        for i, r in enumerate(records[start : start + 4]):
            with Image.open(out / r["file"]) as im:
                im.thumbnail((960, 540))
                x = (i % 2) * 960
                y = (i // 2) * 580
                canvas.paste(im, (x, y + 40))
                draw.text(
                    (x + 8, y + 10),
                    r["file"] + " | " + ", ".join(r["reasons"]),
                    fill="black",
                )
        filename = f"contact_{start // 4 + 1:02d}.jpg"
        canvas.save(out / filename, quality=90)
        sheets.append(str((out / filename).relative_to(ROOT)))
    save_json(
        out / "index.json",
        dict(
            manifest_sha256=sha256(DEST / "manifest.json"),
            images=records,
            sheets=sheets,
            classes=sorted(covered),
        ),
    )
    print(len(records), "images", len(sheets), "contact sheets")


if __name__ == "__main__":
    main()
