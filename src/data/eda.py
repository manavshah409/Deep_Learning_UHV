"""Generate EDA only from executed annotation reads; no synthetic dataset metrics."""

import argparse
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import paths, annotation_paths, load_coco, convert_box, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paths.local.yaml")
    a = p.parse_args()
    cfg = paths(a.config)
    images = []
    boxes = []
    categories = {}
    for split, path in annotation_paths(cfg["raw"]).items():
        d = load_coco(path)
        idx = {r["id"]: r for r in d["images"]}
        categories.update({r["id"]: r["name"] for r in d["categories"]})
        counts = Counter(r["image_id"] for r in d["annotations"])
        images.extend(
            {
                "split": split,
                "id": r["id"],
                "width": r["width"],
                "height": r["height"],
                "objects": counts[r["id"]],
            }
            for r in d["images"]
        )
        for ann in d["annotations"]:
            im = idx.get(ann["image_id"])
            if im is None:
                continue
            try:
                cx, cy, nw, nh = convert_box(ann["bbox"], im["width"], im["height"])
            except ValueError:
                continue
            x, y, w, h = ann["bbox"]
            boxes.append(
                {
                    "split": split,
                    "category": categories[ann["category_id"]],
                    "width": w,
                    "height": h,
                    "area": w * h,
                    "relative_area": nw * nh,
                    "aspect": w / h,
                    "cx": cx,
                    "cy": cy,
                }
            )
    ims = pd.DataFrame(images)
    b = pd.DataFrame(boxes)
    out = cfg["reports"] / "figures"
    out.mkdir(parents=True, exist_ok=True)

    def plot(name, title, draw, xlabel="", ylabel="Count"):
        fig, ax = plt.subplots(figsize=(10, 5))
        draw(ax)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=160)
        plt.close(fig)

    plot(
        "split_distribution",
        "Official MV image counts",
        lambda ax: ims.groupby("split").size().plot.bar(ax=ax, rot=0),
    )
    plot(
        "class_distribution",
        "Valid MV instances by class",
        lambda ax: b.category.value_counts().plot.bar(ax=ax, rot=45),
    )
    plot(
        "objects_per_image",
        "Objects per image (all MV annotations)",
        lambda ax: ax.hist(ims.objects, bins=60),
        "Objects",
    )
    plot(
        "resolution_distribution",
        "Image dimensions from annotation metadata",
        lambda ax: ims.groupby(["width", "height"]).size().plot.bar(ax=ax, rot=45),
    )
    for name, column, title in [
        ("bbox_area_distribution", "area", "Box area in original pixels²"),
        ("bbox_aspect_ratio", "aspect", "Box aspect ratio"),
        ("bbox_width_distribution", "width", "Box width in original pixels"),
        ("bbox_height_distribution", "height", "Box height in original pixels"),
    ]:
        plot(
            name,
            title,
            lambda ax, c=column: (
                ax.hist(
                    b[c],
                    bins=np.geomspace(max(b[c].min(), 1e-6), b[c].max() + 1e-6, 60),
                ),
                ax.set_xscale("log"),
            ),
            column,
        )

    def heat(ax):
        h = ax.hist2d(b.cx, b.cy, bins=50, cmap="magma")
        ax.invert_yaxis()
        plt.colorbar(h[3], ax=ax, label="Instances")

    plot(
        "object_center_heatmap",
        "Normalized object centres",
        heat,
        "x centre",
        "y centre",
    )
    dist = b.groupby(["split", "category"]).size().rename("instances").reset_index()
    dist["share_percent"] = dist.groupby("split")["instances"].transform(
        lambda x: 100 * x / x.sum()
    )
    dist.to_csv(cfg["reports"] / "tables/eda_class_distribution.csv", index=False)
    size = {
        "small": int((b.area < 32**2).sum()),
        "medium": int(((b.area >= 32**2) & (b.area < 96**2)).sum()),
        "large": int((b.area >= 96**2).sum()),
    }
    save_json(
        cfg["reports"] / "tables/eda_summary.json",
        {
            "images": len(ims),
            "valid_boxes": len(b),
            "objects_per_image": ims.objects.describe().to_dict(),
            "box_area_pixels": b.area.describe().to_dict(),
            "size_counts_original_resolution": size,
            "size_definition": "small <32², medium 32² to <96², large >=96² original pixels; not resized training pixels",
            "classes": b.category.value_counts().to_dict(),
        },
    )
    print("EDA generated:", len(ims), "images;", len(b), "valid boxes")


if __name__ == "__main__":
    main()
