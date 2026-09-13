"""Audit COCO references, geometry and every image's decoding/dimensions/hash."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from PIL import Image
from .common import (
    paths,
    annotation_paths,
    load_coco,
    convert_box,
    sha256,
    save_json,
    REVISION,
)


def duplicates(rows, key):
    return [k for k, n in Counter(r.get(key) for r in rows).items() if n > 1]


def check_image(item):
    path, meta = item
    out = {
        "image_id": meta["id"],
        "filename": meta["file_name"],
        "width": None,
        "height": None,
        "sha256": None,
        "error": None,
    }
    if path is None:
        out["error"] = "missing_image"
        return out
    try:
        with Image.open(path) as im:
            im.load()
            out.update(width=im.width, height=im.height)
            if im.size != (meta["width"], meta["height"]):
                out["error"] = "dimension_mismatch"
        out["sha256"] = sha256(path)
    except Exception as e:
        out["error"] = f"unreadable:{type(e).__name__}"
    return out


def audit(config, annotations_only=False):
    cfg = paths(config)
    out = cfg["reports"] / "audit"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "revision": REVISION,
        "image_audit_executed": not annotations_only,
        "splits": {},
    }
    invalid = []
    missing = []
    class_rows = []
    resolutions = []
    all_names = {}
    all_ids = {}
    all_hashes = {}
    for split, path in annotation_paths(cfg["raw"]).items():
        d = load_coco(path)
        ims = {r["id"]: r for r in d["images"] if "id" in r}
        cats = {r["id"]: r.get("name") for r in d["categories"] if "id" in r}
        errors = []
        counts = Counter()
        per_image = Counter()
        for group, fields in [
            ("images", ("id", "file_name", "width", "height")),
            ("annotations", ("id", "image_id", "category_id", "bbox")),
            ("categories", ("id", "name")),
        ]:
            for i, r in enumerate(d[group]):
                for field in fields:
                    if field not in r:
                        errors.append({"group": group, "index": i, "missing": field})
        if errors:
            result["splits"][split] = {
                "missing_fields": errors,
                "status": "schema_blocked",
            }
            save_json(out / "schema_failure.json", result)
            raise ValueError(
                f"Missing mandatory fields in {split}; see schema_failure.json"
            )
        for ann in d["annotations"]:
            reason = None
            im = ims.get(ann.get("image_id"))
            if im is None:
                reason = "missing_image_reference"
                missing.append(
                    {
                        "split": split,
                        "annotation_id": ann.get("id"),
                        "image_id": ann.get("image_id"),
                        "reason": reason,
                    }
                )
            elif ann.get("category_id") not in cats:
                reason = "unknown_category"
            else:
                try:
                    convert_box(ann.get("bbox"), im.get("width"), im.get("height"))
                except (ValueError, TypeError) as e:
                    reason = str(e)
            if reason:
                invalid.append(
                    {
                        "split": split,
                        "annotation_id": ann.get("id"),
                        "image_id": ann.get("image_id"),
                        "reason": reason,
                        "bbox": ann.get("bbox"),
                    }
                )
            counts[ann.get("category_id")] += 1
            per_image[ann.get("image_id")] += 1
        base = cfg["raw"] / f"UVH-26-{split.title()}" / "data"
        files = list(base.rglob("*.png"))
        byname = defaultdict(list)
        for f in files:
            byname[f.name].append(f)
        recs = []
        if not annotations_only:
            with ThreadPoolExecutor(max_workers=4) as pool:
                checked = pool.map(
                    check_image,
                    (
                        (
                            byname[im["file_name"]][0]
                            if byname.get(im["file_name"])
                            else None,
                            im,
                        )
                        for im in d["images"]
                    ),
                )
                for i, record in enumerate(checked, 1):
                    recs.append(record)
                    if i % 1000 == 0:
                        print(
                            f"{split}: decoded/hashed {i}/{len(d['images'])} images",
                            flush=True,
                        )
            for r in recs:
                r["split"] = split
            resolutions.extend(recs)
        all_names[split] = set(r["file_name"] for r in d["images"])
        all_ids[split] = set(ims)
        all_hashes[split] = set(r["sha256"] for r in recs if r["sha256"])
        result["splits"][split] = {
            "images": len(d["images"]),
            "annotations": len(d["annotations"]),
            "disk_images": len(files),
            "empty_annotation_list": not d["annotations"],
            "empty_images": not d["images"],
            "images_without_annotations": sum(per_image[i] == 0 for i in ims),
            "duplicate_image_ids": duplicates(d["images"], "id"),
            "duplicate_annotation_ids": duplicates(d["annotations"], "id"),
            "duplicate_category_ids": duplicates(d["categories"], "id"),
            "image_bytes": sum(f.stat().st_size for f in files),
            "duplicate_filenames": duplicates(d["images"], "file_name"),
            "duplicate_disk_basenames": [k for k, v in byname.items() if len(v) > 1],
            "missing_fields": errors,
            "invalid_annotations": sum(r["split"] == split for r in invalid),
            "image_errors": [r for r in recs if r["error"]],
            "annotation_sha256": sha256(path),
            "unreferenced_files": sorted(set(byname) - all_names[split]),
        }
        for cid, name in cats.items():
            class_rows.append(
                {
                    "split": split,
                    "category_id": cid,
                    "name": name,
                    "instances": counts[cid],
                }
            )
        print(
            split,
            len(ims),
            len(d["annotations"]),
            "image checks",
            len(recs),
            flush=True,
        )
    result["leakage"] = {
        "shared_image_ids": len(all_ids["train"] & all_ids["val"]),
        "shared_filenames": sorted(all_names["train"] & all_names["val"]),
        "shared_sha256": sorted(all_hashes["train"] & all_hashes["val"]),
        "note": "COCO image IDs are split-scoped; filename and content overlap determine leakage. No camera/temporal grouping inferred.",
    }
    filename = "annotation_audit.json" if annotations_only else "raw_dataset_audit.json"
    save_json(out / filename, result)
    pd.DataFrame(class_rows).to_csv(out / "class_distribution.csv", index=False)
    pd.DataFrame(
        invalid, columns=["split", "annotation_id", "image_id", "reason", "bbox"]
    ).to_csv(out / "invalid_boxes.csv", index=False)
    pd.DataFrame(
        missing, columns=["split", "annotation_id", "image_id", "reason"]
    ).to_csv(out / "missing_pairs.csv", index=False)
    if resolutions:
        pd.DataFrame(resolutions).to_csv(out / "resolution_summary.csv", index=False)
    pd.DataFrame(
        [
            {"split": s, **{k: v for k, v in r.items() if isinstance(v, (int, bool))}}
            for s, r in result["splits"].items()
        ]
    ).to_csv(out / "split_summary.csv", index=False)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paths.local.yaml")
    p.add_argument("--annotations-only", action="store_true")
    a = p.parse_args()
    audit(a.config, a.annotations_only)


if __name__ == "__main__":
    main()
