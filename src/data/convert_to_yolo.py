"""Versioned MV conversion with separate labels, symlinks and rejection ledger."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import os
import yaml
import pandas as pd
from .common import (
    paths,
    annotation_paths,
    load_coco,
    unique_index,
    category_mapping,
    convert_box,
    save_json,
    sha256,
    validate_manifest,
    REVISION,
    ROOT,
)


def convert(config, version):
    if Path(version).name != version or version in (".", ".."):
        raise ValueError("Invalid version")
    cfg = paths(config)
    target = cfg["processed"] / version
    sources = annotation_paths(cfg["raw"])
    signature = {
        "revision": REVISION,
        "annotations": {s: sha256(p) for s, p in sources.items()},
        "policy": "reject-invalid-v1",
    }
    if target.exists():
        if (
            not (target / "version.json").exists()
            or json.loads((target / "version.json").read_text()) != signature
        ):
            raise ValueError(
                "Existing processed version differs or is incomplete; choose a new version"
            )
        print("Existing matching version; run validate_yolo to verify contents")
        return target
    audit = json.loads((cfg["reports"] / "audit/raw_dataset_audit.json").read_text())
    if not audit["image_audit_executed"]:
        raise ValueError("Full image audit required")
    if audit["leakage"]["shared_filenames"] or audit["leakage"]["shared_sha256"]:
        raise ValueError("Unresolved train-validation leakage")
    for split, r in audit["splits"].items():
        if r["annotation_sha256"] != signature["annotations"][split]:
            raise ValueError("Stale audit")
        if any(
            r[k]
            for k in (
                "duplicate_image_ids",
                "duplicate_annotation_ids",
                "duplicate_category_ids",
                "duplicate_filenames",
                "duplicate_disk_basenames",
                "missing_fields",
                "image_errors",
            )
        ):
            raise ValueError(f"Unresolved structural problems: {split}")
    mapping = category_mapping(load_coco(sources["train"])["categories"])
    ids = {c["original_id"]: c["yolo_id"] for c in mapping}
    names = {c["yolo_id"]: c["name"] for c in mapping}
    temp = target.with_name(target.name + ".building")
    if temp.exists():
        raise ValueError("Incomplete conversion directory exists; inspect before retry")
    temp.mkdir(parents=True)
    rows = []
    rejected = []
    frequencies = Counter()
    for split, path in sources.items():
        d = load_coco(path)
        if category_mapping(d["categories"]) != mapping:
            raise ValueError("Split category mismatch")
        images = unique_index(d["images"])
        unique_index(d["annotations"])
        annos = defaultdict(list)
        for ann in d["annotations"]:
            if ann["image_id"] not in images:
                raise ValueError("Unknown image reference")
            if ann["category_id"] not in ids:
                raise ValueError("Unknown category")
            annos[ann["image_id"]].append(ann)
        files = unique_index(
            [
                {"name": p.name, "path": p}
                for p in (cfg["raw"] / f"UVH-26-{split.title()}" / "data").rglob(
                    "*.png"
                )
            ],
            "name",
        )
        for part in ("images", "labels"):
            (temp / part / split).mkdir(parents=True)
        for iid, im in sorted(images.items()):
            source = files[im["file_name"]]["path"]
            stem = Path(im["file_name"]).stem
            image_rel = f"images/{split}/{stem}.png"
            label_rel = f"labels/{split}/{stem}.txt"
            (temp / image_rel).symlink_to(
                os.path.relpath(source, temp / "images" / split)
            )
            lines = []
            for ann in sorted(annos[iid], key=lambda r: r["id"]):
                try:
                    coords = convert_box(ann["bbox"], im["width"], im["height"])
                except ValueError as e:
                    rejected.append(
                        {"split": split, "annotation_id": ann["id"], "reason": str(e)}
                    )
                    continue
                cid = ids[ann["category_id"]]
                frequencies[f"{split}:{cid}"] += 1
                lines.append(f"{cid} " + " ".join(f"{v:.10f}" for v in coords))
            (temp / label_rel).write_text("\n".join(lines) + ("\n" if lines else ""))
            rows.append(
                {
                    "split": split,
                    "image_id": iid,
                    "source": str(source.relative_to(cfg["raw"])),
                    "image": image_rel,
                    "label": label_rel,
                    "objects": len(lines),
                    "label_sha256": sha256(temp / label_rel),
                }
            )
    validate_manifest(rows)
    save_json(temp / "manifest.json", rows)
    save_json(temp / "version.json", signature)
    (temp / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(target),
                "train": "images/train",
                "val": "images/val",
                "names": names,
            },
            sort_keys=False,
        )
    )
    temp.rename(target)
    (ROOT / "configs/class_mapping.yaml").write_text(
        yaml.safe_dump(mapping, sort_keys=False)
    )
    (ROOT / "configs/dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": f"data/processed/{version}",
                "train": "images/train",
                "val": "images/val",
                "names": names,
            },
            sort_keys=False,
        )
    )
    pd.DataFrame(mapping).to_csv(
        cfg["reports"] / "tables/class_mapping.csv", index=False
    )
    pd.DataFrame(rejected, columns=["split", "annotation_id", "reason"]).to_csv(
        cfg["reports"] / "audit/conversion_rejected.csv", index=False
    )
    save_json(
        cfg["reports"] / "audit/conversion.json",
        {
            "version": version,
            "images": len(rows),
            "retained": sum(frequencies.values()),
            "rejected": len(rejected),
            "clipped": 0,
            "repaired": 0,
            "class_frequencies": dict(frequencies),
            "manifest_sha256": sha256(target / "manifest.json"),
        },
    )
    return target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paths.local.yaml")
    p.add_argument("--dataset-version", default="uvh26_mv_yolo_v1")
    a = p.parse_args()
    print(convert(a.config, a.dataset_version))


if __name__ == "__main__":
    main()
