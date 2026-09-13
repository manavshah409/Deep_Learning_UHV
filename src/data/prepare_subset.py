"""Acquire and audit a representative subset while the full download continues.

Full-dataset acquisition and audit remain required for Phase 1 completion.
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil

import pandas as pd
import yaml

from .build_subset import select
from .common import (
    ROOT,
    REVISION,
    annotation_paths,
    category_mapping,
    convert_box,
    load_coco,
    paths,
    save_json,
    sha256,
    unique_index,
    validate_line,
    validate_manifest,
)
from .validate_raw import check_image


def prepare(config, version, train_count, val_count, download):
    cfg = paths(config)
    if Path(version).name != version or version in (".", ".."):
        raise ValueError("Invalid version name")
    output = cfg["processed"] / version
    sources = annotation_paths(cfg["raw"])
    identity = {
        "version": version,
        "revision": REVISION,
        "train": train_count,
        "val": val_count,
        "seed": 42,
        "annotation_sha256": {s: sha256(p) for s, p in sources.items()},
    }
    plan_dir = ROOT / "data/interim/subset_plans" / version
    plan_dir.mkdir(parents=True, exist_ok=True)
    identity_path = plan_dir / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Existing selection version differs")
    if output.exists():
        raise ValueError("Prepared subset already exists; choose a new version")
    metadata_path = ROOT / "data/interim/hub_metadata.json"
    if not metadata_path.exists():
        from huggingface_hub import HfApi

        inventory = HfApi().list_repo_files(
            "iisc-aim/UVH-26", repo_type="dataset", revision=REVISION
        )
        save_json(
            metadata_path,
            {"sha": REVISION, "siblings": [{"rfilename": p} for p in inventory]},
        )
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("sha") != REVISION:
        raise ValueError("File inventory does not match the pinned revision")
    file_rows = [
        {"name": Path(r["rfilename"]).name, "path": r["rfilename"]}
        for r in metadata["siblings"]
        if r["rfilename"].endswith(".png")
    ]
    file_index = unique_index(file_rows, "name")
    datasets = {s: load_coco(p) for s, p in sources.items()}
    mapping = category_mapping(datasets["train"]["categories"])
    ids = {c["original_id"]: c["yolo_id"] for c in mapping}
    selected = []
    distribution = []
    annotation_index = {}
    for split, count in [("train", train_count), ("val", val_count)]:
        data = datasets[split]
        if category_mapping(data["categories"]) != mapping:
            raise ValueError("Categories differ across splits")
        unique_index(data["images"])
        unique_index(data["annotations"])
        byid = defaultdict(list)
        counts = defaultdict(Counter)
        for ann in data["annotations"]:
            byid[ann["image_id"]].append(ann)
            counts[ann["image_id"]][ann["category_id"]] += 1
        annotation_index[split] = byid
        corrupt_files = {"22876.png", "163302.png", "368738.png", "671644.png"}
        rows = [
            {
                "split": split,
                "image_id": im["id"],
                "source": file_index[im["file_name"]]["path"],
            }
            for im in data["images"]
            if im["file_name"] not in corrupt_files
        ]
        chosen = select(
            rows, {r["image_id"]: set(counts[r["image_id"]]) for r in rows}, count, 42
        )
        selected.extend(chosen)
        full = sum(counts.values(), Counter())
        sub = sum((counts[r["image_id"]] for r in chosen), Counter())
        full_total = sum(full.values())
        sub_total = sum(sub.values())
        for c in mapping:
            cid = c["original_id"]
            full_share = full[cid] / full_total if full_total else 0.0
            sub_share = sub[cid] / sub_total if sub_total else 0.0
            distribution.append(
                {
                    "split": split,
                    "class_id": ids[cid],
                    "name": c["name"],
                    "full_instances": full[cid],
                    "subset_instances": sub[cid],
                    "full_share": full_share,
                    "subset_share": sub_share,
                    "share_difference_pp": 100 * (sub_share - full_share),
                }
            )
    selection_path = plan_dir / "selection.json"
    save_json(identity_path, identity)
    save_json(selection_path, selected)
    save_json(
        cfg["reports"] / "tables" / f"{version}_image_ids.json",
        [{"split": r["split"], "image_id": r["image_id"]} for r in selected],
    )
    pd.DataFrame(distribution).to_csv(
        cfg["reports"] / "tables" / f"{version}_distribution.csv", index=False
    )
    if download:
        if shutil.disk_usage(cfg["raw"]).free < 50 * 2**30:
            raise RuntimeError("Require at least 50 GiB free before subset acquisition")
        from huggingface_hub import snapshot_download

        snapshot_download(
            "iisc-aim/UVH-26",
            repo_type="dataset",
            revision=REVISION,
            local_dir=cfg["raw"],
            allow_patterns=[r["source"] for r in selected],
            max_workers=8,
        )
    # Audit every selected image before creating any derived dataset version.
    audits = []
    hashes = {}
    checked_by_key = {}
    metadata_by_split = {s: unique_index(d["images"]) for s, d in datasets.items()}
    for i, row in enumerate(selected, 1):
        meta = metadata_by_split[row["split"]][row["image_id"]]
        checked = check_image((cfg["raw"] / row["source"], meta))
        audits.append({"split": row["split"], **checked})
        if checked["error"]:
            save_json(cfg["reports"] / "audit" / f"{version}_failed_audit.json", audits)
            raise ValueError(f"Selected image failed audit: {row['source']}")
        digest = checked["sha256"]
        if digest in hashes and hashes[digest] != row["split"]:
            raise ValueError("Unresolved subset train-validation content leakage")
        hashes[digest] = row["split"]
        checked_by_key[(row["split"], row["image_id"])] = checked
        if i % 500 == 0:
            print(f"Audited {i}/{len(selected)} selected images", flush=True)
    temp = output.with_name(output.name + ".building")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    manifest = []
    frequencies = Counter()
    for row in selected:
        split = row["split"]
        meta = metadata_by_split[split][row["image_id"]]
        source = cfg["raw"] / row["source"]
        stem = source.stem
        image = f"images/{split}/{stem}.png"
        label = f"labels/{split}/{stem}.txt"
        (temp / image).parent.mkdir(parents=True, exist_ok=True)
        (temp / label).parent.mkdir(parents=True, exist_ok=True)
        (temp / image).symlink_to(os.path.relpath(source, (temp / image).parent))
        lines = []
        for ann in sorted(
            annotation_index[split][row["image_id"]], key=lambda a: a["id"]
        ):
            cid = ids[ann["category_id"]]
            box = convert_box(ann["bbox"], meta["width"], meta["height"])
            line = f"{cid} " + " ".join(f"{v:.10f}" for v in box)
            validate_line(line, len(mapping))
            lines.append(line)
            frequencies[f"{split}:{cid}"] += 1
        (temp / label).write_text("\n".join(lines) + ("\n" if lines else ""))
        manifest.append(
            {
                **row,
                "image": image,
                "label": label,
                "objects": len(lines),
                "label_sha256": sha256(temp / label),
                "source_sha256": checked_by_key[(split, row["image_id"])]["sha256"],
            }
        )
    validate_manifest(manifest)
    save_json(temp / "manifest.json", manifest)
    save_json(temp / "version.json", identity)
    (temp / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(output),
                "train": "images/train",
                "val": "images/val",
                "names": {c["yolo_id"]: c["name"] for c in mapping},
            },
            sort_keys=False,
        )
    )
    temp.rename(output)
    save_json(
        cfg["reports"] / "audit" / f"{version}_conversion.json",
        {
            "version": version,
            "class_frequencies": dict(frequencies),
            "rejected": 0,
            "clipped": 0,
            "repaired": 0,
            "manifest_sha256": sha256(output / "manifest.json"),
            "images": len(manifest),
            "scope": "Independently audited representative subset, full image audit still required",
        },
    )
    save_json(cfg["reports"] / "audit" / f"{version}_image_audit.json", audits)
    print(output / "dataset.yaml", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paths.local.yaml")
    p.add_argument("--dataset-version", default="uvh26_mv_baseline_subset_v1")
    p.add_argument("--train", type=int, default=8000)
    p.add_argument("--val", type=int, default=2000)
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    prepare(a.config, a.dataset_version, a.train, a.val, a.download)


if __name__ == "__main__":
    main()
