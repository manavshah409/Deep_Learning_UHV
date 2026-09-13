"""Prepare an explicitly small, independently audited pilot before full acquisition."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import os
import yaml
from .common import (
    paths,
    annotation_paths,
    load_coco,
    category_mapping,
    convert_box,
    save_json,
    sha256,
    validate_manifest,
    validate_line,
    REVISION,
    ROOT,
)
from .build_subset import select
from .validate_raw import check_image


def plan():
    cfg = paths()
    metadata = json.loads((ROOT / "data/interim/hub_metadata.json").read_text())
    files = {
        Path(r["rfilename"]).name: r["rfilename"]
        for r in metadata["siblings"]
        if r["rfilename"].endswith(".png")
    }
    output = cfg["processed"] / "uvh26_mv_smoke_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        return output, json.loads((output / "selection.json").read_text())
    selected = []
    for split, n in [("train", 64), ("val", 32)]:
        d = load_coco(annotation_paths(cfg["raw"])[split])
        classes = defaultdict(set)
        for ann in d["annotations"]:
            classes[ann["image_id"]].add(ann["category_id"])
        rows = [
            {"split": split, "image_id": im["id"], "source": files[im["file_name"]]}
            for im in d["images"]
        ]
        selected += select(rows, classes, n, 42)
    save_json(output / "selection.json", selected)
    return output, selected


def prepare(download=False):
    cfg = paths()
    output, selected = plan()
    if (output / "manifest.json").exists():
        raise ValueError("Smoke dataset already prepared")
    if download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            "iisc-aim/UVH-26",
            repo_type="dataset",
            revision=REVISION,
            local_dir=cfg["raw"],
            allow_patterns=[r["source"] for r in selected],
            max_workers=8,
        )
    mapping = category_mapping(
        load_coco(annotation_paths(cfg["raw"])["train"])["categories"]
    )
    idmap = {r["original_id"]: r["yolo_id"] for r in mapping}
    manifest = []
    audits = []
    hashes = {}
    freqs = Counter()
    for split, path in annotation_paths(cfg["raw"]).items():
        d = load_coco(path)
        idx = {r["id"]: r for r in d["images"]}
        annos = defaultdict(list)
        for ann in d["annotations"]:
            annos[ann["image_id"]].append(ann)
        for r in selected:
            if r["split"] != split:
                continue
            source = cfg["raw"] / r["source"]
            meta = idx[r["image_id"]]
            checked = check_image((source, meta))
            if checked["error"]:
                raise ValueError(checked)
            if checked["sha256"] in hashes and hashes[checked["sha256"]] != split:
                raise ValueError("Smoke content leakage")
            hashes[checked["sha256"]] = split
            audits.append({"split": split, **checked})
            stem = source.stem
            image = f"images/{split}/{stem}.png"
            label = f"labels/{split}/{stem}.txt"
            (output / image).parent.mkdir(parents=True, exist_ok=True)
            (output / label).parent.mkdir(parents=True, exist_ok=True)
            (output / image).symlink_to(
                os.path.relpath(source, (output / image).parent)
            )
            lines = []
            for ann in sorted(annos[r["image_id"]], key=lambda a: a["id"]):
                cid = idmap[ann["category_id"]]
                coords = convert_box(ann["bbox"], meta["width"], meta["height"])
                line = f"{cid} " + " ".join(f"{v:.10f}" for v in coords)
                validate_line(line, len(mapping))
                lines.append(line)
                freqs[f"{split}:{cid}"] += 1
            (output / label).write_text("\n".join(lines) + "\n" if lines else "")
            manifest.append(
                {
                    **r,
                    "image": image,
                    "label": label,
                    "objects": len(lines),
                    "label_sha256": sha256(output / label),
                }
            )
    validate_manifest(manifest)
    save_json(output / "manifest.json", manifest)
    spec = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "names": {r["yolo_id"]: r["name"] for r in mapping},
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    save_json(
        cfg["reports"] / "audit/smoke_data_audit.json",
        {
            "scope": "96-image pilot only, not full raw-dataset audit",
            "revision": REVISION,
            "train_images": 64,
            "val_images": 32,
            "images": audits,
            "class_frequencies": dict(freqs),
            "manifest_sha256": sha256(output / "manifest.json"),
        },
    )
    print(output / "dataset.yaml")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    prepare(a.download)


if __name__ == "__main__":
    main()
