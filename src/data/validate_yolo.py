import argparse
from collections import Counter
import json
from pathlib import Path
import yaml
from .common import paths, validate_line, validate_manifest, save_json, sha256


def validate(config, version):
    cfg = paths(config)
    root = cfg["processed"] / version
    rows = json.loads((root / "manifest.json").read_text())
    validate_manifest(rows)
    spec = yaml.safe_load((root / "dataset.yaml").read_text())
    if Path(spec["path"]).resolve() != root.resolve():
        raise ValueError("Dataset YAML points outside its version")
    counts = Counter()
    splits = Counter()
    for row in rows:
        image = root / row["image"]
        label = root / row["label"]
        if not image.is_file() or not label.is_file():
            raise ValueError(f"Missing pair: {row}")
        if row.get("label_sha256") != sha256(label):
            raise ValueError("Label checksum mismatch")
        lines = label.read_text().splitlines()
        if len(lines) != row["objects"]:
            raise ValueError("Manifest object-count mismatch")
        for line in lines:
            cid, _ = validate_line(line, len(spec["names"]))
            counts[f"{row['split']}:{cid}"] += 1
        splits[row["split"]] += 1
    for split in splits:
        if len(list((root / "labels" / split).glob("*.txt"))) != splits[split]:
            raise ValueError("Extra labels")
        if len(list((root / "images" / split).glob("*.png"))) != splits[split]:
            raise ValueError("Extra images")
    conversion_path = root / "conversion.json"
    if not conversion_path.exists():
        conversion_path = cfg["reports"] / "audit" / f"{version}_conversion.json"
    if not conversion_path.exists():
        conversion_path = cfg["reports"] / "audit/conversion.json"
    conversion = json.loads(conversion_path.read_text())
    if conversion["version"] != version:
        raise ValueError("Conversion report belongs to another version")
    if counts != Counter(conversion["class_frequencies"]):
        raise ValueError("Class-frequency mismatch")
    from ultralytics.data.utils import check_det_dataset

    loaded = check_det_dataset(str(root / "dataset.yaml"), autodownload=False)
    from ultralytics.data.dataset import YOLODataset

    for split in splits:
        dataset = YOLODataset(
            img_path=loaded[split],
            data=loaded,
            imgsz=640,
            batch_size=8,
            augment=False,
            cache=False,
        )
        expected_objects = {
            str((root / r["image"]).resolve()): r["objects"]
            for r in rows
            if r["split"] == split
        }
        for item in dataset.labels:
            if (
                len(item["cls"])
                != expected_objects[str(Path(item["im_file"]).resolve())]
            ):
                raise ValueError("Ultralytics removed or changed labels during scan")
        if len(dataset) != splits[split]:
            raise ValueError("Ultralytics scan count differs")
    result = {
        "version": version,
        "split_counts": dict(splits),
        "class_frequencies": dict(counts),
        "ultralytics_scan": "passed",
    }
    save_json(
        cfg["reports"]
        / "audit"
        / f"{version.replace(chr(47), chr(95))}_yolo_validation.json",
        result,
    )
    if version == "uvh26_mv_yolo_v1":
        save_json(cfg["reports"] / "audit/yolo_validation.json", result)
    print(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paths.local.yaml")
    p.add_argument("--dataset-version", default="uvh26_mv_yolo_v1")
    a = p.parse_args()
    validate(a.config, a.dataset_version)


if __name__ == "__main__":
    main()
