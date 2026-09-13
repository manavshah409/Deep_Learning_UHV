"""Strict, deterministic data primitives. Raw files are never modified."""

from pathlib import Path
import hashlib
import json
import math
import yaml

ROOT = Path(__file__).resolve().parents[2]
REVISION = "59f82c57821e8a54dc40bc1f42e83909dbad0b70"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def paths(config="configs/paths.local.yaml"):
    config = Path(config).resolve()
    settings = yaml.safe_load(config.read_text())
    root = config.parent.parent
    return {k: (root / v).resolve() for k, v in settings.items()}


def require_directory(path):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def load_coco(path):
    data = json.loads(Path(path).read_text())
    for name in ("images", "annotations", "categories"):
        if not isinstance(data.get(name), list):
            raise ValueError(f"Missing or invalid {name}")
    return data


def unique_index(rows, key="id"):
    result = {}
    for row in rows:
        if key not in row:
            raise ValueError(f"Missing {key}")
        if row[key] in result:
            raise ValueError(f"Duplicate {key}: {row[key]}")
        result[row[key]] = row
    return result


def category_mapping(categories):
    unique_index(categories)
    if not categories or any(not isinstance(c.get("name"), str) for c in categories):
        raise ValueError("Invalid categories")
    return [
        {
            "original_id": c["id"],
            "original_name": c["name"],
            "yolo_id": i,
            "name": c["name"],
        }
        for i, c in enumerate(sorted(categories, key=lambda c: c["id"]))
    ]


def convert_box(box, width, height):
    """Reject invalid/out-of-frame boxes; no implicit clipping or repair."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("bbox must have four numbers")
    try:
        x, y, w, h = map(float, box)
        width, height = float(width), float(height)
    except (TypeError, ValueError) as e:
        raise ValueError("non_numeric") from e
    if not all(math.isfinite(v) for v in (x, y, w, h, width, height)):
        raise ValueError("non_finite")
    if width <= 0 or height <= 0:
        raise ValueError("invalid_image_dimensions")
    if w <= 0 or h <= 0:
        raise ValueError("non_positive_area")
    if min(x, y) < 0 or x + w > width or y + h > height:
        raise ValueError("outside_image")
    return ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)


def validate_line(line, nclasses):
    fields = line.split()
    if len(fields) != 5:
        raise ValueError("Expected five fields")
    if not fields[0].isdigit() or not 0 <= int(fields[0]) < nclasses:
        raise ValueError("Invalid class")
    values = list(map(float, fields[1:]))
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("Invalid normalized coordinates")
    x, y, w, h = values
    if (
        min(w, h) <= 0
        or min(x - w / 2, y - h / 2) < -1e-8
        or max(x + w / 2, y + h / 2) > 1 + 1e-8
    ):
        raise ValueError("Invalid box extent")
    return int(fields[0]), values


def validate_manifest(rows):
    seen = set()
    for row in rows:
        key = (row["split"], row["image_id"])
        if key in seen:
            raise ValueError(f"Duplicate or split leakage: {key}")
        seen.add(key)
        if row["split"] not in ("train", "val"):
            raise ValueError("Unknown split")
    files = [Path(r["source"]).name for r in rows]
    if len(set(files)) != len(files):
        raise ValueError("Duplicate source image")


def annotation_paths(raw, variant="MV"):
    return {
        s: raw / f"UVH-26-{s.title()}" / f"UVH-26-{variant}-{s.title()}.json"
        for s in ("train", "val")
    }
