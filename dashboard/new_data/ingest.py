"""Bounded, memory-only upload validation and explicit annotation mapping."""

import hashlib
import io
import json
import math
import re
import time
import zipfile
from pathlib import PurePosixPath

from dashboard.services import decode_image

NAMES = [
    "Hatchback",
    "Sedan",
    "SUV",
    "MUV",
    "Bus",
    "Truck",
    "Three-wheeler",
    "Two-wheeler",
    "LCV",
    "Mini-bus",
    "Tempo-traveller",
    "Bicycle",
    "Van",
    "Others",
]
LIMIT = 100 * 1024**2


def unpack(data):
    if len(data) > LIMIT:
        raise ValueError("ZIP exceeds 100 MB.")
    files = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) > 2000 or sum(i.file_size for i in members) > LIMIT:
            raise ValueError("Archive exceeds 2000 entries or 100 MB expanded.")
        for item in members:
            p = PurePosixPath(item.filename)
            if (
                p.is_absolute()
                or ".." in p.parts
                or "\\" in item.filename
                or (item.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("Unsafe archive path or symlink.")
            if item.is_dir():
                continue
            if p.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".txt",
                ".json",
                ".yaml",
                ".yml",
            }:
                raise ValueError("Unsupported archive file type.")
            if p.name.casefold() in {n.casefold() for n in files}:
                raise ValueError(
                    "Duplicate filenames are not allowed, including nested folders."
                )
            files[p.name] = archive.read(item)
    return files


def box(cls, coords, width, height):
    if not isinstance(cls, int) or not 0 <= cls < 14:
        raise ValueError("Unknown class ID.")
    if len(coords) != 4 or not all(math.isfinite(float(x)) for x in coords):
        raise ValueError("Nonfinite or malformed box.")
    x1, y1, x2, y2 = map(float, coords)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Invalid or out-of-range box.")
    return [x1, y1, x2, y2]


def ingest(files, fmt, metadata, known_hashes=None):
    """known_hashes is an explicitly supplied safe index; never scan local datasets."""
    if fmt not in {"unlabeled", "yolo", "coco"}:
        raise ValueError("Unknown annotation format.")
    if (
        not metadata.get("source")
        or not metadata.get("permission")
        or metadata.get("format") != fmt
    ):
        raise ValueError(
            "Source, permission and annotation format confirmation required."
        )
    if not isinstance(metadata.get("independent_annotations"), bool) or not isinstance(
        metadata.get("used_for_training_or_tuning"), bool
    ):
        raise TypeError(
            "Explicit annotation independence and prior-use declarations required."
        )
    if metadata.get("names") != NAMES:
        raise ValueError(
            "Confirm the exact ordered 14-class mapping in metadata names."
        )
    if any(not re.fullmatch(r"[A-Za-z0-9_. -]+", n) for n in files):
        raise ValueError(
            "Filenames must use letters, digits, spaces, dot, dash or underscore."
        )
    if len(files) > 2000 or sum(len(v) for v in files.values()) > LIMIT:
        raise ValueError("Upload exceeds limits.")
    if any(PurePosixPath(n).name != n or "\\" in n for n in files):
        raise ValueError("Use sanitized base filenames.")
    if len({n.casefold() for n in files}) != len(files):
        raise ValueError("Duplicate filenames.")
    images = {}
    decode_times = {}
    total_pixels = 0
    for n, v in files.items():
        if PurePosixPath(n).suffix.lower() in {".png", ".jpg", ".jpeg"}:
            start = time.perf_counter()
            images[n] = decode_image(v, n)
            total_pixels += images[n].width * images[n].height
            if len(images) > 500 or total_pixels > 100_000_000:
                raise ValueError(
                    "Limit: 500 images and 100 megapixels decoded in total."
                )
            decode_times[n] = (time.perf_counter() - start) * 1000
    if (
        len(images) > 500
        or sum(im.width * im.height for im in images.values()) > 100_000_000
    ):
        raise ValueError("Limit: 500 images and 100 megapixels decoded in total.")
    if not images:
        raise ValueError("No images supplied.")
    stems = [PurePosixPath(n).stem.casefold() for n in images]
    if len(stems) != len(set(stems)):
        raise ValueError("Ambiguous duplicate image stems.")
    gt = {n: [] for n in images}
    if fmt == "yolo":
        labels = {
            PurePosixPath(n).stem: v for n, v in files.items() if n.endswith(".txt")
        }
        expected = {PurePosixPath(n).stem for n in images}
        if set(labels) != expected:
            raise ValueError(
                "Missing or orphan YOLO labels; use empty files for empty images."
            )
        for n, im in images.items():
            for line in labels[PurePosixPath(n).stem].decode("utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError("YOLO rows require class xc yc width height.")
                c = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                W, H = im.size
                coords = box(
                    c,
                    [
                        (x - w / 2) * W,
                        (y - h / 2) * H,
                        (x + w / 2) * W,
                        (y + h / 2) * H,
                    ],
                    W,
                    H,
                )
                gt[n].append((c, coords))
    elif fmt == "coco":
        if "annotations.json" not in files:
            raise ValueError("COCO JSON must be named annotations.json.")
        coco = json.loads(files["annotations.json"])
        cats = coco["categories"]
        mapping = {c["id"]: NAMES.index(c["name"]) for c in cats}
        if len(mapping) != len(cats) or len(set(mapping.values())) != len(cats):
            raise ValueError("Duplicate COCO categories.")
        ids = {}
        annids = set()
        for row in coco["images"]:
            n = row["file_name"]
            if n not in images or row["id"] in ids or n in ids.values():
                raise ValueError("Unknown, duplicate or unsafe COCO image reference.")
            if images[n].size != (row["width"], row["height"]):
                raise ValueError("COCO dimensions disagree with decoded image.")
            ids[row["id"]] = n
        if set(ids.values()) != set(images):
            raise ValueError("Images absent from COCO manifest.")
        for a in coco["annotations"]:
            if a["id"] in annids or a.get("iscrowd", 0) or a.get("ignore", 0):
                raise ValueError(
                    "Duplicate annotation ID or unsupported crowd/ignore annotation."
                )
            annids.add(a["id"])
            if a["image_id"] not in ids or a["category_id"] not in mapping:
                raise ValueError("Orphan annotation or unknown category.")
            n = ids[a["image_id"]]
            c = mapping[a["category_id"]]
            x, y, w, h = a["bbox"]
            gt[n].append((c, box(c, [x, y, x + w, y + h], *images[n].size)))
    elif any(n.endswith(".txt") or n == "annotations.json" for n in files):
        raise ValueError("Labels supplied in unlabeled mode; select labeled workflow.")
    records = []
    seen = {}
    duplicates = []
    overlap = []
    for i, (n, im) in enumerate(sorted(images.items())):
        digest = hashlib.sha256(files[n]).hexdigest()
        pixel = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        if pixel in seen:
            duplicates.append([seen[pixel], n])
        seen[pixel] = n
        if known_hashes and (digest in known_hashes or pixel in known_hashes):
            overlap.append(n)
        records.append(
            {
                "image_id": i + 1,
                "name": n,
                "width": im.width,
                "height": im.height,
                "sha256": digest,
                "decode_ms": decode_times[n],
                "pixel_sha256": pixel,
                "gt_boxes": [b for c, b in gt[n]],
                "gt_classes": [c for c, b in gt[n]],
            }
        )
    return (
        images,
        records,
        {
            "input_sha256": {
                n: hashlib.sha256(v).hexdigest() for n, v in files.items()
            },
            "duplicates": duplicates,
            "overlap": overlap,
            "overlap_check": "available index only"
            if known_hashes is not None
            else "Unavailable: no safe hash index supplied",
            "independence": "not independent"
            if overlap
            or metadata["used_for_training_or_tuning"]
            or not metadata["independent_annotations"]
            else "declared independent; overlap verification incomplete"
            if known_hashes is None
            else "declared independent against supplied index",
            "empty_images": sum(not r["gt_boxes"] for r in records)
            if fmt != "unlabeled"
            else None,
            "invalid_boxes": 0,
        },
    )
