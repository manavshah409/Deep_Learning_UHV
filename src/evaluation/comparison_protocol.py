"""Frozen comparison schema and fail-closed calibration access boundary."""

import hashlib
import json
import math
from pathlib import Path

from src.data.common import ROOT

SCHEMA_VERSION = "uvh-common-predictions-1"
CALIBRATION = ROOT / "reports/accuracy_stage_a/protocol_v2/calibration_500.json"
CALIBRATION_SHA = "a436b78d561516d61ae2f7eedb4a8acb4a58222d2c13df80411ae807f288c502"
RESERVED = ROOT / "reports/accuracy_stage_a/protocol_v2/final_evaluation_1500.json"
RESERVED_SHA = "6fc3e8928e5e90c3390ab298a234f90ad883d413f292dc04f38b2e63dbe5d0e0"
MAPPING_SHA = "6fd0b458fed2cb174c924d4fa9ead86db0a292dc7c55fd0f4d4b3b9b20d08ea8"
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
GRID = [0.001] + [i / 100 for i in range(1, 100)]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def seal(value):
    return {"payload": value, "canonical_sha256": canonical_hash(value)}


def verify_seal(value):
    if canonical_hash(value["payload"]) != value["canonical_sha256"]:
        raise ValueError("Protocol content hash mismatch")
    return value["payload"]


def load_manifest(path=CALIBRATION, *, authorize_reserved_evaluation=False):
    path = Path(path).resolve()
    allowed = {CALIBRATION.resolve(): CALIBRATION_SHA}
    if authorize_reserved_evaluation:
        allowed[RESERVED.resolve()] = RESERVED_SHA
    if path not in allowed:
        raise PermissionError(
            "Manifest outside calibration allowlist; reserved evaluation requires separate future authorization"
        )
    if sha(path) != allowed[path]:
        raise ValueError("Manifest checksum mismatch")
    rows = json.loads(path.read_text())
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image IDs")
    return sorted(rows, key=lambda r: r["image_id"])


def guard_path(path, root, rows):
    """Call before image/label decoding. Rows must come from load_manifest."""
    allowed = {(Path(root) / r[k]).resolve() for r in rows for k in ["image", "label"]}
    if Path(path).resolve() not in allowed:
        raise PermissionError("Image/annotation path outside authorized manifest")
    return Path(path)


def vehicle_class(native_id, model_id):
    if (
        isinstance(native_id, bool)
        or not math.isfinite(native_id)
        or int(native_id) != native_id
    ):
        raise ValueError("Non-integer class ID")
    native_id = int(native_id)
    if model_id == "E3":
        if native_id == 0:
            return None  # Explicit background exclusion, counted by the exporter.
        if not 1 <= native_id <= 14:
            raise ValueError("Invalid Faster R-CNN foreground class")
        return native_id - 1
    if model_id != "E1" or not 0 <= native_id < 14:
        raise ValueError("Unknown detector or class")
    return native_id


class OutsideImage(ValueError):
    """Valid raw box entirely outside image: explicitly rejected, counted, not exported."""


def detection(box, score, native_id, model_id, width, height):
    if len(box) != 4 or not all(math.isfinite(x) for x in [*box, score, width, height]):
        raise ValueError("Invalid/non-finite prediction")
    if width <= 0 or height <= 0 or not 0 <= score <= 1:
        raise ValueError("Invalid score or dimensions")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Non-positive raw box")
    cls = vehicle_class(native_id, model_id)
    if cls is None:
        return None
    clipped = [
        max(0.0, min(float(width), box[0])),
        max(0.0, min(float(height), box[1])),
        max(0.0, min(float(width), box[2])),
        max(0.0, min(float(height), box[3])),
    ]
    x1, y1, x2, y2 = clipped
    if x2 <= x1 or y2 <= y1:
        raise OutsideImage("Positive raw box has no intersection with original image")
    return {
        "class_id": cls,
        "class_name": NAMES[cls],
        "confidence": float(score),
        "xyxy": clipped,
        "xywh": [x1, y1, x2 - x1, y2 - y1],
        "clipped": clipped != list(box),
    }


def validate_bundle(path, receipt, expected_ids):
    if sha(path) != receipt["prediction_sha256"]:
        raise ValueError("Prediction bundle checksum mismatch")
    ids = []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            ids.append(row["image_id"])
            if (
                row["schema_version"] != SCHEMA_VERSION
                or row["checkpoint_sha256"] != receipt["checkpoint_sha256"]
                or row["model_id"] != receipt["model_id"]
            ):
                raise ValueError("Prediction identity mismatch")
            if (
                row["preprocessing"]
                != receipt["inference_configuration"]["preprocessing"]
                or row["inference_configuration"] != receipt["inference_configuration"]
            ):
                raise ValueError("Inference metadata mismatch")
            if not row["image_key"] or len(row["detections"]) > 300:
                raise ValueError("Image key or detection count invalid")
            for d in row["detections"]:
                validated = detection(
                    d["xyxy"],
                    d["confidence"],
                    d["class_id"],
                    "E1",
                    row["width"],
                    row["height"],
                )
                if (
                    validated is None
                    or validated["class_name"] != d["class_name"]
                    or validated["xywh"] != d["xywh"]
                    or validated["xyxy"] != d["xyxy"]
                    or d["confidence"] < 0.001
                ):
                    raise ValueError("Malformed common prediction")
    if ids != list(expected_ids) or len(set(ids)) != len(ids):
        raise ValueError("Wrong, duplicate or unordered calibration image IDs")
    return True


def unletterbox_boxes(boxes, input_shape, original_shape):
    """Ultralytics scale_boxes arithmetic without its final clipping operation."""
    boxes = boxes.clone()
    gain = min(input_shape[0] / original_shape[0], input_shape[1] / original_shape[1])
    px = round((input_shape[1] - round(original_shape[1] * gain)) / 2 - 0.1)
    py = round((input_shape[0] - round(original_shape[0] * gain)) / 2 - 0.1)
    boxes[:, 0] -= px
    boxes[:, 2] -= px
    boxes[:, 1] -= py
    boxes[:, 3] -= py
    boxes /= gain
    return boxes
