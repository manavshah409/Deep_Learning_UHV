"""Bounded artifact access and session-local inference using the frozen detector."""

import csv
import hashlib
import io
import json
import threading
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "runs/E1_yolov8s_uvh26_mv_640_seed42/weights/best.pt"
EXPECTED_SHA = "9f1045381024445b50e33539791da224b732d61781c73b347013477ebb077fab"


def artifact(relative: str):
    """Only presentation artifacts; never dataset or arbitrary filesystem access."""
    p = (ROOT / relative).resolve()
    if not p.is_relative_to(ROOT) or Path(relative).parts[0] not in {
        "reports",
        "docs",
        "configs",
    }:
        return None
    try:
        if p.suffix == ".json":
            return json.loads(p.read_text())
        if p.suffix == ".csv":
            return pd.read_csv(p)
        return p.read_text()
    except (OSError, ValueError, UnicodeError):
        return None


def verify_checkpoint(path: Path, expected: str = EXPECTED_SHA) -> str:
    if not path.is_file():
        raise ValueError(
            "E1 weights are not available locally. Use Saved Evidence Mode."
        )
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != expected:
        raise ValueError("Checkpoint verification failed. Inference is disabled.")
    return h.hexdigest()


def safe_device(requested: str, mps_available: bool) -> str:
    return "mps" if requested == "mps" and mps_available else "cpu"


def decode_image(data: bytes, name: str) -> Image.Image:
    if (
        Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}
        or len(data) > 10 * 1024**2
    ):
        raise ValueError("Upload a JPG or PNG of at most 10 MB.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in {"JPEG", "PNG"} or im.width * im.height > 20_000_000:
                raise ValueError("Unsupported image or image exceeds 20 megapixels.")
            im.load()
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded image could not be decoded safely.") from exc


def validate_video(name: str, size: int) -> None:
    if (
        Path(name).suffix.lower() not in {".mp4", ".mov", ".avi"}
        or size > 100 * 1024**2
    ):
        raise ValueError("Video must be MP4, MOV or AVI and at most 100 MB.")


def counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(row["name"] for row in rows))


def density(count: int, low: int = 5, moderate: int = 15) -> str:
    if not 0 <= low < moderate or count < 0:
        raise ValueError("Density boundaries must be ordered and nonnegative.")
    return "Low" if count <= low else "Moderate" if count <= moderate else "High"


def exports(rows: list[dict]) -> tuple[str, str]:
    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=["class_id", "name", "confidence", "x1", "y1", "x2", "y2"]
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue(), json.dumps(rows, indent=2, allow_nan=False)


def make_detector(device: str, confidence: float, iou: float, maximum: int):
    import torch
    import yaml

    from src.video.pipeline import CLASS_NAMES, FrozenDetector

    mapping = yaml.safe_load((ROOT / "configs/class_mapping.yaml").read_text())
    if [
        row["name"] for row in sorted(mapping, key=lambda row: row["yolo_id"])
    ] != CLASS_NAMES:
        raise ValueError("Class mapping verification failed.")
    device = safe_device(device, torch.backends.mps.is_available())

    detector = FrozenDetector(
        {
            "checkpoint": CHECKPOINT,
            "device": device,
            "cpu_fallback": True,
            "seed": 42,
            "confidence": confidence,
            "nms_iou": iou,
            "max_det": maximum,
        }
    )
    return detector, threading.Lock()


def infer(image: Image.Image, resource, labels: bool, scores: bool) -> dict:
    import numpy as np

    detector, lock = resource
    # The cache is shared; prediction calls are serialized, outputs are session-local.
    with lock:
        detector.sync()
        begin = time.perf_counter()
        frame = np.asarray(image)[:, :, ::-1].copy()
        detector.sync()
        started = time.perf_counter()
        boxes = detector(frame)
        detector.sync()
        inferred = time.perf_counter()
        names = detector.model.names
        rows = [
            {
                "class_id": int(c),
                "name": names[int(c)],
                "confidence": float(s),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
            for x1, y1, x2, y2, s, c in boxes
        ]
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for r in rows:
            draw.rectangle(
                (r["x1"], r["y1"], r["x2"], r["y2"]), outline="#ff9e40", width=3
            )
            text = ("{} ".format(r["name"]) if labels else "") + (
                "{:.2f}".format(r["confidence"]) if scores else ""
            )
            if text:
                draw.text((r["x1"], max(0, r["y1"] - 12)), text, fill="#ff9e40")
        out = io.BytesIO()
        annotated.save(out, format="PNG")
        return {
            "rows": rows,
            "png": out.getvalue(),
            "device": detector.device,
            "inference_ms": 1000 * (inferred - started),
            "pipeline_ms": 1000 * (time.perf_counter() - begin),
        }
