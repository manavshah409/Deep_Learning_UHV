"""Session-local prediction with explicit timing and immutable downloadable evidence."""

import csv
import hashlib
import io
import json
import platform
import re
import time
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dashboard.new_data import analysis
from dashboard.services import (
    CHECKPOINT,
    EXPECTED_SHA,
    ROOT,
    make_detector,
    verify_checkpoint,
)


def run(images, records, audit, metadata, device="cpu", bootstrap=100):
    import torch
    import ultralytics

    protocol_path = (
        ROOT / "reports/comparisons/E1_E3_stageE_v2/operating_thresholds.json"
    )
    protocol_bytes = protocol_path.read_bytes()
    if json.loads(protocol_bytes)["payload"]["thresholds"]["E1"] != analysis.CONFIDENCE:
        raise ValueError("Frozen confidence protocol mismatch.")
    verify_checkpoint(ROOT / CHECKPOINT)
    started = time.perf_counter()
    detector, _ = make_detector(device, 0.001, 0.7, 300)
    detector.sync()
    load_ms = (time.perf_counter() - started) * 1000
    first = np.asarray(next(iter(images.values())))[:, :, ::-1].copy()
    for _ in range(10):
        detector(first)
    detector.sync()
    timing = []
    for record in records:
        im = images[record["name"]]
        detector.sync()
        start = time.perf_counter()
        frame = np.asarray(im)[:, :, ::-1].copy()
        pre = time.perf_counter()
        # Ultralytics stage timers include internal resize/tensor preparation and NMS.
        result = detector.model.predict(
            frame,
            imgsz=640,
            conf=0.001,
            iou=0.7,
            max_det=300,
            device=detector.device,
            verbose=False,
            quantize=32,
        )[0]
        detector.sync()
        arr = result.boxes.data.cpu().numpy()
        end = time.perf_counter()
        record.update(
            boxes=arr[:, :4].tolist(),
            scores=arr[:, 4].tolist(),
            classes=arr[:, 5].astype(int).tolist(),
        )
        timing.append(
            {
                "name": record["name"],
                "width": im.width,
                "height": im.height,
                "decode_ms": record.get("decode_ms", 0),
                "preprocessing_ms": record.get("decode_ms", 0)
                + (pre - start) * 1000
                + result.speed["preprocess"],
                "inference_ms": result.speed["inference"],
                "postprocessing_ms": result.speed["postprocess"],
                "end_to_end_ms": record.get("decode_ms", 0) + (end - start) * 1000,
            }
        )
    config = {
        "threshold_protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "class_names": metadata["names"],
        "model": "YOLOv8s E1 epoch22",
        "checkpoint_sha256": EXPECTED_SHA,
        "imgsz": 640,
        "batch": 1,
        "device": detector.device,
        "ap_score_floor": 0.001,
        "confidence": 0.34,
        "nms_iou": 0.7,
        "matching_iou": 0.5,
        "max_det": 300,
        "warmup": 10,
        "precision": "float32",
        "metric_protocol": "common_metrics COCO v1; macro over GT-present classes",
        "timing": "Ultralytics stage timers; synchronized per-image outer time includes array conversion and result transfer; plus separately measured upload decode time; excludes upload transfer, loading, plotting, export. Stage timers do not sum to outer time.",
    }
    metrics = (
        analysis.unlabeled(records)
        if metadata["format"] == "unlabeled"
        else analysis.labeled(records, bootstrap)
    )
    return {
        "schema_version": 1,
        "run_id": "newdata_"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8],
        "status": "COMPLETE",
        "created_utc": datetime.now(UTC).isoformat(),
        "metadata": metadata,
        "config": config,
        "class_mapping_sha256": hashlib.sha256(
            json.dumps(metadata["names"]).encode()
        ).hexdigest(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("analysis.py"),
                Path(__file__).with_name("ingest.py"),
                ROOT / "src/evaluation/common_metrics.py",
                ROOT / "src/video/pipeline.py",
            ]
        },
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "audit": audit,
        "manifest_sha256": hashlib.sha256(
            json.dumps(
                [
                    {
                        k: v
                        for k, v in r.items()
                        if k not in {"boxes", "scores", "classes", "decode_ms"}
                    }
                    for r in records
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "records": records,
        "metrics": metrics,
        "timing": timing,
        "latency": analysis.latency(timing),
        "model_load_ms": load_ms,
    }


def bundle(result):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("result.json", json.dumps(result, allow_nan=False, indent=2))
        for key in [
            "config",
            "audit",
            "metadata",
            "environment",
            "metrics",
            "timing",
            "records",
        ]:
            z.writestr(
                key + ".json", json.dumps(result.get(key), allow_nan=False, indent=2)
            )
        for name in ["per_class", "per_image", "errors"]:
            rows = result.get("metrics", {}).get(name, [])
            if rows:
                out = io.StringIO()
                writer = csv.DictWriter(
                    out, fieldnames=sorted({k for row in rows for k in row})
                )
                writer.writeheader()
                writer.writerows(rows)
                z.writestr(name + ".csv", out.getvalue())
        z.writestr(
            result["status"] + ".json",
            json.dumps({"run_id": result["run_id"], "status": result["status"]}),
        )
    return buf.getvalue()


def publish(result, parent):
    """Exclusive directory claim; atomic payload publication; never overwrite."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", result["run_id"]):
        raise ValueError("Invalid run ID.")
    target = Path(parent) / result["run_id"]
    target.mkdir(parents=True, exist_ok=False)
    try:
        temp = target / "bundle.tmp"
        temp.write_bytes(bundle(result))
        temp.replace(target / "evidence.zip")
    except Exception:
        (target / "FAILED.json").write_text(
            json.dumps({"status": "FAILED", "reason": "Export failed"})
        )
        raise
    return target


def load_saved(data):
    """Validate an exported result without loading weights or trusting uploaded metrics."""
    if len(data) > 100 * 1024**2:
        raise ValueError("Saved result exceeds 100 MB.")
    r = json.loads(data)
    if r.get("schema_version") != 1 or r.get("status") != "COMPLETE":
        raise ValueError("Only completed schema-v1 results supported.")
    c = r["config"]
    if (
        c["checkpoint_sha256"] != EXPECTED_SHA
        or c["class_names"] != analysis.NAMES
        or c["confidence"] != 0.34
        or c["imgsz"] != 640
    ):
        raise ValueError("Saved protocol mismatch.")
    if (
        hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
        != r["config_sha256"]
    ):
        raise ValueError("Configuration hash mismatch.")
    from dashboard.new_data.ingest import box

    if not 0 < len(r["records"]) <= 1000:
        raise ValueError("Invalid record count.")
    ids = set()
    for rec in r["records"]:
        if rec["image_id"] in ids:
            raise ValueError("Duplicate image ID.")
        ids.add(rec["image_id"])
        if not (len(rec["boxes"]) == len(rec["classes"]) == len(rec["scores"]) <= 300):
            raise ValueError("Invalid prediction lengths.")
        if (
            len(rec["gt_boxes"]) != len(rec["gt_classes"])
            or len(rec["gt_boxes"]) > 10000
        ):
            raise ValueError("Invalid annotation lengths.")
        for b, cl in zip(
            rec["boxes"] + rec["gt_boxes"], rec["classes"] + rec["gt_classes"]
        ):
            box(cl, b, rec["width"], rec["height"])
        if not all(np.isfinite(s) and 0.001 <= s <= 1 for s in rec["scores"]):
            raise ValueError("Invalid scores.")
    expected = hashlib.sha256(
        json.dumps(
            [
                {
                    k: v
                    for k, v in rec.items()
                    if k not in {"boxes", "scores", "classes", "decode_ms"}
                }
                for rec in r["records"]
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if expected != r["manifest_sha256"]:
        raise ValueError("Manifest hash mismatch.")
    r["metrics"] = (
        analysis.unlabeled(r["records"])
        if r["metadata"]["format"] == "unlabeled"
        else analysis.labeled(r["records"], 100)
    )
    r["latency"] = analysis.latency(r["timing"])
    r["import_note"] = (
        "Metrics recomputed from supplied records. Prediction, timing and source provenance remain user-supplied; images are not bundled."
    )
    return r
