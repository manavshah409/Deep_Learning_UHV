"""Synchronized batch-one still-image latency; not a live video claim."""

import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
from src.data.common import ROOT, save_json, sha256


def summarize(samples):
    values = np.asarray(samples, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Timing samples must be finite and positive")
    return dict(
        count=len(values),
        mean_ms=float(values.mean()),
        median_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        fps_from_total_time=float(1000 / values.mean()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionPredictor

    if a.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    target = ROOT / "reports/tables" / f"{a.name}_latency.json"
    if target.exists():
        raise ValueError("Timing report exists")
    root = Path(a.data).resolve().parent
    rows = json.loads((root / "val_manifest.json").read_text())
    chosen = random.Random(42).sample(rows, min(a.count, len(rows)))
    if a.count <= 0 or a.warmup < 1:
        raise ValueError("Positive count and warmup required")
    model = YOLO(a.weights)

    def sync():
        if a.device == "mps":
            torch.mps.synchronize()
        elif a.device.startswith("cuda"):
            torch.cuda.synchronize()

    class TimedPredictor(DetectionPredictor):
        """Synchronize each stage because Ultralytics Profile excludes MPS."""

        def measure(self, name, method, *args, **kwargs):
            sync()
            start = time.perf_counter()
            value = method(*args, **kwargs)
            sync()
            if not hasattr(self, "stage_ms"):
                self.stage_ms = {}
            self.stage_ms[name] = (time.perf_counter() - start) * 1000
            return value

        def preprocess(self, *args, **kwargs):
            value = self.measure("preprocess_ms", super().preprocess, *args, **kwargs)
            self.input_shape = list(value.shape)
            return value

        def inference(self, *args, **kwargs):
            return self.measure("inference_ms", super().inference, *args, **kwargs)

        def postprocess(self, *args, **kwargs):
            return self.measure("postprocess_ms", super().postprocess, *args, **kwargs)

    def predict(row):
        return model.predict(
            str(root / row["image"]),
            device=a.device,
            predictor=TimedPredictor,
            imgsz=640,
            batch=1,
            conf=0.25,
            iou=0.7,
            max_det=300,
            verbose=False,
            save=False,
        )[0]

    for i in range(a.warmup):
        predict(chosen[i % len(chosen)])
        sync()
    records = []
    for row in chosen:
        sync()
        start = time.perf_counter()
        predict(row)
        sync()
        elapsed = (time.perf_counter() - start) * 1000
        records.append(
            dict(
                image_id=row["image_id"],
                end_to_end_ms=elapsed,
                input_shape=model.predictor.input_shape,
                **model.predictor.stage_ms,
            )
        )
    summary = {
        k: summarize([r[k] for r in records])
        for k in ["preprocess_ms", "inference_ms", "postprocess_ms", "end_to_end_ms"]
    }
    save_json(
        target,
        dict(
            scope="Proper subset checkpoint, sequential batch-one still-image benchmark",
            weights_sha256=sha256(a.weights),
            validation_manifest_sha256=sha256(root / "val_manifest.json"),
            device=a.device,
            batch=1,
            imgsz=640,
            warmup_runs=a.warmup,
            measured_images=len(chosen),
            seed=42,
            confidence=0.25,
            nms_iou=0.7,
            max_det=300,
            summary=summary,
            records=records,
            end_to_end_definition="Wall time around predict(path), including local image read/decode, preprocessing, inference and postprocessing plus device synchronization; excludes model loading, drawing, video capture and UI.",
            realtime_claim=False,
            stage_timing="Explicit torch.mps.synchronize before and after each stage; no reliance on unsynchronized result.speed. End-to-end includes these instrumentation barriers.",
            preprocessing="OpenCV local path read/decode, stride-aligned rectangular letterbox to imgsz=640, BGR-to-RGB, CHW float32 tensor /255; actual tensor shapes recorded per sample.",
            postprocessing="Confidence filtering and class-aware NMS at conf .25 / IoU .7 / max_det 300; box scaling and Results construction.",
            model_precision="float32",
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
