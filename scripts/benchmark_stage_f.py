"""Actual sequential E1+E3+fusion inference on the existing calibration100 sample."""

import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage_f_fusion import DATA, OUT, E, note, save, table
from src.evaluation.comparison_protocol import (
    NAMES,
    OutsideImage,
    detection,
    guard_path,
    load_manifest,
    sha,
    unletterbox_boxes,
    verify_seal,
)
from src.evaluation.fusion import fuse
from src.experiments.faster_rcnn import build_model


def sync():
    torch.mps.synchronize()


def main():
    if (OUT / "latency.json").exists():
        raise FileExistsError("Immutable latency report exists")
    try:
        # Revalidate historical hashes without replacing the original F1 integrity record.
        integrity = json.loads((OUT / "integrity.json").read_text())
        for path, expected in integrity["preservation"].items():
            if sha(ROOT / path) != expected:
                raise ValueError(f"Historical preservation mismatch: {path}")
        final = verify_seal(json.loads((OUT / "final_amendment.json").read_text()))
        _plan = verify_seal(json.loads((OUT / "protocol.json").read_text()))
        for path, expected in final["source_sha256"].items():
            if sha(ROOT / path) != expected:
                raise ValueError("Fusion source hash mismatch")
        rows = load_manifest()
        allowed = {r["image_id"]: r for r in rows}
        sample_path = (
            ROOT
            / "reports/evaluations/E3_fasterrcnn_best_calibration500_v1/benchmark_sample.json"
        )
        sample = json.loads(sample_path.read_text())
        if len(sample) != 100 or len({r["image_id"] for r in sample}) != 100:
            raise ValueError("Invalid benchmark sample")
        for r in sample:
            if r != allowed.get(r["image_id"]):
                raise ValueError("Benchmark sample outside calibration allowlist")
            path = guard_path(DATA / r["image"], DATA, rows)
            if sha(path) != r["source_sha256"]:
                raise ValueError("Benchmark image hash mismatch")
        cached = {}
        configs = {}
        for name in ["E1", "E3"]:
            receipt = json.loads((E / f"{name}_prediction_receipt.json").read_text())
            cfg = receipt["inference_configuration"]
            configs[name] = cfg
            if sha(ROOT / cfg["checkpoint"]) != receipt["checkpoint_sha256"]:
                raise ValueError("Checkpoint hash mismatch")
            path = ROOT / "runs" / receipt["id"] / "predictions.jsonl"
            if sha(path) != receipt["prediction_sha256"]:
                raise ValueError("Prediction bundle hash mismatch")
            cached[name] = {
                r["image_id"]: r for r in [json.loads(line) for line in path.open()]
            }
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        from ultralytics import YOLO
        from ultralytics.engine.results import Results
        from ultralytics.models.yolo.detect.predict import DetectionPredictor

        class TimedPredictor(DetectionPredictor):
            def timed(self, key, function, *args, **kwargs):
                sync()
                start = time.perf_counter()
                result = function(*args, **kwargs)
                sync()
                self.measured[key] = (time.perf_counter() - start) * 1000
                return result

            def preprocess(self, images):
                self.measured = {}
                result = self.timed("preprocess_ms", super().preprocess, images)
                self.shape = list(result.shape)
                return result

            def inference(self, *args, **kwargs):
                return self.timed("forward_ms", super().inference, *args, **kwargs)

            def postprocess(self, *args, **kwargs):
                return self.timed(
                    "postprocess_ms", super().postprocess, *args, **kwargs
                )

            def construct_result(self, pred, img, orig_img, img_path):
                sync()
                start = time.perf_counter()
                pred[:, :4] = unletterbox_boxes(
                    pred[:, :4], img.shape[2:], orig_img.shape[:2]
                )
                result = Results(
                    orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]
                )
                sync()
                self.measured["coordinate_ms"] = (time.perf_counter() - start) * 1000
                return result

        start = time.perf_counter()
        yolo = YOLO(ROOT / configs["E1"]["checkpoint"])
        frcnn = build_model()
        payload = torch.load(
            ROOT / configs["E3"]["checkpoint"], map_location="cpu", weights_only=True
        )
        frcnn.load_state_dict(payload["model"])
        del payload
        frcnn.to("mps").float().eval()
        sync()
        loading = time.perf_counter() - start
        if [yolo.names[i] for i in range(14)] != NAMES:
            raise ValueError("YOLO class mapping mismatch")

        def common(name, row, boxes, scores, classes, w, h):
            ds = []
            for box, score, cls in zip(boxes, scores, classes):
                try:
                    d = detection(box, score, cls, name, w, h)
                except OutsideImage:
                    continue
                if d is not None:
                    ds.append(d)
            return {
                "image_id": row["image_id"],
                "width": w,
                "height": h,
                "boxes": [d["xyxy"] for d in ds],
                "scores": [d["confidence"] for d in ds],
                "classes": [d["class_id"] for d in ds],
            }

        def predict(row):
            path = DATA / row["image"]
            sync()
            start = time.perf_counter()
            r = yolo.predict(
                str(path),
                predictor=TimedPredictor,
                device="mps",
                imgsz=640,
                rect=True,
                batch=1,
                conf=0.001,
                iou=0.7,
                max_det=300,
                agnostic_nms=False,
                augment=False,
                verbose=False,
                save=False,
            )[0]
            h, w = r.orig_shape
            a = common(
                "E1",
                row,
                r.boxes.xyxy.cpu().tolist(),
                r.boxes.conf.cpu().tolist(),
                r.boxes.cls.cpu().tolist(),
                w,
                h,
            )
            sync()
            end_e1 = time.perf_counter()
            with Image.open(path) as im:
                image = to_tensor(im.convert("RGB")).to("mps")
            original = [image.shape[-2:]]
            images, _ = frcnn.transform([image], None)
            sync()
            start_e3_forward = time.perf_counter()
            features = frcnn.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])
            proposals, _ = frcnn.rpn(images, features, None)
            predictions, _ = frcnn.roi_heads(
                features, proposals, images.image_sizes, None
            )
            sync()
            end_e3_forward = time.perf_counter()
            predictions = frcnn.transform.postprocess(
                predictions, images.image_sizes, original
            )[0]
            predictions = {k: v.cpu() for k, v in predictions.items()}
            b = common(
                "E3",
                row,
                predictions["boxes"].tolist(),
                predictions["scores"].tolist(),
                predictions["labels"].tolist(),
                w,
                h,
            )
            sync()
            end_e3 = time.perf_counter()
            fused = fuse(a, b, final["configuration"], final["context"])
            kept = [i for i, s in enumerate(fused["scores"]) if s >= final["threshold"]]
            sync()
            end = time.perf_counter()
            ym = yolo.predictor.measured
            inferred_e1 = ym["forward_ms"] + ym["postprocess_ms"] - ym["coordinate_ms"]
            item = {
                "image_id": row["image_id"],
                "E1_inference_ms": inferred_e1,
                "E1_forward_only_ms": ym["forward_ms"],
                "E3_inference_ms": (end_e3_forward - start_e3_forward) * 1000,
                "fusion_postprocessing_ms": (end - end_e3) * 1000,
                "E1_detector_ms": (end_e1 - start) * 1000,
                "E3_detector_ms": (end_e3 - end_e1) * 1000,
                "end_to_end_ms": (end - start) * 1000,
                "output_detections": len(kept),
                "MPS_allocated_bytes": torch.mps.current_allocated_memory(),
                "MPS_driver_bytes": torch.mps.driver_allocated_memory(),
            }
            item["other_preprocessing_transfer_ms"] = (
                item["end_to_end_ms"]
                - item["E1_inference_ms"]
                - item["E3_inference_ms"]
                - item["fusion_postprocessing_ms"]
            )
            return item, a, b

        measured = []
        parity = []
        with torch.no_grad():
            for row in sample[:10]:
                predict(row)
            if next(yolo.model.parameters()).dtype != torch.float32:
                raise ValueError("YOLO benchmark is not float32")
            for row in sample:
                item, a, b = predict(row)
                measured.append(item)
                for name, new in [("E1", a), ("E3", b)]:
                    old = cached[name][row["image_id"]]["detections"]
                    boxes = [d["xyxy"] for d in old]
                    scores = [d["confidence"] for d in old]
                    classes = [d["class_id"] for d in old]
                    if (
                        new["classes"] != classes
                        or not np.allclose(new["boxes"], boxes, atol=1e-4, rtol=1e-5)
                        or not np.allclose(new["scores"], scores, atol=1e-6, rtol=1e-5)
                    ):
                        raise ValueError(
                            "Actual benchmark predictions differ from frozen cache"
                        )
                    parity.append(
                        {"image_id": row["image_id"], "model": name, "passed": True}
                    )
        table(OUT / "latency_samples.csv", measured)
        stats = {
            key: {
                "mean": float(np.mean([r[key] for r in measured])),
                "median": float(np.median([r[key] for r in measured])),
                "p95": float(np.percentile([r[key] for r in measured], 95)),
            }
            for key in measured[0]
            if key.endswith("_ms")
        }
        result = {
            "device": "mps",
            "dtype": "float32",
            "batch": 1,
            "warmups": 10,
            "timed_images": 100,
            "passes": 1,
            "model_loading_seconds_excluded": loading,
            "late_YOLO_backend_initialization": "Excluded in warm-up",
            "sample_file": str(sample_path.relative_to(ROOT)),
            "sample_sha256": sha(sample_path),
            "sample_ids": [r["image_id"] for r in sample],
            "configuration": final,
            "stage_ms": stats,
            "still_image_fps": 1000 / stats["end_to_end_ms"]["mean"],
            "combined_inference_only_images_per_second": 1000
            / (stats["E1_inference_ms"]["mean"] + stats["E3_inference_ms"]["mean"]),
            "boundaries": {
                "E1_inference": "YOLO forward plus NMS/postprocess overhead, minus original-coordinate mapping; internal synchronization overhead included",
                "E3_inference": "Backbone, RPN, ROI heads including box decode/NMS",
                "fusion_postprocessing": "CPU deterministic fusion and final operating confidence filtering",
                "end_to_end": "Actual sequential YOLO then FasterRCNN then fusion; two file decodes, each architecture preprocessing, MPS transfer, inference, original-coordinate conversion/CPU copy, common clipping and fusion/filter. Excludes load, integrity checks, parity checks, logging, video capture/display.",
                "sizes": "YOLO rect640; FasterRCNN short480/max640; batch1 float32",
                "memory": "Sampled MPS allocations after images, not absolute peak; both models resident",
            },
            "sampled_max_allocated_bytes": max(
                r["MPS_allocated_bytes"] for r in measured
            ),
            "sampled_max_driver_bytes": max(r["MPS_driver_bytes"] for r in measured),
            "prediction_parity_checks": len(parity),
            "benchmark_source_sha256": sha(Path(__file__)),
            "reserved_accessed": False,
            "limitations": "Warm filesystem and10warmups,onepass,singlehardware session. Still-image throughput, not live-video FPS. If fusion rejected this is cost of strongest-family fold0 representative, not a selected E4 model.",
        }
        save(OUT / "latency.json", result)
        save(OUT / "latency_prediction_parity.json", parity)
        note(
            "F6",
            "Benchmarked actual sequential E1/E3 inference plus frozen fusion on100existing calibration benchmark images.",
            ".venv/bin/python scripts/benchmark_stage_f.py",
            f"Sample {sha(sample_path)}; source {sha(Path(__file__))}",
            json.dumps(
                {
                    "mean_ms": stats["end_to_end_ms"]["mean"],
                    "median_ms": stats["end_to_end_ms"]["median"],
                    "p95_ms": stats["end_to_end_ms"]["p95"],
                    "still_image_fps": result["still_image_fps"],
                }
            ),
            "Artifact validation, complete tests and Git delivery",
        )
        print(
            json.dumps(
                {
                    "completed": True,
                    "end_to_end_ms": stats["end_to_end_ms"],
                    "still_images_per_second": result["still_image_fps"],
                }
            )
        )
    except BaseException as error:
        save(
            OUT / "LATENCY_FAILURE.json",
            {
                "stage": "F6",
                "type": type(error).__name__,
                "error": str(error),
                "reserved_accessed": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
