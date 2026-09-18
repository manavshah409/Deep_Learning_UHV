"""Standalone validation with retained validator and atomic complete-result bundles."""

import argparse
import inspect
import json
from pathlib import Path
import platform
import time
import numpy as np
from src.data.common import ROOT, sha256


from src.evaluation.evaluate_baseline import extract_metrics, publish_bundle
from src.evaluation.e2_size_ap import evaluate_sizes


def main():
    parser = argparse.ArgumentParser()
    for key in ["weights", "data", "name"]:
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, choices=[640, 960], required=True)
    a = parser.parse_args()
    if f"_{a.imgsz}_" not in a.name:
        raise ValueError("Resolution-specific output ID required")
    source_at_start = sha256(__file__)
    size_source_at_start = sha256(ROOT / "src/evaluation/e2_size_ap.py")
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import ConfusionMatrix

    output = ROOT / "runs" / a.name
    target = ROOT / "reports/evaluations" / a.name
    if output.exists() or target.exists():
        raise FileExistsError(
            "Unique evaluation ID required; existing outputs preserved"
        )
    if a.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    retained = []

    class RecordedValidator(DetectionValidator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.export_counts = []
            self.memory_samples = []
            retained.append(self)

        def update_metrics(self, preds, batch):
            self.memory_samples.append(
                dict(
                    allocated=torch.mps.current_allocated_memory(),
                    driver=torch.mps.driver_allocated_memory(),
                )
            )
            for pred, path in zip(preds, batch["im_file"]):
                cls = pred["cls"].cpu().numpy().astype(int)
                self.export_counts.append(
                    {
                        "image": Path(path).name,
                        "detections": len(cls),
                        "per_class": np.bincount(cls, minlength=14).tolist(),
                    }
                )
            return super().update_metrics(preds, batch)

    yolo = YOLO(a.weights)
    start = time.perf_counter()
    m = yolo.val(
        validator=RecordedValidator,
        data=str(Path(a.data).resolve()),
        split="val",
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=0,
        plots=True,
        save_json=True,
        conf=0.001,
        iou=0.7,
        max_det=300,
        project=str(ROOT / "runs"),
        name=a.name,
    )
    v = retained[0]
    result, rows = extract_metrics(m, yolo.names)
    result["memory"] = {
        "method": "Maximum MPS current allocated and driver allocated bytes sampled once per validation batch; not a hardware absolute peak",
        "samples": len(v.memory_samples),
        "sampled_max_allocated_bytes": max(x["allocated"] for x in v.memory_samples),
        "sampled_max_driver_bytes": max(x["driver"] for x in v.memory_samples),
    }
    manifest = Path(a.data).resolve().parent / "val_manifest.json"
    expected = json.loads(manifest.read_text())
    if v.seen != 2000 or len(v.export_counts) != len(expected):
        raise ValueError("Validation count differs from frozen 2000 images")
    if sorted(x["image"] for x in v.export_counts) != sorted(
        Path(x["image"]).name for x in expected
    ):
        raise ValueError("Validation image identities differ")
    matrix = m.confusion_matrix.matrix
    if matrix.shape != (15, 15) or not np.isfinite(matrix).all():
        raise ValueError("Invalid confusion matrix")
    cm = dict(
        names=yolo.names,
        matrix=matrix.tolist(),
        axes="rows predicted, columns true; final row/column background",
        confidence=float(v.confusion_matrix_conf),
        matching_iou=float(
            inspect.signature(ConfusionMatrix.process_batch)
            .parameters["iou_thres"]
            .default
        ),
    )
    result.update(
        scope="Frozen UVH-26 MV 8000/2000 subset validation; not independent test results",
        validation_manifest_sha256=sha256(manifest),
        dataset_yaml_sha256=sha256(a.data),
        class_mapping_sha256=sha256(ROOT / "configs/class_mapping.yaml"),
        weights_sha256=sha256(a.weights),
        weights_bytes=Path(a.weights).stat().st_size,
        parameters=sum(p.numel() for p in yolo.model.parameters()),
        duration_seconds=time.perf_counter() - start,
        device=a.device,
        batch=a.batch,
        imgsz=a.imgsz,
        workers=0,
        split="val",
        confidence_floor=0.001,
        nms_iou=0.7,
        max_det=300,
        ap_iou_thresholds=v.iouv.cpu().tolist(),
        evaluated_images=v.seen,
        ground_truth_objects=int(np.asarray(m.nt_per_class).sum()),
        prediction_count=sum(x["detections"] for x in v.export_counts),
        f1_note="Harmonic aggregate F1=2*macro_P*macro_R/(macro_P+macro_R); macro_f1 averages class F1. Neither is micro-F1. Each model selects its max smoothed mean-class-F1 confidence.",
        python=platform.python_version(),
        torch=torch.__version__,
        ultralytics=ultralytics.__version__,
        platform=platform.platform(),
        source_sha256=source_at_start,
        wrapper_type=type(yolo).__name__,
        internal_model_type=type(yolo.model).__name__,
        metrics_type=type(m).__name__,
    )
    figures = [p for p in output.glob("*.png") if "batch" not in p.name]
    required = {
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "confusion_matrix.png",
    }
    if not required.issubset({p.name for p in figures}):
        raise ValueError("Required evaluation plots missing")
    sizes = evaluate_sizes(
        Path(a.data).resolve().parent, v.jdict, yolo.names, v.class_map
    )
    sizefile = output / "size_ap.json"
    sizefile.write_text(json.dumps(sizes, indent=2, allow_nan=False) + "\n")
    figures.append(sizefile)
    result["size_ap_source_sha256"] = size_source_at_start
    result["shared_metric_source_sha256"] = sha256(
        ROOT / "src/evaluation/evaluate_baseline.py"
    )
    result["pycocotools"] = "2.0.11"
    publish_bundle(target, result, rows, cm, v.export_counts, figures)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
