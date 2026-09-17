"""Standalone validation with retained validator and atomic complete-result bundles."""

import argparse
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
import numpy as np
from src.data.common import ROOT, sha256


def extract_metrics(metrics, names):
    """Read DetMetrics only; never look for a validator on a model."""
    from ultralytics.utils.metrics import smooth

    try:
        b = metrics.box
        ids = [int(x) for x in b.ap_class_index]
        if set(ids) != set(range(len(names))):
            raise ValueError("Expected evaluated metrics for every frozen class")
        rows = [
            dict(
                class_id=c,
                name=names[c],
                precision=float(b.p[i]),
                recall=float(b.r[i]),
                f1=float(b.f1[i]),
                ap50=float(b.ap50[i]),
                ap50_95=float(b.ap[i]),
            )
            for i, c in enumerate(ids)
        ]
        p, r = float(b.mp), float(b.mr)
        result = dict(
            precision=p,
            recall=r,
            f1=2 * p * r / (p + r) if p + r else 0.0,
            macro_f1=float(b.f1.mean()),
            map50=float(b.map50),
            map50_95=float(b.map),
            f1_operating_confidence=float(
                b.px[int(np.argmax(smooth(b.f1_curve.mean(0), 0.1)))]
            ),
        )
        values = list(result.values()) + [
            v for row in rows for k, v in row.items() if k not in ("name", "class_id")
        ]
        if not np.isfinite(values).all() or min(values) < 0 or max(values) > 1:
            raise ValueError("Metrics must be finite probabilities")
        return result, rows
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError(f"Required DetMetrics field unavailable: {exc}") from exc


def publish_bundle(target, result, rows, confusion, counts, figures):
    """Only a successful directory rename exposes a complete evaluation."""
    import pandas as pd

    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + target.name + "-", dir=target.parent))
    try:
        for name, value in [
            ("metrics", result),
            ("confusion_matrix", confusion),
            ("prediction_counts", counts),
        ]:
            (staging / f"{name}.json").write_text(
                json.dumps(value, indent=2, allow_nan=False) + "\n"
            )
        pd.DataFrame(rows).to_csv(staging / "per_class.csv", index=False)
        for path in figures:
            shutil.copy2(path, staging / path.name)
        (staging / "COMPLETE.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "files_sha256": {p.name: sha256(p) for p in staging.iterdir()},
                },
                indent=2,
            )
            + "\n"
        )
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser()
    for key in ["weights", "data", "name"]:
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=8)
    a = parser.parse_args()
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
            retained.append(self)

        def update_metrics(self, preds, batch):
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
        imgsz=640,
        batch=a.batch,
        device=a.device,
        workers=0,
        plots=True,
        conf=0.001,
        iou=0.7,
        max_det=300,
        project=str(ROOT / "runs"),
        name=a.name,
    )
    v = retained[0]
    result, rows = extract_metrics(m, yolo.names)
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
        imgsz=640,
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
        source_sha256=sha256(__file__),
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
    publish_bundle(target, result, rows, cm, v.export_counts, figures)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
