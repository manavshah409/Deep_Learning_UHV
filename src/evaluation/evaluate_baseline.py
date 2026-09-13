"""Evaluate best weights and preserve actual per-class metrics and plots."""

import argparse
from pathlib import Path
import shutil
import time
import numpy as np
from src.data.common import ROOT, save_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    from ultralytics import YOLO
    from ultralytics.utils.metrics import smooth
    import pandas as pd
    import torch

    model = YOLO(a.weights)
    if a.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no silent CPU fallback")
    device = a.device
    output = ROOT / "runs" / a.name
    if output.exists():
        raise ValueError("Evaluation name already exists")
    start = time.perf_counter()
    m = model.val(
        data=str(Path(a.data).resolve()),
        split="val",
        imgsz=640,
        batch=a.batch,
        device=device,
        workers=0,
        plots=True,
        conf=0.001,
        iou=0.7,
        max_det=300,
        project=str(ROOT / "runs"),
        name=a.name,
    )
    rows = []
    for i, cid in enumerate(m.box.ap_class_index):
        precision = float(m.box.p[i])
        recall = float(m.box.r[i])
        rows.append(
            {
                "class_id": int(cid),
                "name": model.names[int(cid)],
                "precision": precision,
                "recall": recall,
                "f1": float(m.box.f1[i]),
                "ap50": float(m.box.ap50[i]),
                "ap50_95": float(m.box.ap[i]),
            }
        )
    pd.DataFrame(rows).to_csv(
        ROOT / "reports/tables" / f"{a.name}_per_class.csv", index=False
    )
    speed = {k: float(v) for k, v in m.speed.items()}
    infer = speed["inference"]
    result = {
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "f1": 2
        * float(m.box.mp)
        * float(m.box.mr)
        / (float(m.box.mp) + float(m.box.mr))
        if (m.box.mp + m.box.mr)
        else 0.0,
        "macro_f1": float(m.box.f1.mean()),
        "scope": "UVH-26 MV 8,000/2,000 subset validation results",
        "validation_manifest_sha256": sha256(
            Path(a.data).resolve().parent / "val_manifest.json"
        ),
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "speed_ms_per_image": speed,
        "inference_only_fps": 1000 / infer if infer else None,
        "pipeline_fps": 1000 / sum(speed.values()) if sum(speed.values()) else None,
        "fps_note": "Ultralytics validation stage profiler does not synchronize MPS. Stage times and derived throughput are diagnostic only; use the explicit synchronized batch-one latency report.",
        "f1_note": "f1 is the harmonic mean of overall reported Precision and Recall; macro_f1 is the mean of per-class F1. Ultralytics selects the confidence operating point.",
        "weights_bytes": Path(a.weights).stat().st_size,
        "weights_sha256": sha256(a.weights),
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "duration_seconds": time.perf_counter() - start,
        "device": device,
        "batch": a.batch,
        "imgsz": 640,
        "split": "val",
        "confidence_floor": 0.001,
        "nms_iou": 0.7,
        "max_det": 300,
        "f1_operating_confidence": float(
            m.box.px[int(np.argmax(smooth(m.box.f1_curve.mean(0), 0.1)))]
        ),
    }
    save_json(ROOT / "reports/tables" / f"{a.name}_metrics.json", result)
    save_json(
        ROOT / "reports/tables" / f"{a.name}_confusion_matrix.json",
        {
            "names": model.names,
            "matrix": m.confusion_matrix.matrix.tolist(),
            "axes": "rows predicted, columns true; final row/column background",
            "confidence": float(model.validator.confusion_matrix_conf),
            "matching_iou": 0.45,
        },
    )
    for path in output.glob("*.png"):
        if "batch" not in path.name:
            shutil.copy2(path, ROOT / "reports/figures" / f"{a.name}_{path.name}")
    print(result)


if __name__ == "__main__":
    main()
