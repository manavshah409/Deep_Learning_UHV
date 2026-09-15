"""Validate completed E1 preflight and export a few local prediction previews."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO
from src.data.common import ROOT, sha256, save_json

NAME = "E1_yolov8s_uvh26_mv_640_seed42_preflight_v2"
DATA = ROOT / "data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2"


def main():
    prov = json.loads((ROOT / f"reports/tables/{NAME}_provenance.json").read_text())
    assert (
        prov["status"] == "completed"
        and prov["epochs_completed"] == 1
        and prov["preflight"]
    )
    cfg = yaml.safe_load(
        (ROOT / "configs/E1_yolov8s_uvh26_mv_640_seed42.yaml").read_text()
    )
    assert prov["config_sha256"] == sha256(
        ROOT / "configs/E1_yolov8s_uvh26_mv_640_seed42.yaml"
    )
    for k, v in prov["config"].items():
        assert v == (1 if k == "epochs" else 0.1 if k == "fraction" else cfg[k]), k
    base = json.loads(
        (
            ROOT / "reports/tables/yolov8n_uvh26_mv_baseline_seed42_v1_provenance.json"
        ).read_text()
    )
    assert prov["dataset_manifest_sha256"] == base["dataset_manifest_sha256"]
    assert prov["split_manifest_sha256"] == base["split_manifest_sha256"]
    assert prov["class_mapping_sha256"] == base["class_mapping_sha256"]
    assert (
        prov["effective_batch"] == 8
        and prov["effective_amp"] is False
        and prov["effective_workers"] == 0
    )
    assert (
        prov["optimizer_class"] == "AdamW"
        and prov["startup_optimizer"]["accumulate"] == 8
    )
    df = pd.read_csv(ROOT / f"runs/{NAME}/results.csv")
    assert len(df) == 1 and np.isfinite(df.to_numpy()).all()
    spec = yaml.safe_load((DATA / "dataset.yaml").read_text())
    for k in ["weights", "last_weights"]:
        obj = prov[k]
        path = ROOT / obj["path"]
        assert path.stat().st_size == obj["bytes"] and sha256(path) == obj["sha256"]
        model = YOLO(path)
        assert model.names == spec["names"]
        assert all(
            bool(torch.isfinite(t).all()) for t in model.model.state_dict().values()
        )
    model = YOLO(ROOT / prov["weights"]["path"])
    out = ROOT / "reports/predictions/E1_preflight_v2"
    if out.exists():
        raise ValueError("Preflight prediction output exists")
    out.mkdir(parents=True)
    rows = {
        r["image_id"]: r for r in json.loads((DATA / "val_manifest.json").read_text())
    }
    preview = []
    for cid in [4232, 4711, 954]:
        r = model.predict(
            str(DATA / rows[cid]["image"]),
            device="mps",
            imgsz=640,
            conf=0.25,
            iou=0.7,
            max_det=300,
            verbose=False,
        )[0]
        r.save(filename=str(out / f"val_{cid}.jpg"))
        preview.append(
            dict(
                image_id=cid,
                predictions=len(r.boxes),
                classes=sorted({r.names[int(x)] for x in r.boxes.cls.cpu()}),
            )
        )
    run_bytes = sum(
        p.stat().st_size for p in (ROOT / f"runs/{NAME}").rglob("*") if p.is_file()
    )
    wall = prov["duration_seconds"]
    failed = json.loads(
        (ROOT / "reports/audit/E1_preflight_v1_failure.json").read_text()
    )
    training_pass_wall = failed["duration_seconds"]
    assert prov["startup_optimizer"]["training_images"] == 800
    assert prov["startup_optimizer"]["validation_images"] == 2000
    save_json(
        ROOT / "reports/audit/E1_preflight_verification.json",
        dict(
            status="integrity_passed_awaiting_visual_review",
            epochs=1,
            checkpoints="readable_finite_hashes_match",
            controls="Recovery preflight: first 800 training images via fraction=.1; all 2000 validation images; batch8/AdamW unchanged. Proper E1 remains 8000/2000 and 30 epochs.",
            duration_seconds=wall,
            observed_run_bytes=run_bytes,
            full_30_epoch_duration_estimate_seconds=(training_pass_wall + wall) * 30,
            duration_estimate_range_seconds=[
                training_pass_wall * 30,
                (training_pass_wall + wall) * 30,
            ],
            duration_estimate_note="Lower bound scales the observed full 1000-batch v1 training pass; conservative upper bound adds the entire short v2 preflight wall time per epoch, including validation twice. Estimates only; thermal/load variation can change throughput.",
            storage_reservation_bytes=max(1024**3, run_bytes * 3),
            storage_note="Conservative reservation, not measured final E1 storage. Includes unstripped optimizer checkpoints and plots.",
            previews=preview,
            accuracy_result=False,
        ),
    )
    print("E1 preflight integrity passed; three prediction previews ready")


if __name__ == "__main__":
    main()
