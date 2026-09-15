"""Verify E1 checkpoint/provenance before any measured evaluation."""

import argparse
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

NAME = "E1_yolov8s_uvh26_mv_640_seed42"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--process-exit-code", type=int, required=True)
    a = p.parse_args()
    assert a.process_exit_code == 0, "Nonzero training exit; preserve and stop"
    prov = json.loads((ROOT / f"reports/tables/{NAME}_provenance.json").read_text())
    base = json.loads(
        (
            ROOT / "reports/tables/yolov8n_uvh26_mv_baseline_seed42_v1_provenance.json"
        ).read_text()
    )
    cfg = yaml.safe_load((ROOT / f"configs/{NAME}.yaml").read_text())
    assert prov["status"] == "completed" and not prov["preflight"] and not prov["smoke"]
    assert prov["config_sha256"] == sha256(ROOT / f"configs/{NAME}.yaml")
    assert (
        prov["pretrained_sha256"]
        == "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a"
    )
    for key in [
        "dataset_manifest_sha256",
        "split_manifest_sha256",
        "class_mapping_sha256",
        "requirements_sha256",
    ]:
        assert prov[key] == base[key], key
    df = pd.read_csv(ROOT / f"runs/{NAME}/results.csv")
    df.columns = df.columns.str.strip()
    assert (
        len(df) == prov["epochs_completed"]
        and 1 <= len(df) <= 30
        and np.isfinite(df.to_numpy()).all()
    )
    assert list(df.epoch) == list(range(1, len(df) + 1))
    assert len(df) == 30 or prov["early_stopped"], (
        "Short run without recorded early stopping"
    )
    assert int(prov["best_epoch"]) in df.epoch.values
    spec = yaml.safe_load(
        (
            ROOT
            / "data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2/dataset.yaml"
        ).read_text()
    )
    args = yaml.safe_load((ROOT / f"runs/{NAME}/args.yaml").read_text())
    for key, value in cfg.items():
        if key != "model":
            assert args[key] == value, (key, args[key], value)
    checks = {}
    for key in ["weights", "last_weights"]:
        w = prov[key]
        path = ROOT / w["path"]
        assert path.stat().st_size == w["bytes"] and sha256(path) == w["sha256"]
        model = YOLO(path)
        assert model.names == spec["names"]
        assert all(
            bool(torch.isfinite(v).all()) for v in model.model.state_dict().values()
        )
        checks[key] = {
            **w,
            "readable": True,
            "finite": True,
            "stored_epoch": model.ckpt.get("epoch"),
        }
    e0 = json.loads(
        (ROOT / "reports/audit/baseline_closeout_integrity.json").read_text()
    )
    assert all(
        sha256(ROOT / path) == digest for path, digest in e0["run_file_sha256"].items()
    )
    differences = []
    for key, value in {**base["config"], **base["augmentation"]}.items():
        expected = (
            base["effective_amp"]
            if key == "amp"
            else base["effective_workers"]
            if key == "workers"
            else value
        )
        actual = prov["config"][key]
        if actual != expected:
            differences.append(dict(setting=key, E0_effective=expected, E1=actual))
    assert not differences, differences
    log = (ROOT / "data/interim/E1_training.log").read_text(errors="replace")
    assert "Traceback (most recent call last)" not in log
    warnings = sorted(
        set(
            line.strip()
            .replace(str(ROOT), "<project>")
            .replace("/Users/runner/work/pytorch/pytorch/", "<torch-build>/")
            for line in log.splitlines()
            if "UserWarning:" in line or "WARNING" in line
        )
    )
    report = dict(
        status="passed",
        process_exit_code=a.process_exit_code,
        epochs_completed=len(df),
        best_epoch=prov["best_epoch"],
        early_stopped=prov["early_stopped"],
        effective_control_differences=differences,
        E0_run_unchanged=True,
        checkpoints=checks,
        warnings=warnings,
        run_file_sha256={
            str(path.relative_to(ROOT)): sha256(path)
            for path in (ROOT / f"runs/{NAME}").rglob("*")
            if path.is_file()
        },
    )
    save_json(ROOT / "reports/audit/E1_completed_run_verification.json", report)
    print("E1 completed-run verification passed")


if __name__ == "__main__":
    main()
