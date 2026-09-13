"""Read-only verification of the completed immutable Phase 1 run."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO
from src.data.common import ROOT, sha256, save_json, paths

NAME = "yolov8n_uvh26_mv_baseline_seed42_v1"
DATA = ROOT / "data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2"


def main():
    prov = json.loads((ROOT / f"reports/tables/{NAME}_provenance.json").read_text())
    frozen = json.loads(
        (ROOT / "reports/audit/baseline_subset_frozen_provenance.json").read_text()
    )
    assert prov["status"] == "completed" and prov["epochs_completed"] == 30
    assert prov["dataset_version"] == frozen
    checks = {}
    for key, path, expected in [
        (
            "mapping",
            ROOT / "configs/class_mapping.yaml",
            frozen["class_mapping_sha256"],
        ),
        ("manifest", DATA / "manifest.json", frozen["combined_manifest_sha256"]),
        (
            "train_manifest",
            DATA / "train_manifest.json",
            frozen["manifest_sha256"]["train"],
        ),
        ("val_manifest", DATA / "val_manifest.json", frozen["manifest_sha256"]["val"]),
        (
            "audit",
            ROOT / "reports/audit/baseline_subset_final_v2_image_audit.json",
            frozen["image_audit_sha256"],
        ),
        ("config", ROOT / "configs/baseline_yolov8n.yaml", prov["config_sha256"]),
        (
            "training_source",
            ROOT / "src/training/train_baseline.py",
            prov["source_sha256"],
        ),
        ("requirements", ROOT / "requirements.txt", prov["requirements_sha256"]),
    ]:
        checks[key] = sha256(path) == expected
    assert all(checks.values()), checks
    spec = yaml.safe_load((DATA / "dataset.yaml").read_text())
    args = yaml.safe_load((ROOT / f"runs/{NAME}/args.yaml").read_text())
    assert Path(spec["path"]).resolve() == DATA.resolve()
    assert Path(args["data"]).resolve() == DATA / "dataset.yaml"
    assert spec["train"] == "images/train" and spec["val"] == "images/val"
    for key, value in prov["config"].items():
        assert args[key] == value, (key, args[key], value)
    checkpoints = {}
    for key in ["weights", "last_weights"]:
        item = prov[key]
        weight = ROOT / item["path"]
        assert (
            weight.stat().st_size == item["bytes"] and sha256(weight) == item["sha256"]
        )
        model = YOLO(weight)
        assert model.names == spec["names"] and len(model.names) == 14
        assert all(
            bool(torch.isfinite(v).all()) for v in model.model.state_dict().values()
        )
        checkpoints[key] = {
            **item,
            "readable": True,
            "finite_tensors": True,
            "stored_epoch": model.ckpt.get("epoch"),
            "epoch_note": "Ultralytics strips optimizer and resets stored epoch to -1; best epoch comes from on_model_save provenance and results.csv.",
        }
    rows = json.loads((DATA / "manifest.json").read_text())

    def check(row):
        image, label = DATA / row["image"], DATA / row["label"]
        return (
            image.resolve() == (paths()["raw"] / row["source"]).resolve()
            and sha256(image) == row["source_sha256"]
            and sha256(label) == row["label_sha256"]
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(check, rows)), "Image/label checksum or path mismatch"
    assert {s: sum(r["split"] == s for r in rows) for s in ["train", "val"]} == {
        "train": 8000,
        "val": 2000,
    }
    df = pd.read_csv(ROOT / f"runs/{NAME}/results.csv")
    df.columns = df.columns.str.strip()
    assert list(df.epoch) == list(range(1, 31))
    assert np.isfinite(df.to_numpy()).all()
    best_epoch = int(df.loc[df["metrics/mAP50-95(B)"].idxmax(), "epoch"])
    assert best_epoch == prov["best_epoch"] == 30 and not prov["early_stopped"]
    log = (ROOT / "data/interim/baseline_training_v1.log").read_text(errors="replace")
    warning_lines = sorted(
        set(
            line.strip()
            .replace(str(ROOT), "<project>")
            .replace("/Users/runner/work/pytorch/pytorch/", "<torch-build>/")
            for line in log.splitlines()
            if "UserWarning:" in line or "WARNING" in line
        )
    )
    failures = [
        line for line in log.splitlines() if "Traceback (most recent call last)" in line
    ]
    assert not failures
    inventory = {
        str(p.relative_to(ROOT)): sha256(p)
        for p in (ROOT / f"runs/{NAME}").rglob("*")
        if p.is_file()
    }
    save_json(
        ROOT / "reports/audit/baseline_closeout_integrity.json",
        dict(
            status="passed",
            experiment_id=NAME,
            process_exit_code=0,
            exit_evidence="Original exec session 42980 returned exit_code 0 when polled after training completion.",
            epochs_completed=30,
            best_epoch=best_epoch,
            early_stopped=False,
            duration_seconds=prov["duration_seconds"],
            epoch_csv_elapsed_seconds=float(df.iloc[-1]["time"]),
            checksums=checks,
            frozen_manifest_sha256=frozen["combined_manifest_sha256"],
            images_and_labels_verified=len(rows),
            class_count=14,
            checkpoints=checkpoints,
            warnings=warning_lines,
            failures=failures,
            training_log_sha256=sha256(ROOT / "data/interim/baseline_training_v1.log"),
            run_file_sha256=inventory,
            scope="Frozen MV subset only; 8000 training / 2000 validation images",
            effective_config=prov["config"],
            effective_amp=prov["effective_amp"],
            effective_workers=prov["effective_workers"],
            initial_git_commit=prov["git_commit"],
            git_note="Training began before first repository commit; source/config/requirements hashes are preserved.",
        ),
    )
    print(
        "PASS: 30 epochs; best epoch 30; both checkpoints readable; 10000 image/label hashes verified"
    )
    print("Warnings:", warning_lines)


if __name__ == "__main__":
    main()
