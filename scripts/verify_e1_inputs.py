"""Verify E1 inputs without rewriting frozen E0 evidence."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from ultralytics import YOLO
from src.data.common import ROOT, sha256, save_json, paths

DATA = ROOT / "data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2"
E0 = "yolov8n_uvh26_mv_baseline_seed42_v1"


def main():
    frozen = json.loads(
        (ROOT / "reports/audit/baseline_subset_frozen_provenance.json").read_text()
    )
    run = json.loads((ROOT / f"reports/tables/{E0}_provenance.json").read_text())
    checks = {}
    for key, path, digest in [
        ("manifest", DATA / "manifest.json", frozen["combined_manifest_sha256"]),
        (
            "train_manifest",
            DATA / "train_manifest.json",
            frozen["manifest_sha256"]["train"],
        ),
        ("val_manifest", DATA / "val_manifest.json", frozen["manifest_sha256"]["val"]),
        (
            "mapping",
            ROOT / "configs/class_mapping.yaml",
            frozen["class_mapping_sha256"],
        ),
        (
            "baseline_config",
            ROOT / "configs/baseline_yolov8n.yaml",
            run["config_sha256"],
        ),
        (
            "training_source",
            ROOT / "src/training/train_baseline.py",
            run["source_sha256"],
        ),
        ("requirements", ROOT / "requirements.txt", run["requirements_sha256"]),
        (
            "best_checkpoint",
            ROOT / run["weights"]["path"],
            "85b9e089ec3dfaa676e30a6191b7e5726f320d5bf85391f1675922b88f9778b3",
        ),
    ]:
        checks[key] = sha256(path) == digest
    assert all(checks.values()), checks
    spec = yaml.safe_load((DATA / "dataset.yaml").read_text())
    assert (
        Path(spec["path"]).resolve() == DATA
        and spec["train"] == "images/train"
        and spec["val"] == "images/val"
    )
    model = YOLO(ROOT / run["weights"]["path"])
    assert model.names == spec["names"] and len(model.names) == 14
    assert all(bool(torch.isfinite(v).all()) for v in model.model.state_dict().values())
    rows = json.loads((DATA / "manifest.json").read_text())
    raw = paths()["raw"]

    def check(r):
        return (
            (DATA / r["image"]).resolve() == (raw / r["source"]).resolve()
            and sha256(DATA / r["image"]) == r["source_sha256"]
            and sha256(DATA / r["label"]) == r["label_sha256"]
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(check, rows))
    counts = {s: sum(r["split"] == s for r in rows) for s in ["train", "val"]}
    assert counts == {"train": 8000, "val": 2000}
    old = json.loads(
        (ROOT / "reports/audit/baseline_closeout_integrity.json").read_text()
    )
    assert all(sha256(ROOT / p) == h for p, h in old["run_file_sha256"].items())
    save_json(
        ROOT / "reports/audit/E1_input_verification.json",
        dict(
            status="passed",
            checksums=checks,
            images_and_labels_verified=10000,
            split_counts=counts,
            class_mapping="14 classes match",
            dataset_paths="passed",
            E0_run_unchanged=True,
            manifest_sha256=frozen["combined_manifest_sha256"],
            val_manifest_sha256=frozen["manifest_sha256"]["val"],
        ),
    )
    print(
        "E1 input verification passed: frozen E0, 8000/2000 images+labels, mapping and dataset paths"
    )


if __name__ == "__main__":
    main()
