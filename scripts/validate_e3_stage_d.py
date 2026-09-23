"""Read-only Stage D evidence validation; never opens reserved evaluation data."""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / "reports/evaluations/E3_fasterrcnn_best_calibration500_v1"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_metrics(metrics, classes):
    assert len(classes) == 14
    assert sum(int(c["gt_count"]) for c in classes) == metrics["objects"] == 6148
    assert (
        sum(int(c["prediction_count_fixed"]) for c in classes)
        == metrics["predictions_fixed"]
    )
    for c in classes:
        tp, fp, fn = (int(c[k]) for k in ["tp", "fp", "fn"])
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f = 2 * p * r / (p + r) if p + r else 0
        assert tp + fn == int(c["gt_count"]) and tp + fp == int(
            c["prediction_count_fixed"]
        )
        for key, expected in [("precision", p), ("recall", r), ("f1", f)]:
            assert math.isclose(float(c[key]), expected, abs_tol=1e-12)
    for key, source in [
        ("precision", "precision"),
        ("recall", "recall"),
        ("macro_class_f1", "f1"),
        ("map50", "ap50"),
        ("map50_95", "ap50_95"),
    ]:
        assert math.isclose(
            metrics[key], np.mean([float(c[source]) for c in classes]), abs_tol=1e-12
        )
    p, r = metrics["precision"], metrics["recall"]
    assert math.isclose(
        metrics["harmonic_aggregate_f1"], 2 * p * r / (p + r), abs_tol=1e-12
    )


def validate(local=False):
    def read(name):
        return json.loads((REPORT / (name + ".json")).read_text())

    metrics, audit, protocol = (read(n) for n in ["metrics", "audit", "protocol"])
    validate_metrics(metrics, list(csv.DictReader((REPORT / "per_class.csv").open())))
    assert (
        metrics["images"] == 500
        and metrics["selected_epoch"] == audit["selected_epoch"] == 13
    )
    assert audit["passed"] and audit["complete"]["completed_epochs"] == 20
    assert (
        audit["complete"]["exit_status"]
        == audit["process_history"][-1]["exit_code"]
        == 0
    )
    assert read("process")["exit_code"] == 0
    assert (
        not audit["reserved_split_accessed"] and not protocol["reserved_split_accessed"]
    )
    for path, expected in protocol["evaluator_source_sha256"].items():
        assert digest(ROOT / path) == expected, path
    rows = list(csv.DictReader((REPORT / "epoch_timing.csv").open()))
    assert [int(r["epoch"]) for r in rows] == list(range(1, 21))
    for i, row in enumerate(rows):
        assert float(row["learning_rate"]) == audit["configuration"]["lr_by_epoch"][i]
        for key, value in row.items():
            if key.endswith(("_loss", "_seconds")):
                assert math.isfinite(float(value)) and float(value) >= 0
    assert math.isclose(
        sum(float(r["total_epoch_seconds"]) for r in rows),
        audit["active_completed_epoch_seconds"],
        abs_tol=1e-6,
    )
    agreement = read("agreement")
    assert agreement["passed"] and all(
        abs(v) <= protocol["numerical_tolerance_absolute"]
        for v in agreement["differences"].values()
    )
    assert metrics["checkpoint_sha256"] == audit["checkpoints"][12]["sha256"]
    assert metrics["manifest_sha256"] == audit["configuration"]["calibration_sha256"]
    assert metrics["configuration_sha256"] == audit["config_hash"]
    errors = read("image_errors")
    ids = {r["image_id"] for r in errors}
    assert len(errors) == len(ids) == 500
    sample = read("benchmark_sample")
    assert len(sample) == 100 and len({r["image_id"] for r in sample}) == 100
    assert {r["image_id"] for r in sample} <= ids
    selected = read("qualitative_selection")["samples"]
    assert len(selected) == 13 and {r["image_id"] for r in selected} <= ids
    assert read("qualitative_review")["reviewed_all_panels"]
    matrix = np.array(read("confusion_matrix")["matrix"])
    assert matrix.shape == (15, 15) and int(matrix[:14, :].sum()) == 6148
    assert int(matrix[:, :14].sum()) == metrics["predictions_fixed"]
    timings = list(csv.DictReader((REPORT / "benchmark_samples.csv").open()))
    bench = read("benchmark")
    assert len(timings) == bench["timed_images"] == 100 and bench["warmups"] == 10
    assert (
        bench["device"] == "mps"
        and bench["batch"] == 1
        and bench["split_forward_matches_standard"]
    )
    for stage, summary in bench["timings"].items():
        values = np.array([float(r[stage]) for r in timings])
        for name, actual in [
            ("mean", np.mean(values)),
            ("median", np.median(values)),
            ("p95", np.percentile(values, 95)),
        ]:
            assert math.isclose(summary[name], actual, abs_tol=1e-9)
    for key, stage in [
        ("inference_only_images_per_second", "inference_ms"),
        ("end_to_end_still_images_per_second", "end_to_end_ms"),
    ]:
        assert math.isclose(
            bench[key], 1000 / bench["timings"][stage]["mean"], abs_tol=1e-9
        )
    index = read("artifact_index")
    for path in index["summary_files"] + index["figures"]:
        assert (REPORT / path).is_file(), path
    for key in ["technical_report", "faculty_summary", "reproduction"]:
        assert (ROOT / index[key]).is_file()
    for name in index["figures"]:
        with Image.open(REPORT / name) as im:
            im.verify()
    historical = read("historical_preservation")
    assert historical["passed"]
    for path, expected in historical["hashes"].items():
        assert digest(ROOT / path) == expected, path
    if local:
        import torch

        from src.training.epoch_resume import finite_tree

        cfg = audit["configuration"]
        for path, expected in [
            (cfg["train_manifest"], cfg["train_sha256"]),
            (cfg["calibration_manifest"], cfg["calibration_sha256"]),
            ("configs/class_mapping.yaml", cfg["class_mapping_sha256"]),
            (cfg["initializer"]["path"], cfg["initializer_sha256"]),
        ]:
            assert digest(ROOT / path) == expected
        # Open only the explicit calibration500 allowlist, never val2000/reserved1500.
        calibration = json.loads((ROOT / cfg["calibration_manifest"]).read_text())
        assert {r["image_id"] for r in calibration} == ids
        ordered = sorted(calibration, key=lambda r: r["image_id"])
        random.Random(42).shuffle(ordered)
        assert [r["image_id"] for r in ordered[:100]] == [r["image_id"] for r in sample]
        run = ROOT / "runs/E3_fasterrcnn_unweighted_20ep_seed42_v1"
        for alias, epoch in [("best.pth", 13), ("last.pth", 20)]:
            path = run / alias
            assert digest(path) == audit["checkpoints"][epoch - 1]["sha256"]
            state = torch.load(path, map_location="cpu", weights_only=True)
            assert state["completed_epoch"] == epoch and finite_tree(state)
        for row in selected:
            with Image.open(ROOT / row["render"]) as im:
                im.verify()
        bundle = ROOT / "runs/E3_fasterrcnn_best_calibration500_v1"
        assert (bundle / "COMPLETE.json").is_file()
    return {
        "passed": True,
        "local_checkpoint_and_manifest_checks": local,
        "epochs": 20,
        "calibration_images": 500,
        "benchmark_images": 100,
        "qualitative_images": 13,
        "historical_files_verified": historical["files_checked"],
        "reserved_accessed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Also verify local checkpoint hashes/tensors and allowed manifests",
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.local), indent=2))
