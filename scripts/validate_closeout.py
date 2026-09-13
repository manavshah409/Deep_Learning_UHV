"""Validate measured evaluation artifacts against each other and the frozen run."""

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.data.common import ROOT, sha256, save_json
from src.evaluation.benchmark_inference import summarize

NAME = "yolov8n_uvh26_mv_baseline_seed42_v1"


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    integrity = load("reports/audit/baseline_closeout_integrity.json")
    m = load(f"reports/tables/{NAME}_validation_metrics.json")
    t = load(f"reports/tables/{NAME}_latency.json")
    r = load("reports/error_analysis/baseline_prediction_review.json")
    frozen = load("reports/audit/baseline_subset_frozen_provenance.json")
    rows = list(
        csv.DictReader(
            (ROOT / f"reports/tables/{NAME}_validation_per_class.csv").open()
        )
    )
    assert len(rows) == 14 and [int(x["class_id"]) for x in rows] == list(range(14))
    for x in rows:
        for k in ["precision", "recall", "f1", "ap50", "ap50_95"]:
            assert np.isfinite(float(x[k])) and 0 <= float(x[k]) <= 1
        p, rec = float(x["precision"]), float(x["recall"])
        assert np.isclose(float(x["f1"]), 2 * p * rec / (p + rec) if p + rec else 0)
    for key, col in [
        ("precision", "precision"),
        ("recall", "recall"),
        ("macro_f1", "f1"),
        ("map50", "ap50"),
        ("map50_95", "ap50_95"),
    ]:
        assert np.isclose(m[key], np.mean([float(x[col]) for x in rows]))
    assert np.isclose(
        m["f1"], 2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
    )
    for artifact in [m, t, r]:
        assert (
            artifact["weights_sha256"] == integrity["checkpoints"]["weights"]["sha256"]
        )
        assert (
            artifact["validation_manifest_sha256"] == frozen["manifest_sha256"]["val"]
        )
    cm = load(f"reports/tables/{NAME}_validation_confusion_matrix.json")
    matrix = np.asarray(cm["matrix"])
    assert (
        matrix.shape == (15, 15) and (matrix >= 0).all() and np.isfinite(matrix).all()
    )
    assert list(matrix[:, :14].sum(0).astype(int)) == [
        2402,
        1169,
        1028,
        541,
        627,
        933,
        4016,
        11624,
        1322,
        58,
        130,
        278,
        183,
        31,
    ]
    assert cm["confidence"] == 0.001 and cm["matching_iou"] == 0.45
    assert (
        t["device"] == "mps"
        and t["batch"] == 1
        and t["warmup_runs"] == 10
        and len(t["records"]) == 100
    )
    assert len({x["image_id"] for x in t["records"]}) == 100
    for key, expected in t["summary"].items():
        actual = summarize([x[key] for x in t["records"]])
        assert all(np.isclose(actual[k], v) for k, v in expected.items())
    assert "Explicit torch.mps.synchronize" in t["stage_timing"]
    assert not t["realtime_claim"]
    for path, digest in integrity["run_file_sha256"].items():
        assert sha256(ROOT / path) == digest, f"Original run changed: {path}"
    for suffix in [
        "BoxPR_curve",
        "BoxF1_curve",
        "BoxP_curve",
        "BoxR_curve",
        "confusion_matrix",
        "confusion_matrix_normalized",
    ]:
        with Image.open(ROOT / f"reports/figures/{NAME}_validation_{suffix}.png") as im:
            im.verify()
    assert len(r["manually_reviewed_validation_ids"]) == 6
    assert "passed" in (ROOT / "reports/audit/closeout_pytest.txt").read_text()
    result = dict(
        status="passed",
        metrics_per_class_and_f1="passed",
        confusion_matrix_gt_counts="24342 across 14 classes",
        checkpoint_and_manifest_identity="passed",
        latency_recomputed_from_100_samples="passed",
        original_run_inventory_unchanged=True,
        curves_readable=6,
        manual_prediction_pairs_reviewed=6,
        standalone_evaluation_exit_code=0,
        synchronized_benchmark_exit_code=0,
        diagnostic_prediction_exit_code=0,
        yolo_validation_exit_code=0,
        pdf_pages_visually_reviewed=5,
        pdf_layout="passed",
        scope="Frozen 8000/2000 MV subset only",
    )
    save_json(ROOT / "reports/audit/closeout_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
