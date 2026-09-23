"""Validate frozen comparison artifacts without opening reserved data."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.comparison_protocol import (
    CALIBRATION_SHA,
    MAPPING_SHA,
    NAMES,
    load_manifest,
    sha,
    validate_bundle,
    verify_seal,
)
from src.evaluation.complementarity import select_threshold

REPORT = ROOT / "reports/comparisons/E1_E3_stageE_v2"


def validate(local=False):
    def read(name):
        return json.loads((REPORT / (name + ".json")).read_text())

    protocol = verify_seal(
        json.loads((ROOT / "configs/accuracy/E1_E3_comparison_v1.json").read_text())
    )
    plan = verify_seal(read("calibration_plan"))
    threshold = verify_seal(read("operating_thresholds"))
    receipt = read("freeze_receipt")
    assert sha(ROOT / receipt["protocol_file"]) == receipt["protocol_file_sha256"]
    for path, expected in receipt["artifact_sha256"].items():
        assert sha(ROOT / path) == expected, path
    for path, expected in plan["source_sha256"].items():
        assert sha(ROOT / path) == expected, path
    assert protocol["common_evaluator_source_sha256"] == sha(
        ROOT / "src/evaluation/common_metrics.py"
    )
    assert protocol["calibration_manifest_sha256"] == CALIBRATION_SHA
    assert (
        protocol["class_mapping_sha256"] == MAPPING_SHA
        and protocol["class_names"] == NAMES
    )
    assert not protocol["reserved_accessed"] and not protocol["fusion_started"]
    assert protocol["operating_thresholds"] == threshold["thresholds"]
    assert protocol["family_gate"]["members"] == ["E1", "E3", "E4"]
    assert "closed" in protocol["status"]
    for model in ["E1", "E3"]:
        curve = list(csv.DictReader((REPORT / f"{model}_threshold_curve.csv").open()))
        curve = [{k: float(v) for k, v in r.items()} for r in curve]
        assert [r["threshold"] for r in curve] == plan["threshold_rule"]["grid"]
        assert select_threshold(curve) == threshold["thresholds"][model]
        metrics = read(model + "_metrics")
        classes = list(csv.DictReader((REPORT / f"{model}_per_class.csv").open()))
        assert metrics["images"] == 500 and metrics["objects"] == 6148
        assert metrics["confidence"] == threshold["thresholds"][model]
        for key, source in [
            ("precision", "precision"),
            ("recall", "recall"),
            ("macro_class_f1", "f1"),
            ("map50", "ap50"),
            ("map50_95", "ap50_95"),
        ]:
            assert math.isclose(
                metrics[key],
                np.mean([float(c[source]) for c in classes]),
                abs_tol=1e-12,
            )
        assert sum(int(c["gt_count"]) for c in classes) == 6148
        matrix = np.array(read(model + "_confusion")["matrix"])
        assert (
            matrix.shape == (15, 15)
            and matrix[:14, :].sum() == 6148
            and matrix[:, :14].sum() == metrics["predictions_fixed"]
        )
        bundle_receipt = read(model + "_prediction_receipt")
        assert (
            bundle_receipt["manifest_sha256"] == CALIBRATION_SHA
            and bundle_receipt["mapping_sha256"] == MAPPING_SHA
        )
        assert (
            bundle_receipt["checkpoint_sha256"] == protocol["models"][model]["sha256"]
        )
        if local:
            rows = load_manifest()
            path = ROOT / "runs" / bundle_receipt["id"] / "predictions.jsonl"
            assert validate_bundle(path, bundle_receipt, [r["image_id"] for r in rows])
            assert (
                sha(ROOT / protocol["models"][model]["checkpoint"])
                == bundle_receipt["checkpoint_sha256"]
            )
    comp = read("complementarity")
    assert sum(comp["counts"].values()) == 6148
    for c in comp["per_class"]:
        assert (
            sum(c[k] for k in ["both", "E1_only", "E3_only", "neither"])
            == c["gt_support"]
        )
    return {
        "passed": True,
        "local_prediction_and_checkpoint_hashes_verified": local,
        "images_per_model": 500,
        "reserved_accessed": False,
        "protocol_canonical_sha256": receipt["protocol_canonical_sha256"],
        "E4_gate": "closed pending calibration-only E4 freeze and separate authorization",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.local), indent=2))
