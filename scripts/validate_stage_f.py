"""Read-only Stage F artifact/preservation validation; reserved paths never opened."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.comparison_protocol import canonical_hash, sha, verify_seal
from src.evaluation.fusion import OPERATING_GRID, candidates, improvement_gate

OUT = ROOT / "reports/accuracy_stage_f"


def validate(local=False):
    def read(name):
        return json.loads((OUT / (name + ".json")).read_text())

    plan = verify_seal(read("protocol"))
    final = verify_seal(read("final_amendment"))
    assert (
        verify_seal(
            yaml.safe_load((ROOT / "configs/accuracy/E4_fusion.yaml").read_text())
        )
        == final
    )
    assert final["stage_f_protocol_hash"] == canonical_hash(plan)
    parent = verify_seal(
        json.loads((ROOT / "configs/accuracy/E1_E3_comparison_v1.json").read_text())
    )
    assert canonical_hash(parent) == plan["parent_protocol_hash"]
    for path, expected in plan["source_sha256"].items():
        assert sha(ROOT / path) == expected, path
    integrity = read("integrity")
    assert integrity["passed"] and not integrity["reserved_accessed"]
    for path, expected in integrity["preservation"].items():
        assert sha(ROOT / path) == expected, path
    folds = read("folds")["folds"]
    flat = [i for f in folds for i in f]
    assert (
        len(folds) == 5 and all(len(f) == 100 for f in folds) and len(set(flat)) == 500
    )
    selection = read("fold_selections")
    assert len(selection) == 20
    for row in selection:
        assert (
            row["configuration"] in candidates() and row["threshold"] in OPERATING_GRID
        )
        assert row["training_images"] == 400 and row["heldout_images"] == 100
        if row["context"] is not None:
            train = set(row["context"]["fit_image_ids"])
            test = set(folds[row["fold"]])
            assert (
                len(train) == 400
                and train.isdisjoint(test)
                and train | test == set(flat)
            )
    trials = list(csv.DictReader((OUT / "training_search.csv").open()))
    assert len(trials) == 5 * 19 * 9
    assert len({(r["fold"], r["candidate"], r["threshold"]) for r in trials}) == len(
        trials
    )
    metrics = {}
    classes = {}
    errors = {}
    for family in ["baseline", "union", "wbf", "recovery"]:
        m = read(family + "_oof_metrics")
        metrics[family] = m
        raw = list(csv.DictReader((OUT / (family + "_oof_per_class.csv")).open()))
        cl = []
        for row in raw:
            cl.append(
                {
                    k: (
                        int(v)
                        if k
                        in [
                            "class_id",
                            "gt_count",
                            "prediction_count_raw",
                            "prediction_count_fixed",
                            "tp",
                            "fp",
                            "fn",
                        ]
                        else float(v)
                        if k in ["precision", "recall", "f1", "ap50", "ap50_95"]
                        else v
                    )
                    for k, v in row.items()
                }
            )
        classes[family] = cl
        errors[family] = read(family + "_oof_errors")
        assert (
            len(cl) == 14
            and sum(c["gt_count"] for c in cl) == 6148
            and m["images"] == 500
        )
        for key, source in [
            ("precision", "precision"),
            ("recall", "recall"),
            ("macro_class_f1", "f1"),
            ("map50", "ap50"),
            ("map50_95", "ap50_95"),
        ]:
            assert math.isclose(m[key], np.mean([r[source] for r in cl]), abs_tol=1e-12)
        assert math.isclose(
            m["harmonic_aggregate_f1"],
            2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"]),
            abs_tol=1e-12,
        )
        assert sum(c["prediction_count_fixed"] for c in cl) == m["predictions_fixed"]
        matrix = np.array(read(family + "_oof_confusion")["matrix"])
        assert (
            matrix.shape == (15, 15)
            and matrix[:14, :].sum() == 6148
            and matrix[:, :14].sum() == m["predictions_fixed"]
        )
        curve = np.load(OUT / (family + "_oof_pr_curves.npz"))
        assert curve["precision"].shape == (10, 101, 14)
    cv = read("cross_validation_results")
    for f in ["union", "wbf", "recovery"]:
        assert (
            improvement_gate(
                metrics["baseline"],
                metrics[f],
                classes["baseline"],
                classes[f],
                errors["baseline"],
                errors[f],
            )
            == cv["gates"][f]
        )
    if cv["selected_fusion"] is None:
        assert (
            not any(g["passed"] for g in cv["gates"].values())
            and final["purpose"] == "rejected_fusion_latency_only_not_selected"
        )
    old = json.loads(
        (ROOT / "reports/comparisons/E1_E3_stageE_v2/E1_metrics.json").read_text()
    )
    assert metrics["baseline"]["map50_95"] == old["map50_95"]
    latency = read("latency")
    assert latency["timed_images"] == 100 and latency["warmups"] == 10
    assert len(set(latency["sample_ids"])) == 100 and set(latency["sample_ids"]) <= set(
        flat
    )
    assert sha(ROOT / latency["sample_file"]) == latency["sample_sha256"]
    provenance = read("benchmark_lint_provenance")
    source = (ROOT / "scripts/benchmark_stage_f.py").read_text()
    assert sha(ROOT / "scripts/benchmark_stage_f.py") == provenance["committed_sha256"]
    assert source.count(provenance["new_line"]) == 1
    import hashlib

    executed = source.replace(provenance["new_line"], provenance["old_line"])
    assert (
        hashlib.sha256(executed.encode()).hexdigest()
        == provenance["executed_sha256"]
        == latency["benchmark_source_sha256"]
    )
    samples = list(csv.DictReader((OUT / "latency_samples.csv").open()))
    for key, stats in latency["stage_ms"].items():
        values = np.array([float(r[key]) for r in samples])
        assert np.isfinite(values).all() and (values >= 0).all()
        for field, value in [
            ("mean", values.mean()),
            ("median", np.median(values)),
            ("p95", np.percentile(values, 95)),
        ]:
            assert math.isclose(stats[field], value, abs_tol=1e-9)
    assert math.isclose(
        latency["still_image_fps"],
        1000 / latency["stage_ms"]["end_to_end_ms"]["mean"],
        abs_tol=1e-12,
    )
    assert latency["prediction_parity_checks"] == 200
    for result in read("latency_prediction_parity"):
        assert result["passed"]
    if local:
        from src.evaluation.comparison_protocol import load_manifest, validate_bundle

        allowed = load_manifest()
        ids = [r["image_id"] for r in allowed]
        assert set(ids) == set(flat)
        for model in ["E1", "E3"]:
            receipt = json.loads(
                (
                    ROOT
                    / f"reports/comparisons/E1_E3_stageE_v2/{model}_prediction_receipt.json"
                ).read_text()
            )
            assert (
                sha(ROOT / receipt["inference_configuration"]["checkpoint"])
                == receipt["checkpoint_sha256"]
            )
            assert validate_bundle(
                ROOT / "runs" / receipt["id"] / "predictions.jsonl", receipt, ids
            )
        repro = read("reproducibility")
        for family in ["baseline", "union", "wbf", "recovery"]:
            rows = [
                json.loads(line)
                for line in (
                    ROOT / f"runs/E4_stageF_v1/{family}_oof_predictions.jsonl"
                ).open()
            ]
            for fold in range(5):
                chosen = [r for r in rows if r["image_id"] in folds[fold]]
                saved = next(
                    r for r in repro if r["family"] == family and r["fold"] == fold
                )
                assert (
                    canonical_hash(chosen) == saved["prediction_hash"]
                    and saved["equal"]
                )
        for path in [
            "runs/E4_stageF_driver_exit.json",
            "runs/E4_stageF_latency_exit.json",
        ]:
            assert json.loads((ROOT / path).read_text())["exit_code"] == 0
    return {
        "passed": True,
        "local_hashes_verified": local,
        "historical_files_unchanged": len(integrity["preservation"]),
        "folds": 5,
        "training_search_rows": 855,
        "decision": cv["decision"],
        "reserved_accessed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.local), indent=2))
    except BaseException as error:
        (OUT / "VALIDATION_FAILURE.json").write_text(
            json.dumps(
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "reserved_accessed": False,
                },
                indent=2,
            )
            + "\n"
        )
        raise
