"""Check complete evaluation bundles and measured comparison arithmetic."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from PIL import Image
from src.data.common import ROOT, sha256
from src.evaluation.compare_e1 import check_timing_protocol


def main():
    load = lambda p: json.loads((ROOT / p).read_text())
    metrics = []
    plots = 0
    for model, num in [("n", 0), ("s", 1)]:
        base = Path(
            f"reports/evaluations/yolov8{model}_uvh26_mv_e{num}_validation_seed42_v2"
        )
        marker = load(base / "COMPLETE.json")
        assert marker["status"] == "complete"
        assert all(
            sha256(ROOT / base / p) == h for p, h in marker["files_sha256"].items()
        )
        m = load(base / "metrics.json")
        metrics.append(m)
        per = pd.read_csv(ROOT / base / "per_class.csv")
        assert len(per) == 14 and np.isfinite(per.select_dtypes("number")).all().all()
        assert np.isclose(per.precision.mean(), m["precision"]) and np.isclose(
            per.recall.mean(), m["recall"]
        )
        assert np.isclose(per.f1.mean(), m["macro_f1"]) and np.isclose(
            per.ap50_95.mean(), m["map50_95"]
        )
        assert np.isclose(
            m["f1"], 2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
        )
        counts = load(base / "prediction_counts.json")
        assert len(counts) == 2000
        assert sum(r["detections"] for r in counts) == m["prediction_count"]
        assert all(sum(r["per_class"]) == r["detections"] for r in counts)
        cm = np.array(load(base / "confusion_matrix.json")["matrix"])
        assert cm.shape == (15, 15) and cm[:, :14].sum() == 24342
        assert cm[:14, :].sum() == m["prediction_count"]
        for p in (ROOT / base).glob("*.png"):
            with Image.open(p) as im:
                im.verify()
            plots += 1
    assert (
        metrics[0]["source_sha256"]
        == metrics[1]["source_sha256"]
        == sha256(ROOT / "src/evaluation/evaluate_baseline.py")
    )
    assert metrics[0]["dataset_yaml_sha256"] == metrics[1]["dataset_yaml_sha256"]
    assert metrics[0]["class_mapping_sha256"] == metrics[1]["class_mapping_sha256"]
    t0 = load("reports/tables/E0_common_protocol_v2_latency.json")
    t1 = load("reports/tables/E1_common_protocol_v2_latency.json")
    check_timing_protocol(t0, t1)
    for m, t in zip(metrics, [t0, t1]):
        assert m["weights_sha256"] == t["weights_sha256"] and len(t["records"]) == 100
        for stage, summary in t["summary"].items():
            a = np.array([r[stage] for r in t["records"]])
            assert (a > 0).all() and np.isfinite(a).all()
            assert np.isclose(a.mean(), summary["mean_ms"]) and np.isclose(
                np.median(a), summary["median_ms"]
            )
            assert np.isclose(np.percentile(a, 95), summary["p95_ms"]) and np.isclose(
                1000 / a.mean(), summary["fps_from_total_time"]
            )
    overall = pd.read_csv(ROOT / "reports/comparisons/E1_vs_E0/overall.csv")
    assert np.allclose(overall.delta_percentage_points, 100 * (overall.E1 - overall.E0))
    comp = pd.read_csv(ROOT / "reports/comparisons/E1_vs_E0/comparison.csv")
    assert np.allclose(comp.absolute_change, comp.E1 - comp.E0)
    assert np.allclose(comp.relative_change_percent, 100 * (comp.E1 / comp.E0 - 1))
    for audit in [
        "reports/audit/baseline_closeout_integrity.json",
        "reports/audit/E1_completed_run_verification.json",
    ]:
        assert all(
            sha256(ROOT / p) == h for p, h in load(audit)["run_file_sha256"].items()
        )
    assert all(
        sha256(ROOT / p) == h
        for p, h in load("reports/audit/E1_evaluation_recovery_preservation.json")[
            "original_file_sha256"
        ].items()
    )
    result = dict(
        status="passed",
        complete_evaluation_bundles=2,
        images_per_evaluation=2000,
        classes=14,
        plots_verified=plots,
        matched_benchmark_images=100,
        checkpoint_and_run_preservation="passed",
        comparison_arithmetic="passed",
    )
    (ROOT / "reports/audit/E1_artifact_validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(result)


if __name__ == "__main__":
    main()
