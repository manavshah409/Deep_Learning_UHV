"""Gate A protocol checks and resolution-specific measured comparison."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.common import ROOT, sha256


def check_protocol(a, b):
    if [a["imgsz"], b["imgsz"]] != [640, 960]:
        raise ValueError("Expected 640 then 960")
    for key in [
        "weights_sha256",
        "validation_manifest_sha256",
        "dataset_yaml_sha256",
        "class_mapping_sha256",
        "device",
        "batch",
        "workers",
        "confidence_floor",
        "nms_iou",
        "max_det",
        "source_sha256",
        "size_ap_source_sha256",
        "shared_metric_source_sha256",
        "python",
        "torch",
        "ultralytics",
        "pycocotools",
    ]:
        if a[key] != b[key]:
            raise ValueError(f"Protocol differs: {key}")


def main():
    load = lambda p: json.loads((ROOT / p).read_text())
    metrics = []
    sizes = []
    per = []
    counts = []
    conf = []
    for size in [640, 960]:
        folder = Path(f"reports/evaluations/E2_gateA_{size}_seed42_v2")
        marker = load(folder / "COMPLETE.json")
        assert marker["status"] == "complete"
        assert all(
            sha256(ROOT / folder / p) == h for p, h in marker["files_sha256"].items()
        )
        metrics.append(load(folder / "metrics.json"))
        sizes.append(load(folder / "size_ap.json"))
        counts.append(load(folder / "prediction_counts.json"))
        conf.append(load(folder / "confusion_matrix.json"))
        per.append(pd.read_csv(ROOT / folder / "per_class.csv"))
    a, b = metrics
    check_protocol(a, b)
    assert [r["image"] for r in counts[0]] == [r["image"] for r in counts[1]]
    assert a["evaluated_images"] == b["evaluated_images"] == 2000
    assert a["ground_truth_objects"] == b["ground_truth_objects"] == 24342
    for x in per:
        assert (
            set(x.class_id) == set(range(14))
            and np.isfinite(x.select_dtypes("number")).all().all()
        )
    timings = [
        load(f"reports/tables/E2_gateA_{s}_benchmark_v1_latency.json")
        for s in [640, 960]
    ]
    for k in [
        "weights_sha256",
        "validation_manifest_sha256",
        "device",
        "batch",
        "warmup_runs",
        "measured_images",
        "seed",
        "confidence",
        "nms_iou",
        "max_det",
        "model_precision",
        "stage_timing",
        "environment",
    ]:
        assert timings[0][k] == timings[1][k], k
    assert [r["image_id"] for r in timings[0]["records"]] == [
        r["image_id"] for r in timings[1]["records"]
    ]
    target = ROOT / "reports/comparisons/E2_gateA_v2"
    assert not target.exists()
    target.mkdir(parents=True)
    rows = [
        dict(metric=k, res640=a[k], res960=b[k], delta_pp=100 * (b[k] - a[k]))
        for k in ["precision", "recall", "f1", "macro_f1", "map50", "map50_95"]
    ]
    pd.DataFrame(rows).to_csv(target / "overall.csv", index=False)
    joined = per[0].merge(
        per[1],
        on=["class_id", "name"],
        suffixes=("_640", "_960"),
        validate="one_to_one",
    )
    for key in ["precision", "recall", "f1", "ap50", "ap50_95"]:
        joined[key + "_delta_pp"] = 100 * (joined[key + "_960"] - joined[key + "_640"])
    joined.to_csv(target / "per_class.csv", index=False)
    size_rows = []
    for group in ["all", "small", "medium", "large"]:
        x, y = [s["groups"][group] for s in sizes]
        assert x["objects"] == y["objects"]
        size_rows.append(
            dict(
                size=group,
                objects=x["objects"],
                ap50_95_640=x["ap50_95"],
                ap50_95_960=y["ap50_95"],
                delta_pp=100 * (y["ap50_95"] - x["ap50_95"]),
            )
        )
    pd.DataFrame(size_rows).to_csv(target / "size_ap.csv", index=False)
    latency = []
    for stage in ["preprocess_ms", "inference_ms", "postprocess_ms", "end_to_end_ms"]:
        for key in ["mean_ms", "median_ms", "p95_ms", "fps_from_total_time"]:
            x, y = [t["summary"][stage][key] for t in timings]
            latency.append(
                dict(stage=stage, metric=key, res640=x, res960=y, ratio=y / x)
            )
    pd.DataFrame(latency).to_csv(target / "latency.csv", index=False)
    for k in ["names", "axes", "confidence", "matching_iou"]:
        assert conf[0][k] == conf[1][k]
    d = np.array(conf[1]["matrix"]) - np.array(conf[0]["matrix"])
    (target / "confusion_delta.json").write_text(
        json.dumps(
            {
                "names": conf[0]["names"],
                "confidence": conf[0]["confidence"],
                "matching_iou": conf[0]["matching_iou"],
                "matrix_960_minus_640": d.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    small = size_rows[1]["delta_pp"]
    overall = 100 * (b["map50_95"] - a["map50_95"])
    gate = {
        "status": "awaiting_qualitative_and_cost_review",
        "small_ap50_95_delta_pp": small,
        "overall_map50_95_delta_pp": overall,
        "small_threshold_met": small >= 2,
        "overall_threshold_met": overall >= 1,
        "protocol_integrity": "passed",
        "training_started": False,
        "evaluation_memory": [m["memory"] for m in metrics],
        "benchmark_memory": [t["memory"] for t in timings],
        "evaluation_duration_seconds": [m["duration_seconds"] for m in metrics],
        "prediction_counts": [m["prediction_count"] for m in metrics],
        "expected_gate_rule": "small +2pp OR overall +1pp with clearly improved relevant small/distant classes, plus integrity/cost justification",
    }
    (target / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
