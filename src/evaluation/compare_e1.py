"""Compare frozen E0/E1 measurements; refuse incompatible validation protocols."""

import argparse
import json
import numpy as np
import pandas as pd
from src.data.common import ROOT, save_json, sha256

E0 = "yolov8n_uvh26_mv_baseline_seed42_v1"
E1 = "E1_yolov8s_uvh26_mv_640_seed42"
METRICS = ["precision", "recall", "f1", "macro_f1", "map50", "map50_95"]


def validate_protocol(e0, e1):
    for key in [
        "validation_manifest_sha256",
        "split",
        "imgsz",
        "batch",
        "device",
        "confidence_floor",
        "nms_iou",
        "max_det",
    ]:
        if key not in e0 or key not in e1 or e0[key] != e1[key]:
            raise ValueError(f"Incompatible evaluation field: {key}")
    for obj in [e0, e1]:
        for key in METRICS:
            value = obj.get(key)
            if (
                not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"Invalid metric: {key}")


def overall_comparison(e0, e1):
    validate_protocol(e0, e1)
    return [
        {
            "metric": key,
            "E0": e0[key],
            "E1": e1[key],
            "delta_percentage_points": 100 * (e1[key] - e0[key]),
        }
        for key in METRICS
    ]


def per_class_comparison(e0, e1):
    keys = ["class_id", "name"]
    for frame in [e0, e1]:
        if frame.class_id.duplicated().any() or set(frame.class_id) != set(range(14)):
            raise ValueError("Expected unique IDs for all 14 classes")
    if (
        e0.set_index("class_id").name.to_dict()
        != e1.set_index("class_id").name.to_dict()
    ):
        raise ValueError("Class mapping changed")
    joined = e0.merge(
        e1, on=keys, validate="one_to_one", suffixes=("_E0", "_E1")
    ).sort_values("class_id")
    for key in ["precision", "recall", "f1", "ap50", "ap50_95"]:
        for tag in ["E0", "E1"]:
            values = joined[f"{key}_{tag}"].to_numpy(dtype=float)
            if (
                not np.isfinite(values).all()
                or (values < 0).any()
                or (values > 1).any()
            ):
                raise ValueError("Invalid per-class value")
        joined[f"{key}_delta_pp"] = 100 * (joined[f"{key}_E1"] - joined[f"{key}_E0"])
    return joined


def check_timing_protocol(a, b):
    for key in [
        "validation_manifest_sha256",
        "device",
        "batch",
        "imgsz",
        "warmup_runs",
        "measured_images",
        "seed",
        "confidence",
        "nms_iou",
        "max_det",
        "model_precision",
        "stage_timing",
    ]:
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f"Timing protocol mismatch: {key}")
    if [r["image_id"] for r in a["records"]] != [r["image_id"] for r in b["records"]]:
        raise ValueError("Timing images/order differ")
    if [r["input_shape"] for r in a["records"]] != [
        r["input_shape"] for r in b["records"]
    ]:
        raise ValueError("Timing tensor shapes differ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e0-timing", required=True)
    parser.add_argument("--e1-timing", required=True)
    args = parser.parse_args()

    def load(path):
        return json.loads((ROOT / path).read_text())

    m0 = load(f"reports/tables/{E0}_validation_metrics.json")
    m1 = load(f"reports/tables/{E1}_validation_metrics.json")
    t0 = load(args.e0_timing)
    t1 = load(args.e1_timing)
    check_timing_protocol(t0, t1)
    for metrics, timing in [(m0, t0), (m1, t1)]:
        if metrics["weights_sha256"] != timing["weights_sha256"]:
            raise ValueError("Benchmark checkpoint differs from evaluated checkpoint")
    overall = overall_comparison(m0, m1)
    perclass = per_class_comparison(
        pd.read_csv(ROOT / f"reports/tables/{E0}_validation_per_class.csv"),
        pd.read_csv(ROOT / f"reports/tables/{E1}_validation_per_class.csv"),
    )
    target = ROOT / "reports/comparisons/E1_vs_E0"
    if target.exists():
        raise ValueError("Comparison output exists; preserve it")
    target.mkdir(parents=True)
    pd.DataFrame(overall).to_csv(target / "overall.csv", index=False)
    perclass.to_csv(target / "per_class.csv", index=False)
    c0 = load(f"reports/tables/{E0}_validation_confusion_matrix.json")
    c1 = load(f"reports/tables/{E1}_validation_confusion_matrix.json")
    for key in ["names", "confidence", "matching_iou"]:
        if c0[key] != c1[key]:
            raise ValueError("Confusion matrix protocols differ")
    d = np.asarray(c1["matrix"]) - np.asarray(c0["matrix"])
    if d.shape != (15, 15):
        raise ValueError("Expected 14 classes plus background")
    save_json(
        target / "confusion_delta.json",
        dict(
            names=c0["names"],
            matrix_E1_minus_E0=d.tolist(),
            axes=c0["axes"],
            confidence=c0["confidence"],
            matching_iou=c0["matching_iou"],
        ),
    )
    timing = []
    for stage in ["preprocess_ms", "inference_ms", "postprocess_ms", "end_to_end_ms"]:
        for metric in ["mean_ms", "median_ms", "p95_ms", "fps_from_total_time"]:
            a = t0["summary"][stage][metric]
            b = t1["summary"][stage][metric]
            timing.append(
                dict(
                    stage=stage,
                    metric=metric,
                    E0=a,
                    E1=b,
                    delta=b - a,
                    ratio_E1_to_E0=b / a,
                )
            )
    pd.DataFrame(timing).to_csv(target / "latency.csv", index=False)
    runs = [load(f"reports/tables/{name}_provenance.json") for name in [E0, E1]]
    costs = [
        dict(
            experiment=label,
            epochs=r["epochs_completed"],
            best_epoch=r["best_epoch"],
            early_stopped=r["early_stopped"],
            duration_seconds=r["duration_seconds"],
            checkpoint_bytes=m["weights_bytes"],
            parameters=m["parameters"],
            weights_sha256=m["weights_sha256"],
        )
        for label, r, m in zip(["E0", "E1"], runs, [m0, m1])
    ]
    save_json(target / "training_cost.json", costs)
    save_json(
        target / "provenance.json",
        dict(
            status="measured",
            scope="Same frozen 2000-image MV validation subset; no independent test or live-video claim",
            E0_checkpoint_sha256=m0["weights_sha256"],
            E1_checkpoint_sha256=m1["weights_sha256"],
            validation_manifest_sha256=m0["validation_manifest_sha256"],
            e0_timing_file=args.e0_timing,
            e1_timing_file=args.e1_timing,
            source_sha256=sha256(__file__),
            f1_note="Harmonic aggregate F1 is not micro-F1. Each model selects its own max mean-F1 confidence; AP comparison uses the common validation protocol.",
            rare_class_note="Mini-bus has 58 objects; Others has 31. Single seed and no confidence intervals; gains are descriptive, not significance claims.",
        ),
    )
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].barh(
        perclass.name,
        perclass.ap50_95_delta_pp,
        color=np.where(perclass.ap50_95_delta_pp >= 0, "#007d80", "#c94d4d"),
    )
    axes[0].axvline(0, color="black", linewidth=0.7)
    axes[0].set_xlabel("AP50:95 change (percentage points)")
    axes[0].set_title("E1 YOLOv8s minus E0 YOLOv8n")
    for label, m, t in [("E0 YOLOv8n", m0, t0), ("E1 YOLOv8s", m1, t1)]:
        x = t["summary"]["end_to_end_ms"]["mean_ms"]
        y = m["map50_95"] * 100
        axes[1].scatter(x, y, s=100)
        axes[1].annotate(label, (x, y), xytext=(8, 5), textcoords="offset points")
    axes[1].set_xlabel("Mean synchronized file-to-result latency (ms)")
    axes[1].set_ylabel("Subset validation mAP50:95 (%)")
    axes[1].margins(0.3)
    fig.tight_layout()
    fig.savefig(target / "accuracy_speed.png", dpi=180)
    plt.close(fig)
    print(pd.DataFrame(overall).to_string(index=False))


if __name__ == "__main__":
    main()
