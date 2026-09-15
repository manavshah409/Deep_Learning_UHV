"""Write the measured E1 report after comparison and visual review are complete."""

from pathlib import Path
import json
import csv

ROOT = Path(__file__).resolve().parents[1]
E0 = "yolov8n_uvh26_mv_baseline_seed42_v1"
E1 = "E1_yolov8s_uvh26_mv_640_seed42"


def load(path):
    return json.loads((ROOT / path).read_text())


def readcsv(path):
    return list(csv.DictReader((ROOT / path).open()))


def table(headers, rows):
    return (
        "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join(["---"] * len(headers))
        + " |\n"
        + "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in rows)
    )


def main():
    overall = readcsv("reports/comparisons/E1_vs_E0/overall.csv")
    per = readcsv("reports/comparisons/E1_vs_E0/per_class.csv")
    timing = readcsv("reports/comparisons/E1_vs_E0/latency.csv")
    costs = load("reports/comparisons/E1_vs_E0/training_cost.json")
    review = load("reports/comparisons/E1_paired_summary.json")
    assert review["status"] == "manually_reviewed", (
        "Manual paired review must precede final report"
    )
    integrity = load("reports/audit/E1_completed_run_verification.json")
    assert integrity["status"] == "passed"
    run = load(f"reports/tables/{E1}_provenance.json")
    pre = load("reports/audit/E1_preflight_verification.json")
    prov = load("reports/comparisons/E1_vs_E0/provenance.json")
    metric_table = table(
        ["Metric", "E0 YOLOv8n", "E1 YOLOv8s", "E1 - E0 (pp)"],
        [
            [
                r["metric"],
                f"{float(r['E0']):.6f}",
                f"{float(r['E1']):.6f}",
                f"{float(r['delta_percentage_points']):+.3f}",
            ]
            for r in overall
        ],
    )
    class_table = table(
        [
            "Class",
            "P E0 / E1",
            "R E0 / E1",
            "AP50 E0 / E1",
            "AP50:95 E0 / E1",
            "AP50:95 delta pp",
        ],
        [
            [r["name"]]
            + [
                f"{float(r[k + '_E0']):.4f} / {float(r[k + '_E1']):.4f}"
                for k in ["precision", "recall", "ap50", "ap50_95"]
            ]
            + [f"{float(r['ap50_95_delta_pp']):+.3f}"]
            for r in per
        ],
    )
    speed_table = table(
        ["Stage", "Statistic", "E0", "E1", "E1 / E0"],
        [
            [
                r["stage"],
                r["metric"],
                f"{float(r['E0']):.3f}",
                f"{float(r['E1']):.3f}",
                f"{float(r['ratio_E1_to_E0']):.3f}",
            ]
            for r in timing
        ],
    )
    cost_table = table(
        [
            "Model",
            "Epochs / best",
            "Early stop",
            "Training seconds",
            "Parameters (evaluation model)",
            "Checkpoint bytes",
        ],
        [
            [
                r["experiment"],
                f"{r['epochs']} / {r['best_epoch']}",
                r["early_stopped"],
                f"{r['duration_seconds']:.3f}",
                r["parameters"],
                r["checkpoint_bytes"],
            ]
            for r in costs
        ],
    )
    text = f"""# Phase 2 E1: YOLOv8s versus frozen YOLOv8n

E1 is trained and independently evaluated on the frozen subset. This report compares measured accuracy, training cost and common-protocol synchronized MPS timing. No other Phase 2 experiment was started. Git delivery status is recorded in `reports/audit/E1_delivery.json`.

## Question and experimental boundary

Does increasing model capacity from YOLOv8n to YOLOv8s improve UVH-26 detection, especially difficult classes, enough to justify its additional cost and latency? Model size and corresponding official COCO initializer are the intended independent variable. This is one seed, one fixed subset and validation-based checkpoint selection; no independent test-set accuracy, significance claim, live-video FPS or production-readiness claim.

Run IDs: E0 `{E0}`; E1 `{E1}`; recovery preflight `{E1}_preflight_v2`. Dataset is `uvh26_mv_yolo_v1/subsets/baseline_seed42_v2`: 8,000 training / 2,000 validation images, 14 unchanged MV classes; 94,609 / 24,342 objects. Validation manifest SHA-256 `{prov["validation_manifest_sha256"]}`. Pretraining verification rechecked all 10,000 image/label hashes, paths and E0 checkpoint/run hashes. The gate passed before preflight. A pre-existing faculty PDF serialization change was reviewed and preserved; its normalized text matched the committed report.

Full 26,646-image annotation-catalogue audit and 10,000-image local pixel audit remain distinct. Acquisition and integrity verification of unselected images remain incomplete and are outside this comparison.

## Controls and executed training

Frozen E1 config: `configs/E1_yolov8s_uvh26_mv_640_seed42.yaml`. Image size640, seed42, 30-epoch budget, patience10, AdamW lr0 .000556, lrf .01, weight decay .0005, warm-up3, nbs64 and E0 augmentation/schedule settings. Explicit AMP=false and workers=0 match E0 effective behavior (E0 requested true/4). Actual E1 batch **{run["effective_batch"]}**, effective AMP **{run["effective_amp"]}**, workers **{run["effective_workers"]}**, startup accumulation **{run["startup_optimizer"]["accumulate"]}**. Any effective-control differences are recorded in the completed-run verification; no silent equivalence claim.

COCO-pretrained YOLOv8s came from official Ultralytics assets v8.4.0: initializer SHA-256 `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`. E0 initialization and finished checkpoint remain unchanged. E1 uses a separate runner derived from E0, with correct model-source naming and an epoch-level finite-loss guard. E0 source was not edited.

Preflight v1 exited 1 after its training batches because the added loss guard did not accept named-loss dictionaries; it saved no checkpoint. The corrected guard has regression tests. Recovery v2 used one epoch on 800 training images and all 2,000 validation images and took {pre["duration_seconds"]:.3f}s, saved readable finite checkpoints and passed visual class/placement checks. It was a pipeline test, not an E1 accuracy result. Its full-budget duration estimate was {pre["full_30_epoch_duration_estimate_seconds"] / 3600:.2f}h; the actual duration is below. Original preflight and E0 files remain separate and preserved.

{cost_table}
E1 process exit: {integrity["process_exit_code"]}. Start {run["started_at"]}; end {run["completed_at"]}. Best epoch comes from the save callback, cross-checked against CSV validation fitness; optimizer-stripped checkpoint epoch metadata may be -1. Complete effective args, finite losses, per-epoch CSV, warnings and checkpoint hashes are preserved. MPS nondeterminism under warn-only settings limits bitwise numerical replay.

## Fresh standalone validation

{metric_table}
The E1 best checkpoint was freshly validated using E0's evaluator settings: same manifest, MPS, imgsz640, batch8, confidence floor .001, NMS IoU .7, max_det300. Precision/Recall are macro class means at each model's max smoothed mean-F1 confidence. Harmonic aggregate F1 is 2PR/(P+R); macro per-class F1 averages class F1. Neither is micro-F1. AP50:95 averages IoU .50:.05:.95. Differences are absolute percentage points, not relative-percent gains.

{class_table}
Rare-class estimates are uncertain: Mini-bus has 58 validation objects and Others 31; Van has 183. Others precision can equal one when recall is zero due to the evaluator's empty-prediction interpolation convention. Per-class CSV includes separate P/R/F1/AP deltas. No claim that every class benefits is implied by aggregate mAP.

Numeric confusion changes (`confusion_delta.json`) are E1 minus E0, rows predicted / columns true, final row/column background. Both use confidence .001 and matching IoU .45, distinct from the F1 operating point. Negative off-diagonal/background values indicate fewer errors for those cells, not a standalone AP change.

## Comparable MPS speed

Both models were freshly benchmarked after E1 training, sequentially without another model job on MPS. The original Phase 1 E0 measurement was preserved. Common protocol: Apple M5, MPS float32, batch1, imgsz640 rectangular letterbox, identical seed42 100-image order, 10 warm-ups, conf .25/NMS .7/max_det300, explicit synchronization before/after each stage. Actual input tensor shapes are checked for equality.

{speed_table}
End-to-end includes local image read/decode, preprocess, forward pass, postprocess, API overhead and synchronization instrumentation. Excludes model load/warm-up, drawing, capture and display. FPS is 1000/mean ms. OS cache and thermal/background load can affect timing; these sequential measurements are not a controlled laboratory repeated-trials estimate or sustained live-video throughput.

## Paired diagnostic review

GT/E0/E1 compared on the same six validation scenes: 4232,10518,1364,21621,4711,954, spanning dense traffic, small/distant vehicles, occlusions and rare classes. Confidence .10, NMS .7 and diagnostic class-agnostic greedy IoU .5 matching; not unbiased population error rates or COCO AP matching.

"""
    text += "\n\n".join(
        f"- **Validation {r['image_id']}:** {r['manual_finding']}"
        for r in review["cases"]
    )
    text += (
        "\n\nSource omissions, ambiguous fine-grained labels, loose occlusion boxes and redactions remain separate from model failures. Frozen annotations were not edited after observing results. Local comparison images are under `reports/predictions/E1_vs_E0_paired_v1/`; dataset imagery is excluded from Git.\n\n## Assessment and single next experiment\n\n"
        + review["recommendation"]
        + "\n\n## Tests, artifacts and delivery\n\n"
        + (ROOT / "reports/audit/E1_pytest.txt").read_text().strip()
        + "\n\nComparison tests verify percentage-point arithmetic, immutable validation identity, per-class alignment, finite metrics and common benchmark order. Generated tables/plots are validated against source artifacts. E1 checkpoints are local:\n\n"
    )
    for key in ["weights", "last_weights"]:
        w = run[key]
        text += f"- `{w['path']}`: {w['bytes']} bytes; SHA-256 `{w['sha256']}`.\n"
    text += "\nEligible source, config, summaries and documentation are committed normally on the existing branch; raw/processed images, labels, weights, full runs, secrets and local paths are excluded. See `reports/audit/E1_delivery.json` for the actual commit/push result. Stop boundary: E1 only; no higher-resolution run, augmentation/loss experiment, tracking or counting was started.\n"
    (ROOT / "docs/phase_reports/PHASE_2_E1_MODEL_COMPARISON.md").write_text(text)
    print("Measured E1 report written")


if __name__ == "__main__":
    main()
