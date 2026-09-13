"""Build faculty text from measured Phase 1 artifacts; no training or downloads."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "yolov8n_uvh26_mv_baseline_seed42_v1"


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    p = load(f"reports/tables/{NAME}_provenance.json")
    m = load(f"reports/tables/{NAME}_validation_metrics.json")
    t = load(f"reports/tables/{NAME}_latency.json")
    f = load("reports/audit/baseline_subset_frozen_provenance.json")
    review = load("reports/error_analysis/baseline_prediction_review.json")
    rows = list(
        csv.DictReader(
            (ROOT / f"reports/tables/{NAME}_validation_per_class.csv").open()
        )
    )
    table = "| Checkpoint / evaluation | Precision | Recall | F1 (harmonic aggregate) | Macro F1 | mAP@0.5 | mAP@0.5:0.95 |\n|---|---:|---:|---:|---:|---:|---:|\n"
    table += f"| YOLOv8n best epoch 30 / frozen 2,000 val | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['macro_f1']:.6f} | {m['map50']:.6f} | {m['map50_95']:.6f} |\n"
    timing = "| Stage | Mean ms | Median ms | p95 ms |\n|---|---:|---:|---:|\n"
    for k, v in t["summary"].items():
        timing += (
            f"| {k} | {v['mean_ms']:.3f} | {v['median_ms']:.3f} | {v['p95_ms']:.3f} |\n"
        )
    classes = "| Class | Val objects | Precision | Recall | F1 | AP@0.5 | AP@0.5:0.95 |\n|---|---:|---:|---:|---:|---:|---:|\n"
    counts = [2402, 1169, 1028, 541, 627, 933, 4016, 11624, 1322, 58, 130, 278, 183, 31]
    for r, n in zip(rows, counts):
        classes += (
            f"| {r['name']} | {n} | "
            + " | ".join(
                f"{float(r[k]):.4f}"
                for k in ["precision", "recall", "f1", "ap50", "ap50_95"]
            )
            + " |\n"
        )
    text = (
        f"""# Phase 1: UVH-26 MV subset baseline closeout

The proper YOLOv8n baseline completed **30/30 epochs**, process exit **0**, best checkpoint **epoch 30**, with **no early stopping**. Fresh standalone evaluation and synchronized Apple MPS timing are complete. Phase 2 training has not started. The final delivery/gate status is recorded in `reports/audit/phase2_subset_gate.json`; a Git checkpoint is required before controlled Phase 2 work.

This is a **subset validation baseline**, not a full-dataset result, test-set result, production readiness claim or completed traffic-analytics system.

## Data scope and integrity

| Evidence | Images | Object boxes | What was checked |
|---|---:|---:|---|
| Full MV annotation catalogue | 26,646 | 316,220 | JSON schema, classes, box geometry and split identifiers |
| Frozen local training subset | 8,000 | 94,609 | Decode, actual dimensions, paths, content hashes, labels and leakage |
| Frozen local validation subset | 2,000 | 24,342 | Same selected-image integrity checks |

The full catalogue contains 21,349 training and 5,297 validation image records. All selected 10,000 images passed pixel/label integrity, with zero train/validation content overlap and all 14 classes retained. Acquisition and integrity verification of **unselected images remains incomplete**. The catalogue audit must not be described as a full pixel audit. Semantic manual annotation review covered 42 selected images; closeout prediction review covered six validation scenes, not every image.

Source: IISc AIM UVH-26, Majority Voting, revision `{f["revision"]}`, CC BY 4.0. Original class IDs 1-14 map in order to YOLO IDs 0-13. STAPLE annotations are not mixed into this experiment. Maximum selected-vs-full class-share deviation is 0.306231 percentage points.

Frozen version: `{f["version"]}`.

| Record | SHA-256 |
|---|---|
| Combined manifest | `{f["combined_manifest_sha256"]}` |
| Training manifest | `{f["manifest_sha256"]["train"]}` |
| Validation manifest | `{f["manifest_sha256"]["val"]}` |
| Class mapping | `{f["class_mapping_sha256"]}` |
| Selected-image audit | `{f["image_audit_sha256"]}` |

Closeout independently rehashed all 10,000 images and labels, both checkpoints, mapping, manifests, audit, frozen optimizer config, training source and requirements. Saved training dataset paths resolve to the final v2 subset. Evidence: `reports/audit/baseline_closeout_integrity.json` and `baseline_subset_frozen_provenance.json`.

## Source-image recovery history

Train ID 21818, `UVH-26-Train/data/003/803489.png`, decodes at 1620x1080 but declares 1920x1080. Raw-coordinate and width-rescaled overlays gave inconsistent object alignment; neither justified automatic repair. The raw file remains unchanged and quarantined. Its first candidate replacement, ID 5235 (`Train/data/001/341297.png`), was structurally valid but visually degraded and also excluded. Final replacement: train ID 25440 (`Train/data/000/233625.png`), preserving one three-wheeler and two two-wheelers.

Validation ID 20260 (`Val/data/001/986228.png`) was severely visually degraded; final replacement ID 6452 (`Val/data/000/81395.png`) preserves two two-wheelers and one bicycle. Three unique files were quarantined; two original candidate images changed. No clipping, silent resizing or relabelling was performed. The earlier frozen v1 is preserved as superseded before baseline training. `baseline_subset_v2_provenance_addendum.json` clarifies the v2 replacement search: all same-split official candidates, deterministic exposure-distance ranking/tie break and pinned acquisition if needed; the inherited v1 provenance wording is retained unchanged.

## Actual training outcome

Run ID: `{NAME}`. The original run directory and logs are preserved unchanged; no retraining or resume was used for closeout. Start {p["started_at"]}; completion {p["completed_at"]}. Wall duration **{p["duration_seconds"]:.3f} s** (4 h 27 min 9.859 s); epoch CSV elapsed **15,948.1 s**. All 30 CSV epoch rows and losses are finite. Process exit 0 was obtained from the original process session, not inferred from epoch 29. The saved callback identifies best epoch 30; maximum CSV mAP50:95 also occurs at 30.

| Effective setting | Value |
|---|---|
| Initialization | COCO-pretrained YOLOv8n |
| Requested / completed epochs | 30 / 30 |
| imgsz / batch / seed | 640 / 8 / 42 |
| Optimizer | AdamW; lr0 0.000556, lrf 0.01, weight decay 0.0005 |
| Warm-up / patience | 3 epochs / 10; no early stop |
| Device / precision | Apple M5 MPS; effective AMP false |
| Workers | configured 4; effective 0 (Ultralytics MPS behavior) |
| Accumulation / nbs | startup 8 / 64; warm-up may vary accumulation |
| Mosaic | 1.0; disabled for final 5 epochs |
| Environment | Python 3.12.14, PyTorch 2.14.0, Ultralytics 8.4.146 |

Complete configuration/augmentations are saved in run provenance and immutable `args.yaml`. MPS scatter_reduce and index_put_with_accumulate emitted nondeterminism warnings under warn-only deterministic mode. Seed and data selection are reproducible; bitwise numerical replay is not guaranteed. No traceback, nonfinite losses or integrity failure was found. Training began before the initial Git commit, so original provenance honestly has `git_commit: null`; source/config/requirements hashes bind that execution.

## Checkpoint integrity

| File (project-relative local path) | Bytes | SHA-256 |
|---|---:|---|
| `{p["weights"]["path"]}` | {p["weights"]["bytes"]} | `{p["weights"]["sha256"]}` |
| `{p["last_weights"]["path"]}` | {p["last_weights"]["bytes"]} | `{p["last_weights"]["sha256"]}` |

Both checkpoints load, retain the correct 14-class mapping, and contain finite tensors. Ultralytics optimizer stripping resets the internal epoch field to -1; the preserved save callback and CSV establish epoch identity. Both correspond to final epoch 30, but file hashes differ. Weights stay local and are excluded from Git/portable ZIP.

## Fresh standalone best-checkpoint validation

{table}
Evaluation ID: `{NAME}_validation`, completed with exit 0 in {m["duration_seconds"]:.3f} seconds. Frozen val manifest matches the table above: **2,000 images, 24,342 objects**. Settings: MPS, imgsz 640, batch 8, workers 0, confidence floor 0.001, NMS IoU 0.7, max_det 300. AP is evaluated across IoU 0.50:0.05:0.95. These are newly computed metrics, not copied from the final epoch CSV (small numerical differences are preserved).

Precision/Recall are unweighted class means at the common confidence maximizing smoothed mean class F1, **{m["f1_operating_confidence"]:.6f}**, with IoU 0.5 matching. Harmonic aggregate F1 = 2PR/(P+R); macro F1 = mean of 14 per-class F1 scores. Neither is micro-F1. The table explicitly shows both. "Others" precision 1.0 at zero recall is the evaluator's empty-prediction interpolation convention, not perfect detection.

{classes}
Three-wheeler has strongest AP50:95 (0.7403), followed by Bus (0.6430) and Two-wheeler (0.6393). Others (0.0242), Mini-bus (0.1383), and Van (0.3184) are weakest. Only 31 Others and 58 Mini-bus validation instances support those estimates; rare-class reliability is limited. AP confidence intervals and independent test-set generalization have not been measured.

Per-class CSV, numeric 15x15 confusion matrix and PR/F1/P/R curves are under `reports/tables/` and `reports/figures/` with the evaluation ID prefix. Matrix rows are predicted classes, columns are true classes, final row/column are background. The installed validator uses the explicit 0.001 confidence for this matrix, with matching IoU 0.45. It is not the F1 operating-point matrix.

## Prediction review against ground truth

{review["method"]}

"""
        + "\n\n".join(
            f"- **Validation {r['image_id']}:** {r['finding']}" for r in review["cases"]
        )
        + f"""

Source limitations are separate: some visible vehicles lack labels, bus/car subtype distinctions and occlusion box extents can be ambiguous, and source redactions remain. Labels were not changed to improve metrics. Diagnostic counts do not establish a population false-positive rate. No occlusion-stratified AP or small-object AP was measured. Detailed local paired images are in `reports/predictions/error_analysis/review_pair_*.jpg`; portable artifacts retain textual findings, not dataset imagery.

## Proper-checkpoint Apple MPS timing

{timing}
**Inference-only FPS: {t["summary"]["inference_ms"]["fps_from_total_time"]:.3f}. End-to-end still-image FPS: {t["summary"]["end_to_end_ms"]["fps_from_total_time"]:.3f}.** FPS is sample count divided by total stage time (1000 / mean ms), not reciprocal median or average instantaneous FPS.

Apple M5, 24 GiB system memory; MPS, float32, batch 1, imgsz 640, 10 warm-up predictions excluded, 100 deterministic seed42 validation samples. Actual stride-aligned rectangular tensor shapes are recorded per image. {t["preprocessing"]} {t["postprocessing"]}

{t["stage_timing"]} {t["end_to_end_definition"]} Model loading and warm-up are excluded; local image I/O may benefit from OS cache. Median/p95 characterize this sample, not sustained video behavior. The original unsynchronized stage pass is preserved only under ignored `data/interim/` as a diagnostic; it is not used above. The library's batched validation stage timings are also unsynchronized on MPS and must not be presented as useful inference latency.

This single-model accuracy-speed point provides a frozen reference. Inference occupies a small part of file-to-result latency; image I/O and API overhead matter. Low rare-class recall, missed small/occluded vehicles and duplicate detections constrain usefulness despite the measured still-image throughput. Larger inputs/models may improve accuracy but require controlled measurements; no comparative benefit has been demonstrated yet. No predeclared video end-to-end criterion or capture/display pipeline was tested, so no real-time or production-ready claim is made.

## Validation, deliverables and entry gate

The complete saved test result is `reports/audit/closeout_pytest.txt`; additional evaluation consistency, unchanged-run, dashboard, PDF and portable ZIP checks are in `reports/audit/closeout_validation.json`. Data validation re-scans all selected YOLO images/labels with zero corrupt/removed labels. The report, dashboard, faculty PDF, presentation script, offline demonstration and portable ZIP are updated to this proper baseline. Historical smoke/preflight artifacts remain labelled engineering checks.

The subset-based Phase 2 gate requires selected-image integrity, frozen manifests/optimizer, completed training, readable hashed checkpoint, standalone evaluation, proper timing, qualitative review, final report/tests and a Git checkpoint. Acquisition of unselected images is explicitly not required. Run `python scripts/phase2_subset_gate.py` to inspect current evidence. Git branch/push evidence is recorded separately in `reports/audit/closeout_delivery.json`. Do not start Phase 2 training as part of closeout.

## Reproduction commands (do not overwrite the frozen run)

```bash
python3 scripts/show_progress.py
.venv/bin/python -m pytest -q
.venv/bin/python -m src.data.validate_yolo --dataset-version uvh26_mv_yolo_v1/subsets/baseline_seed42_v2
.venv/bin/python scripts/verify_baseline_closeout.py
.venv/bin/python -m streamlit run app.py
```

Actual evaluation command (already executed; its name is immutable and rerunning it intentionally refuses overwrite):

```bash
.venv/bin/python -m src.evaluation.evaluate_baseline --weights runs/{NAME}/weights/best.pt --data data/processed/{f["version"]}/dataset.yaml --name {NAME}_validation --device mps --batch 8
.venv/bin/python -m src.evaluation.benchmark_inference --weights runs/{NAME}/weights/best.pt --data data/processed/{f["version"]}/dataset.yaml --name {NAME} --device mps --warmup 10 --count 100
```

To repeat an evaluation later, supply a new unique output name. Never retrain/overwrite the baseline to reproduce a report. Dataset source: [IISc AIM UVH-26](https://huggingface.co/datasets/iisc-aim/UVH-26); source citation retained in data documentation.
"""
    )
    (ROOT / "docs/phase_reports/PHASE_1_BASELINE.md").write_text(text)
    (
        ROOT / "README.md"
    ).write_text(f"""# UVH-26 vehicle detection: Phase 1 subset baseline

The proper YOLOv8n run completed **30 epochs**, best epoch **30**, exit **0**, no early stopping. Fresh best-checkpoint evaluation, qualitative review and synchronized MPS timing are complete. Phase 2 training has not started; see the [entry gate](reports/audit/phase2_subset_gate.json) and [delivery status](reports/audit/closeout_delivery.json).

**Scope:** 8,000 training / 2,000 validation images, 14 Majority Voting classes. These are subset validation results, not full-dataset or test-set scores.

{table}
F1 above is the harmonic mean of macro Precision/Recall; macro F1 averages class F1 at confidence {m["f1_operating_confidence"]:.4f}. Others has zero recall; its reported precision 1 is an evaluator convention. Three-wheeler AP50:95 is 0.7403; Others 0.0242 and Mini-bus 0.1383 remain weak.

Batch-one Apple M5 MPS timing (640, float32, 10 warm-ups, 100 images): median **{t["summary"]["end_to_end_ms"]["median_ms"]:.2f} ms**, p95 **{t["summary"]["end_to_end_ms"]["p95_ms"]:.2f} ms**, **{t["summary"]["end_to_end_ms"]["fps_from_total_time"]:.2f} end-to-end still images/s**; inference-only **{t["summary"]["inference_ms"]["fps_from_total_time"]:.2f} FPS**. Includes file decode/preprocess/inference/postprocess and synchronized measurement, excludes capture/display. No real-time video or production-readiness claim.

The **26,646-image annotation-catalogue audit** covers metadata. The separate **10,000-image local subset integrity audit** covers actual decoded images, dimensions, hashes, labels and split leakage. Acquisition/integrity verification of unselected images remains incomplete. Frozen subset has 94,609 train and 24,342 validation objects. Raw sources and the completed run are unchanged.

## Faculty demonstration

```bash
python3 scripts/show_progress.py
# Optional dashboard, after installing dependencies:
python -m pip install -r requirements.txt -r requirements-dashboard.txt
python -m streamlit run app.py
python -m pytest -q
```

- [Final Phase 1 report](docs/phase_reports/PHASE_1_BASELINE.md): measured results, configuration, hashes, recovery policy and commands.
- [Presentation script](docs/faculty_review/PRESENTATION_SCRIPT.md) and [start guide](docs/faculty_review/START_HERE.md).
- Faculty PDF: `output/pdf/UVH26_Faculty_Progress_Report.pdf`.
- Portable ZIP (local, generated): `deliverables/UVH26_Faculty_Review_Project.zip`; build with `python scripts/build_faculty_pack.py`.
- [Checkpoint/integrity evidence](reports/audit/baseline_closeout_integrity.json); weights remain local under `runs/{NAME}/weights/`.
- [Measured per-class results](reports/tables/{NAME}_validation_per_class.csv), [latency](reports/tables/{NAME}_latency.json), [prediction review](reports/error_analysis/baseline_prediction_review.json).

## Reproducibility and boundaries

Python 3.12.14 / PyTorch 2.14.0 / Ultralytics 8.4.146, frozen training dependencies in `requirements.txt`. Dashboard dependencies are separate. Follow [data documentation](data/README.md) for the pinned official dataset and immutable preparation policy. The portable ZIP intentionally contains no dataset images, generated YOLO labels or checkpoints. Offline reporting works without training dependencies; inference needs the original local data and weights.

Do not rerun training to demonstrate results. Existing run/evaluation IDs refuse overwrite. No tracking/counting, Phase 2 experiments, independent test scores or full pixel-audit completion is claimed. Initial smoke and full-subset preflight evidence remains historical and separate from this measured baseline.
""")
    (
        ROOT / "docs/faculty_review/PRESENTATION_SCRIPT.md"
    ).write_text(f"""# Faculty presentation script: completed subset baseline

## 1. Problem and scope (45 seconds)

"Ma'am, our project detects 14 vehicle categories in Indian road scenes using YOLOv8n and IISc's UVH-26 Majority Voting annotations. Today I am presenting the completed Phase 1 subset baseline. This is object detection; traffic tracking and counting are future work."

## 2. Data work and integrity (60 seconds)

"We audited the full catalogue of 26,646 image records and 316,220 boxes. Separately, we decoded, dimension-checked and hashed the actual 8,000 training and 2,000 validation images used here. These have 94,609 and 24,342 objects, with all 14 classes and no selected train-validation content overlap. Unselected-image acquisition and integrity are still incomplete."

"One image declared 1920 pixels width but was actually 1620. Visual comparison could not justify a simple coordinate repair, so we quarantined it without changing the source. We also excluded two degraded candidate files and froze deterministic replacements before training."

## 3. What actually trained (45 seconds)

"The COCO-pretrained YOLOv8n trained for all 30 epochs on Apple M5 MPS, using image size 640, batch 8, seed 42 and frozen AdamW settings. It finished with exit code zero, in approximately 4 hours 27 minutes. Best epoch was 30; early stopping did not occur. Both checkpoints are readable and checksummed. MPS nondeterminism warnings mean we do not promise bitwise identical replay."

## 4. Measured accuracy (75 seconds)

"We evaluated best.pt again, independently, on the same frozen 2,000-image validation set. Precision is {m["precision"]:.4f}, Recall {m["recall"]:.4f}, mAP50 {m["map50"]:.4f}, and mAP50 to 95 {m["map50_95"]:.4f}. Harmonic aggregate F1 is {m["f1"]:.4f}; macro per-class F1 is {m["macro_f1"]:.4f}. They differ because averaging and taking a harmonic mean are different operations."

"Three-wheelers perform best, with AP50 to 95 of 0.7403. Mini-bus is 0.1383 and Others 0.0242. Others has zero recall at the selected F1 threshold; the precision value of one is an evaluator convention, not perfect performance. Rare categories have only 31 and 58 validation examples. These are validation results used for checkpoint selection, not independent test accuracy."

## 5. Visual findings and timing (75 seconds)

"In our local paired ground-truth review, the model detects many motorcycles and auto-rickshaws well. It misses distant small vehicles, duplicates boxes in dense scenes, calls a Mini-bus a Bus, and calls a construction vehicle a Truck instead of Others. Some visible vehicles are unlabelled in the source, so not every unmatched prediction is a hallucination."

"We timed the proper checkpoint with MPS synchronization, batch one, 10 warm-ups and 100 images. Inference averaged {t["summary"]["inference_ms"]["mean_ms"]:.2f} milliseconds. File-to-result latency had median {t["summary"]["end_to_end_ms"]["median_ms"]:.2f} and p95 {t["summary"]["end_to_end_ms"]["p95_ms"]:.2f} milliseconds, corresponding to {t["summary"]["end_to_end_ms"]["fps_from_total_time"]:.2f} still images per second. This excludes camera capture and display; I am not claiming real-time video or production readiness."

## 6. Evidence and next step (30 seconds)

"The repository includes measured results, PR/F1 curves, confusion matrix, immutable provenance, tests, dashboard and this faculty report. Controlled Phase 2 experiments can begin only after the saved subset entry gate and Git checkpoint pass. No Phase 2 training has been run in this closeout. The next experiments should change one factor at a time and compare accuracy, rare-class behavior and identically defined latency."

## Two-minute demonstration (safe offline commands)

From the project or extracted ZIP root:

```bash
python3 scripts/show_progress.py
```

With the project environment installed:

```bash
python -m streamlit run app.py
python -m pytest -q
```

Open the PDF, show the current baseline table, per-class values, PR/F1 curves and timing definitions. The ZIP works as an artifact review without images or weights. On the original project machine only, show local `reports/predictions/error_analysis/review_pair_4232.jpg`, `review_pair_10518.jpg`, `review_pair_4711.jpg` and `review_pair_954.jpg` for successes, small misses and rare-class errors. Do not run training/evaluation during a short presentation.

## Likely questions

- **Is Phase 1 the full dataset?** No. Catalogue audit is full; actual pixel integrity/training/evaluation are explicitly subset-based.
- **Why best epoch equals last?** Validation fitness peaked at epoch 30; both provenance callback and CSV agree. We still evaluated best.pt independently.
- **Why are two F1 values different?** One is harmonic mean of averaged P/R, the other averages per-class F1. Neither is micro-F1.
- **Can this count traffic yet?** No; tracking, counting, sustained video latency and deployment validation are future work.
- **Why not repair bad labels during validation?** Changing frozen ground truth after seeing predictions would compromise comparison. Source limitations are documented separately.
- **Does 31.89 FPS mean real-time?** It measures sequential local still-image processing only. A video criterion and complete video pipeline have not been tested.
""")
    (ROOT / "docs/faculty_review/START_HERE.md").write_text("""# Faculty review package

Phase 1 proper YOLOv8n subset baseline: 30 completed epochs, best epoch 30, standalone 2,000-image validation and synchronized batch-one MPS timing.

1. Open `output/pdf/UVH26_Faculty_Progress_Report.pdf`.
2. Read `docs/faculty_review/PRESENTATION_SCRIPT.md`.
3. Run `python3 scripts/show_progress.py` from this folder; standard library only, no downloads.
4. For the dashboard, install `requirements.txt` and `requirements-dashboard.txt`, then run `python -m streamlit run app.py`.
5. With project dependencies installed, run `python -m pytest -q`.

The ZIP includes source, configuration, summarized audits, measured metrics, plots, tests and documentation. It excludes dataset images, generated labels, model weights, full runs and machine-local paths. Local inference/reverification of original data requires the original project machine; offline demonstration reads saved evidence.

These are subset validation results, not test-set or full-dataset results. Full catalogue annotation audit and 10,000-image local pixel audit are distinct. Unselected-image integrity remains incomplete. No real-time video, production-readiness or Phase 2 completion claim is made. Current entry-gate and Git delivery status are saved under `reports/audit/`.
""")
    (ROOT / "README_STREAMLIT.md").write_text("""# Artifact-based Phase 1 dashboard

The Streamlit dashboard reads the completed proper 30-epoch baseline provenance, fresh best-checkpoint evaluation, per-class scores, curves and synchronized MPS timing from saved artifacts. The historical smoke pilot remains explicitly labelled separately.

```bash
python -m pip install -r requirements.txt -r requirements-dashboard.txt
python -m streamlit run app.py
```

Launch from the project root or extracted faculty package. No training or download starts on launch. The optional test button runs pytest. Dataset images/weights are deliberately absent from the portable ZIP; metrics, figures and PDF still work. The animated hero boxes are illustrative, not model predictions.

Present the proper baseline section first, then class results, timing definitions, audit boundaries and limitations. Full 26,646-image catalogue audit does not imply full pixel verification. All selected 10,000 images passed integrity; unselected acquisition remains incomplete. End-to-end timing is still-image file-to-result processing, not video capture/display FPS. No Phase 2 training is included.
""")


if __name__ == "__main__":
    main()
