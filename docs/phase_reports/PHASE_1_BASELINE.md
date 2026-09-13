# Phase 1: UVH-26 MV subset baseline closeout

The proper YOLOv8n baseline completed **30/30 epochs**, process exit **0**, best checkpoint **epoch 30**, with **no early stopping**. Fresh standalone evaluation and synchronized Apple MPS timing are complete. Phase 2 training has not started. The final delivery/gate status is recorded in `reports/audit/phase2_subset_gate.json`; a Git checkpoint is required before controlled Phase 2 work.

This is a **subset validation baseline**, not a full-dataset result, test-set result, production readiness claim or completed traffic-analytics system.

## Data scope and integrity

| Evidence | Images | Object boxes | What was checked |
|---|---:|---:|---|
| Full MV annotation catalogue | 26,646 | 316,220 | JSON schema, classes, box geometry and split identifiers |
| Frozen local training subset | 8,000 | 94,609 | Decode, actual dimensions, paths, content hashes, labels and leakage |
| Frozen local validation subset | 2,000 | 24,342 | Same selected-image integrity checks |

The full catalogue contains 21,349 training and 5,297 validation image records. All selected 10,000 images passed pixel/label integrity, with zero train/validation content overlap and all 14 classes retained. Acquisition and integrity verification of **unselected images remains incomplete**. The catalogue audit must not be described as a full pixel audit. Semantic manual annotation review covered 42 selected images; closeout prediction review covered six validation scenes, not every image.

Source: IISc AIM UVH-26, Majority Voting, revision `59f82c57821e8a54dc40bc1f42e83909dbad0b70`, CC BY 4.0. Original class IDs 1-14 map in order to YOLO IDs 0-13. STAPLE annotations are not mixed into this experiment. Maximum selected-vs-full class-share deviation is 0.306231 percentage points.

Frozen version: `uvh26_mv_yolo_v1/subsets/baseline_seed42_v2`.

| Record | SHA-256 |
|---|---|
| Combined manifest | `990054a93e300a90321db19b3d0bcd98a488a891cd4e2dbd88425f4eb592c2af` |
| Training manifest | `8e4a72413caecc9defee68e75f498b71160b06fb33f30f6cbe2dc8b3677032cf` |
| Validation manifest | `fd23d2e417d70a8614b6312cb3eff1deb0ed98f0532269c22878269fdc5959d3` |
| Class mapping | `6fd0b458fed2cb174c924d4fa9ead86db0a292dc7c55fd0f4d4b3b9b20d08ea8` |
| Selected-image audit | `c9534e58ca40a16e7a7777d790156a97a2add846f76d43735a6c0b48226c6408` |

Closeout independently rehashed all 10,000 images and labels, both checkpoints, mapping, manifests, audit, frozen optimizer config, training source and requirements. Saved training dataset paths resolve to the final v2 subset. Evidence: `reports/audit/baseline_closeout_integrity.json` and `baseline_subset_frozen_provenance.json`.

## Source-image recovery history

Train ID 21818, `UVH-26-Train/data/003/803489.png`, decodes at 1620x1080 but declares 1920x1080. Raw-coordinate and width-rescaled overlays gave inconsistent object alignment; neither justified automatic repair. The raw file remains unchanged and quarantined. Its first candidate replacement, ID 5235 (`Train/data/001/341297.png`), was structurally valid but visually degraded and also excluded. Final replacement: train ID 25440 (`Train/data/000/233625.png`), preserving one three-wheeler and two two-wheelers.

Validation ID 20260 (`Val/data/001/986228.png`) was severely visually degraded; final replacement ID 6452 (`Val/data/000/81395.png`) preserves two two-wheelers and one bicycle. Three unique files were quarantined; two original candidate images changed. No clipping, silent resizing or relabelling was performed. The earlier frozen v1 is preserved as superseded before baseline training. `baseline_subset_v2_provenance_addendum.json` clarifies the v2 replacement search: all same-split official candidates, deterministic exposure-distance ranking/tie break and pinned acquisition if needed; the inherited v1 provenance wording is retained unchanged.

## Actual training outcome

Run ID: `yolov8n_uvh26_mv_baseline_seed42_v1`. The original run directory and logs are preserved unchanged; no retraining or resume was used for closeout. Start 2026-09-13T13:32:35.295730+00:00; completion 2026-09-13T17:59:45.280500+00:00. Wall duration **16029.859 s** (4 h 27 min 9.859 s); epoch CSV elapsed **15,948.1 s**. All 30 CSV epoch rows and losses are finite. Process exit 0 was obtained from the original process session, not inferred from epoch 29. The saved callback identifies best epoch 30; maximum CSV mAP50:95 also occurs at 30.

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
| `runs/yolov8n_uvh26_mv_baseline_seed42_v1/weights/best.pt` | 6228714 | `85b9e089ec3dfaa676e30a6191b7e5726f320d5bf85391f1675922b88f9778b3` |
| `runs/yolov8n_uvh26_mv_baseline_seed42_v1/weights/last.pt` | 6228714 | `19cf82c3848fafae370fa36dd16e845a26dc38eb83bd80a473bb24691268551a` |

Both checkpoints load, retain the correct 14-class mapping, and contain finite tensors. Ultralytics optimizer stripping resets the internal epoch field to -1; the preserved save callback and CSV establish epoch identity. Both correspond to final epoch 30, but file hashes differ. Weights stay local and are excluded from Git/portable ZIP.

## Fresh standalone best-checkpoint validation

| Checkpoint / evaluation | Precision | Recall | F1 (harmonic aggregate) | Macro F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n best epoch 30 / frozen 2,000 val | 0.609993 | 0.543547 | 0.574856 | 0.539397 | 0.560049 | 0.458407 |

Evaluation ID: `yolov8n_uvh26_mv_baseline_seed42_v1_validation`, completed with exit 0 in 82.495 seconds. Frozen val manifest matches the table above: **2,000 images, 24,342 objects**. Settings: MPS, imgsz 640, batch 8, workers 0, confidence floor 0.001, NMS IoU 0.7, max_det 300. AP is evaluated across IoU 0.50:0.05:0.95. These are newly computed metrics, not copied from the final epoch CSV (small numerical differences are preserved).

Precision/Recall are unweighted class means at the common confidence maximizing smoothed mean class F1, **0.309309**, with IoU 0.5 matching. Harmonic aggregate F1 = 2PR/(P+R); macro F1 = mean of 14 per-class F1 scores. Neither is micro-F1. The table explicitly shows both. "Others" precision 1.0 at zero recall is the evaluator's empty-prediction interpolation convention, not perfect detection.

| Class | Val objects | Precision | Recall | F1 | AP@0.5 | AP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| Hatchback | 2402 | 0.5621 | 0.6574 | 0.6060 | 0.6345 | 0.5341 |
| Sedan | 1169 | 0.5180 | 0.4936 | 0.5055 | 0.5603 | 0.4884 |
| SUV | 1028 | 0.4360 | 0.5691 | 0.4937 | 0.4720 | 0.4115 |
| MUV | 541 | 0.4421 | 0.4291 | 0.4355 | 0.4075 | 0.3645 |
| Bus | 627 | 0.7511 | 0.7124 | 0.7312 | 0.7695 | 0.6430 |
| Truck | 933 | 0.6074 | 0.6506 | 0.6282 | 0.6801 | 0.5454 |
| Three-wheeler | 4016 | 0.8537 | 0.8391 | 0.8464 | 0.9036 | 0.7403 |
| Two-wheeler | 11624 | 0.8152 | 0.8016 | 0.8084 | 0.8776 | 0.6393 |
| LCV | 1322 | 0.6659 | 0.7080 | 0.6863 | 0.7196 | 0.5790 |
| Mini-bus | 58 | 0.2179 | 0.1552 | 0.1813 | 0.1891 | 0.1383 |
| Tempo-traveller | 130 | 0.6747 | 0.6062 | 0.6386 | 0.6504 | 0.5694 |
| Bicycle | 278 | 0.5884 | 0.5502 | 0.5686 | 0.5742 | 0.4220 |
| Van | 183 | 0.4074 | 0.4372 | 0.4218 | 0.3626 | 0.3184 |
| Others | 31 | 1.0000 | 0.0000 | 0.0000 | 0.0396 | 0.0242 |

Three-wheeler has strongest AP50:95 (0.7403), followed by Bus (0.6430) and Two-wheeler (0.6393). Others (0.0242), Mini-bus (0.1383), and Van (0.3184) are weakest. Only 31 Others and 58 Mini-bus validation instances support those estimates; rare-class reliability is limited. AP confidence intervals and independent test-set generalization have not been measured.

Per-class CSV, numeric 15x15 confusion matrix and PR/F1/P/R curves are under `reports/tables/` and `reports/figures/` with the evaluation ID prefix. Matrix rows are predicted classes, columns are true classes, final row/column are background. The installed validator uses the explicit 0.001 confidence for this matrix, with matching IoU 0.45. It is not the F1 operating-point matrix.

## Prediction review against ground truth

100 seeded random validation images plus dense/sparse/class-coverage selection, deduplicated to 101. Six paired GT/prediction images manually inspected. conf .10, NMS IoU .7; class-agnostic greedy diagnostic matching at IoU .5. Counts are not AP, not an unbiased error-rate estimate, and unmatched predictions may include unlabelled true vehicles.

- **Validation 4232:** All 10 annotated objects have correct-class IoU matches, including foreground truck and distant/partly occluded two-wheelers. Two unmatched predictions remain; this is a success example, not perfect scene accuracy.

- **Validation 10518:** 15 correct-class matches out of 17 GT; distant annotated Two-wheeler index16 is missed (65.44 square pixels after long-side scaling to 640). Foreground motorcycles are detected. One LCV/Truck class confusion.

- **Validation 1364:** Dense intersection: 56 GT, 48 correct-class matches, 3 unmatched GT and 71 unmatched predictions at low conf .10. Heavy overlap and partial occlusion; duplicate cross-class boxes, Bus/Mini-bus/LCV/Truck confusion; pedestrian at lower right is predicted as Bicycle. These are model errors, though not every unmatched prediction is a false vehicle.

- **Validation 21621:** Dense occluded queue: 40 GT, 33 correct-class matches; no unmatched GT under class-agnostic matching, but 7 class confusions. Foreground handcart labelled Others is predicted Three-wheeler; additional low-confidence Two-wheeler box on produce cart is a false positive. Passenger-car subtype confusion and duplicate boxes persist.

- **Validation 4711:** Large foreground MUV and two-/three-wheelers detected correctly; source Mini-bus is predicted Bus. Visible partial auto-rickshaw at right border lacks a corresponding GT box; its detection is an annotation-relative unmatched prediction, not established model hallucination. Fine-grained bus taxonomy needs source review, with labels kept unchanged.

- **Validation 954:** Eight motorcycles correctly matched, but the clearly visible construction vehicle labelled Others is confidently predicted Truck (0.88). Rare-class semantic confusion, not a dimension/alignment problem.

Source limitations are separate: some visible vehicles lack labels, bus/car subtype distinctions and occlusion box extents can be ambiguous, and source redactions remain. Labels were not changed to improve metrics. Diagnostic counts do not establish a population false-positive rate. No occlusion-stratified AP or small-object AP was measured. Detailed local paired images are in `reports/predictions/error_analysis/review_pair_*.jpg`; portable artifacts retain textual findings, not dataset imagery.

## Proper-checkpoint Apple MPS timing

| Stage | Mean ms | Median ms | p95 ms |
|---|---:|---:|---:|
| end_to_end_ms | 31.357 | 31.732 | 34.359 |
| inference_ms | 4.956 | 3.768 | 7.358 |
| postprocess_ms | 2.466 | 1.404 | 4.532 |
| preprocess_ms | 1.691 | 1.017 | 3.902 |

**Inference-only FPS: 201.766. End-to-end still-image FPS: 31.891.** FPS is sample count divided by total stage time (1000 / mean ms), not reciprocal median or average instantaneous FPS.

Apple M5, 24 GiB system memory; MPS, float32, batch 1, imgsz 640, 10 warm-up predictions excluded, 100 deterministic seed42 validation samples. Actual stride-aligned rectangular tensor shapes are recorded per image. OpenCV local path read/decode, stride-aligned rectangular letterbox to imgsz=640, BGR-to-RGB, CHW float32 tensor /255; actual tensor shapes recorded per sample. Confidence filtering and class-aware NMS at conf .25 / IoU .7 / max_det 300; box scaling and Results construction.

Explicit torch.mps.synchronize before and after each stage; no reliance on unsynchronized result.speed. End-to-end includes these instrumentation barriers. Wall time around predict(path), including local image read/decode, preprocessing, inference and postprocessing plus device synchronization; excludes model loading, drawing, video capture and UI. Model loading and warm-up are excluded; local image I/O may benefit from OS cache. Median/p95 characterize this sample, not sustained video behavior. The original unsynchronized stage pass is preserved only under ignored `data/interim/` as a diagnostic; it is not used above. The library's batched validation stage timings are also unsynchronized on MPS and must not be presented as useful inference latency.

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
.venv/bin/python -m src.evaluation.evaluate_baseline --weights runs/yolov8n_uvh26_mv_baseline_seed42_v1/weights/best.pt --data data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2/dataset.yaml --name yolov8n_uvh26_mv_baseline_seed42_v1_validation --device mps --batch 8
.venv/bin/python -m src.evaluation.benchmark_inference --weights runs/yolov8n_uvh26_mv_baseline_seed42_v1/weights/best.pt --data data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2/dataset.yaml --name yolov8n_uvh26_mv_baseline_seed42_v1 --device mps --warmup 10 --count 100
```

To repeat an evaluation later, supply a new unique output name. Never retrain/overwrite the baseline to reproduce a report. Dataset source: [IISc AIM UVH-26](https://huggingface.co/datasets/iisc-aim/UVH-26); source citation retained in data documentation.
