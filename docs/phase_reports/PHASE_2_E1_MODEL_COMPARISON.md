# Phase 2 E1: YOLOv8s versus frozen YOLOv8n

E1 is trained and independently evaluated on the frozen subset. This report compares measured accuracy, training cost and common-protocol synchronized MPS timing. No other Phase 2 experiment was started. Git delivery status is recorded in `reports/audit/E1_delivery.json`.

## Question and experimental boundary

Does increasing model capacity from YOLOv8n to YOLOv8s improve UVH-26 detection, especially difficult classes, enough to justify its additional cost and latency? Model size and corresponding official COCO initializer are the intended independent variable. This is one seed, one fixed subset and validation-based checkpoint selection; no independent test-set accuracy, significance claim, live-video FPS or production-readiness claim.

Run IDs: E0 `yolov8n_uvh26_mv_baseline_seed42_v1`; E1 `E1_yolov8s_uvh26_mv_640_seed42`; recovery preflight `E1_yolov8s_uvh26_mv_640_seed42_preflight_v2`. Dataset is `uvh26_mv_yolo_v1/subsets/baseline_seed42_v2`: 8,000 training / 2,000 validation images, 14 unchanged MV classes; 94,609 / 24,342 objects. Validation manifest SHA-256 `fd23d2e417d70a8614b6312cb3eff1deb0ed98f0532269c22878269fdc5959d3`. Pretraining verification rechecked all 10,000 image/label hashes, paths and E0 checkpoint/run hashes. The gate passed before preflight. A pre-existing faculty PDF serialization change was reviewed and preserved; its normalized text matched the committed report.

Full 26,646-image annotation-catalogue audit and 10,000-image local pixel audit remain distinct. Acquisition and integrity verification of unselected images remain incomplete and are outside this comparison.

## Controls and executed training

Frozen E1 config: `configs/E1_yolov8s_uvh26_mv_640_seed42.yaml`. Image size640, seed42, 30-epoch budget, patience10, AdamW lr0 .000556, lrf .01, weight decay .0005, warm-up3, nbs64 and E0 augmentation/schedule settings. Explicit AMP=false and workers=0 match E0 effective behavior (E0 requested true/4). Actual E1 batch **8**, effective AMP **False**, workers **0**, startup accumulation **8**. Any effective-control differences are recorded in the completed-run verification; no silent equivalence claim.

COCO-pretrained YOLOv8s came from official Ultralytics assets v8.4.0: initializer SHA-256 `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`. E0 initialization and finished checkpoint remain unchanged. E1 uses a separate runner derived from E0, with correct model-source naming and an epoch-level finite-loss guard. E0 source was not edited.

Preflight v1 exited 1 after its training batches because the added loss guard did not accept named-loss dictionaries; it saved no checkpoint. The corrected guard has regression tests. Recovery v2 used one epoch on 800 training images and all 2,000 validation images and took 249.748s, saved readable finite checkpoints and passed visual class/placement checks. It was a pipeline test, not an E1 accuracy result. Its full-budget duration estimate was 9.44h; the actual duration is below. Original preflight and E0 files remain separate and preserved.

| Model | Epochs / best | Early stop | Recorded timer seconds | Parameters (evaluation model) | Checkpoint bytes |
| --- | --- | --- | --- | --- | --- |
| E0 | 30 / 30 | False | 16029.859 | 3013578 | 6228714 |
| E1 | 30 / 22 | False | 28459.257 | 11141018 | 22502762 |

Calendar elapsed seconds: E0 16029.985; E1 145894.819. E1's recorded perf_counter duration is 28459.257 s (7.905 h), whereas its timestamp span is 145894.819 s (40.526 h). macOS sleep cycles are documented in E1_execution_environment_note.json. Neither the calendar ratio nor the timer ratio is a clean controlled estimate of model-capacity compute cost; sleep, background load and timer behavior limit interpretation.

E1 process exit: 0. Start 2026-09-15T06:56:52.970261+00:00; end 2026-09-16T23:28:27.789026+00:00. Best epoch comes from the save callback, cross-checked against CSV validation fitness; optimizer-stripped checkpoint epoch metadata may be -1. Complete effective args, finite losses, per-epoch CSV, warnings and checkpoint hashes are preserved. MPS nondeterminism under warn-only settings limits bitwise numerical replay.

## Evaluation recovery and fresh standalone validation

Training and checkpoint integrity exited 0; only the original evaluation export failed. Installed YOLO.val keeps its validator local, so the old model.validator access fell through to DetectionModel. The repaired evaluator retains a validator through the supported val(validator=...) hook, extracts standard DetMetrics and atomically publishes complete bundles. See [recovery note](E1_EVALUATION_RECOVERY.md). Both checkpoints were reevaluated using the same repaired source; E0 exactly reproduces its Phase 1 overall metrics. Source/package versions and prediction counts are recorded in each bundle.

Evaluation IDs: yolov8n_uvh26_mv_e0_validation_seed42_v2 and yolov8s_uvh26_mv_e1_validation_seed42_v2. Python 3.12.14, PyTorch 2.14.0, Ultralytics 8.4.146, macOS 26.6.2 arm64, Apple M5 with 24 GiB RAM. Both processed 2000 images / 24342 GT objects. These do not constitute independent test results.

| Metric | E0 YOLOv8n | E1 YOLOv8s | E1 - E0 (pp) |
| --- | --- | --- | --- |
| precision | 0.609993 | 0.650335 | +4.034 |
| recall | 0.543547 | 0.610255 | +6.671 |
| f1 | 0.574856 | 0.629658 | +5.480 |
| macro_f1 | 0.539397 | 0.591856 | +5.246 |
| map50 | 0.560049 | 0.623274 | +6.322 |
| map50_95 | 0.458407 | 0.524760 | +6.635 |


| Measurement | E0 | E1 | Absolute change (listed unit) | Relative change % | Unit |
| --- | --- | --- | --- | --- | --- |
| precision | 0.609993 | 0.650335 | +0.040342 | +6.614 | fraction |
| recall | 0.543547 | 0.610255 | +0.066709 | +12.273 | fraction |
| f1 | 0.574856 | 0.629658 | +0.054802 | +9.533 | fraction |
| macro_f1 | 0.539397 | 0.591856 | +0.052458 | +9.725 | fraction |
| map50 | 0.560049 | 0.623274 | +0.063225 | +11.289 | fraction |
| map50_95 | 0.458407 | 0.524760 | +0.066353 | +14.475 | fraction |
| parameters | 3013578.000000 | 11141018.000000 | +8127440.000000 | +269.694 | count |
| checkpoint_bytes | 6228714.000000 | 22502762.000000 | +16274048.000000 | +261.275 | bytes |
| end_to_end_mean_ms | 30.363652 | 30.292472 | -0.071181 | -0.234 | ms |
| end_to_end_median_ms | 31.166604 | 30.536812 | -0.629791 | -2.021 | ms |
| end_to_end_p95_ms | 33.681289 | 32.062650 | -1.618639 | -4.806 | ms |
| end_to_end_fps_from_total_time | 32.934114 | 33.011502 | +0.077388 | +0.235 | FPS |

Absolute accuracy changes in the first table are percentage points; the combined table uses raw fractions. Relative change is 100*(E1/E0-1).
The E1 best checkpoint was freshly validated using E0's evaluator settings: same manifest, MPS, imgsz640, batch8, confidence floor .001, NMS IoU .7, max_det300. Precision/Recall are macro class means at each model's max smoothed mean-F1 confidence. Harmonic aggregate F1 is 2PR/(P+R); macro per-class F1 averages class F1. Neither is micro-F1. AP50:95 averages IoU .50:.05:.95. Accuracy differences in the first table are absolute percentage points; the combined table separately labels relative-percent changes.

| Class | P E0 / E1 | R E0 / E1 | AP50 E0 / E1 | AP50:95 E0 / E1 | AP50:95 delta pp |
| --- | --- | --- | --- | --- | --- |
| Hatchback | 0.5621 / 0.5744 | 0.6574 / 0.7190 | 0.6345 / 0.6916 | 0.5341 / 0.5971 | +6.302 |
| Sedan | 0.5180 / 0.5883 | 0.4936 / 0.6039 | 0.5603 / 0.6583 | 0.4884 / 0.5916 | +10.322 |
| SUV | 0.4360 / 0.4581 | 0.5691 / 0.5866 | 0.4720 / 0.5154 | 0.4115 / 0.4608 | +4.922 |
| MUV | 0.4421 / 0.4400 | 0.4291 / 0.4850 | 0.4075 / 0.4730 | 0.3645 / 0.4305 | +6.596 |
| Bus | 0.7511 / 0.7695 | 0.7124 / 0.7615 | 0.7695 / 0.8266 | 0.6430 / 0.7044 | +6.140 |
| Truck | 0.6074 / 0.6525 | 0.6506 / 0.6838 | 0.6801 / 0.7103 | 0.5454 / 0.5788 | +3.339 |
| Three-wheeler | 0.8537 / 0.8615 | 0.8391 / 0.8725 | 0.9036 / 0.9237 | 0.7403 / 0.7814 | +4.110 |
| Two-wheeler | 0.8152 / 0.8256 | 0.8016 / 0.8615 | 0.8776 / 0.9119 | 0.6393 / 0.6925 | +5.329 |
| LCV | 0.6659 / 0.6572 | 0.7080 / 0.7685 | 0.7196 / 0.7584 | 0.5790 / 0.6198 | +4.090 |
| Mini-bus | 0.2179 / 0.3632 | 0.1552 / 0.3103 | 0.1891 / 0.2242 | 0.1383 / 0.1736 | +3.531 |
| Tempo-traveller | 0.6747 / 0.7627 | 0.6062 / 0.6385 | 0.6504 / 0.7581 | 0.5694 / 0.6850 | +11.552 |
| Bicycle | 0.5884 / 0.6612 | 0.5502 / 0.6295 | 0.5742 / 0.6979 | 0.4220 / 0.5441 | +12.217 |
| Van | 0.4074 / 0.4907 | 0.4372 / 0.6230 | 0.3626 / 0.5166 | 0.3184 / 0.4565 | +13.808 |
| Others | 1.0000 / 1.0000 | 0.0000 / 0.0000 | 0.0396 / 0.0599 | 0.0242 / 0.0306 | +0.638 |

All 14 AP50:95 changes are positive; no class AP regression was observed. Largest gains: Van +13.808 pp, Bicycle +12.217 pp, Tempo-traveller +11.552 pp, Sedan +10.322 pp. Strongest E1 class is Three-wheeler (0.7814); Two-wheeler reaches 0.6925. Weakest remain Others (0.0306) and Mini-bus (0.1736). Precision regresses for LCV (-0.877 pp) and MUV (-0.213 pp), while recall improves; P/R are measured at each model's own max-F1 confidence. Others recall stays zero.

Rare-class estimates are uncertain: Mini-bus has 58 validation objects and Others 31; Van has 183. Others precision can equal one when recall is zero due to the evaluator's empty-prediction interpolation convention. Per-class CSV includes separate P/R/F1/AP deltas. No claim that every class benefits is implied by aggregate mAP.

Numeric confusion changes (`confusion_delta.json`) are E1 minus E0, rows predicted / columns true, final row/column background. Both use confidence .001 and matching IoU .45, distinct from the F1 operating point. Negative off-diagonal/background values indicate fewer errors for those cells, not a standalone AP change.

At this low .001 confusion threshold, correct-class matches rise 12671 to 14540; wrong-class matches fall 11471 to 9659; unmatched GT falls 200 to 143; unmatched predictions fall 290417 to 247459. These permissive-threshold counts are not operating-point error rates. Others-to-background false positives increase by 281; Hatchback predictions on true Sedan increase by 40. Annotation omissions can contribute to unmatched predictions.

## Comparable MPS speed

Both models were freshly benchmarked after E1 training, sequentially without another model job on MPS. The original Phase 1 E0 measurement was preserved. Common protocol: Apple M5, MPS float32, batch1, imgsz640 rectangular letterbox, identical seed42 100-image order, 10 warm-ups, conf .25/NMS .7/max_det300, explicit synchronization before/after each stage. Actual input tensor shapes are checked for equality.

| Stage | Statistic | E0 | E1 | E1 / E0 |
| --- | --- | --- | --- | --- |
| preprocess_ms | mean_ms | 1.120 | 0.964 | 0.861 |
| preprocess_ms | median_ms | 1.225 | 0.834 | 0.681 |
| preprocess_ms | p95_ms | 1.356 | 1.335 | 0.985 |
| preprocess_ms | fps_from_total_time | 892.979 | 1037.334 | 1.162 |
| inference_ms | mean_ms | 6.179 | 7.233 | 1.171 |
| inference_ms | median_ms | 6.753 | 7.054 | 1.045 |
| inference_ms | p95_ms | 7.725 | 7.776 | 1.007 |
| inference_ms | fps_from_total_time | 161.851 | 138.256 | 0.854 |
| postprocess_ms | mean_ms | 1.822 | 1.466 | 0.804 |
| postprocess_ms | median_ms | 1.685 | 1.165 | 0.691 |
| postprocess_ms | p95_ms | 2.664 | 2.428 | 0.911 |
| postprocess_ms | fps_from_total_time | 548.881 | 682.294 | 1.243 |
| end_to_end_ms | mean_ms | 30.364 | 30.292 | 0.998 |
| end_to_end_ms | median_ms | 31.167 | 30.537 | 0.980 |
| end_to_end_ms | p95_ms | 33.681 | 32.063 | 0.952 |
| end_to_end_ms | fps_from_total_time | 32.934 | 33.012 | 1.002 |

End-to-end includes local image read/decode, preprocess, forward pass, postprocess, API overhead and synchronization instrumentation. Excludes model load/warm-up, drawing, capture and display. FPS is 1000/mean ms. E1 mean inference is 17.1% slower, but its measured mean end-to-end latency is 0.23% lower; this tiny sequential-run difference does not establish a speed advantage. Disk decode/API overhead and cache/load variation dominate this still-image comparison. OS cache and thermal/background load can affect timing; these sequential measurements are not a controlled laboratory repeated-trials estimate or sustained live-video throughput.

## Paired diagnostic review

GT/E0/E1 compared on the same six validation scenes: 4232,10518,1364,21621,4711,954, spanning dense traffic, small/distant vehicles, occlusions and rare classes. Confidence .10, NMS .7 and diagnostic class-agnostic greedy IoU .5 matching; not unbiased population error rates or COCO AP matching.

- **Validation 4232:** Both models correctly match all 10 GT vehicles, including foreground truck and parked/occluded motorcycles. E1 has four unmatched predictions versus two for E0; the pedestrian beside the truck receives a spurious vehicle box. Larger capacity does not remove false positives.

- **Validation 10518:** E1 increases correct matches from 15 to 16 and recovers the distant small two-wheeler missed by E0 (GT index 16). Both still confuse the LCV with Truck; E1 has six unmatched predictions versus five. This single selected example is not a size-stratified AP estimate.

- **Validation 1364:** Dense junction: correct matches rise 48 to 50 of 56 GT; unmatched GT falls 3 to 2 and unmatched predictions 71 to 40 at confidence .10. Duplicate subtype boxes and fine-grained vehicle confusions remain. Occlusion makes visible extents and source box boundaries uncertain.

- **Validation 21621:** Dense occluded street is a counterexample: E1 correct matches fall 33 to 31 of 40 GT while class confusions rise seven to nine. Both call the Others handcart Three-wheeler. Source subtype ambiguity and partial-vehicle boxes are separate from these scored model errors.

- **Validation 4711:** Correct matches increase six to seven of nine GT and unmatched predictions drop 11 to seven. E1 improves the SUV label, but both classify the Mini-bus as Bus and a Sedan as Hatchback. The extreme-right auto is an apparent source-label omission; an unmatched detection there is not necessarily hallucinated.

- **Validation 954:** Both correctly match eight two-wheelers; both misclassify the Others construction vehicle as Truck. E1 adds a duplicate around a distant motorcycle and has four unmatched predictions versus three. Rare-class failure persists despite the overall gain.

Source omissions, ambiguous fine-grained labels, loose occlusion boxes and redactions remain separate from model failures. Frozen annotations were not edited after observing results. Local comparison images are under `reports/predictions/E1_vs_E0_paired_v1/`; dataset imagery is excluded from Git.

## Assessment and single next experiment

Select E1 YOLOv8s at 640 as the preferred research detector for the next traffic-analytics development stage: mAP50:95 improves by 6.635 percentage points, all 14 class AP50:95 values improve, and the matched still-image mean end-to-end latency is effectively unchanged (30.364 vs 30.292 ms). This choice prioritizes detection quality on the measured subset; it is not a deployment/real-time certification. E1 costs 3.70x evaluation-model parameters and 3.61x checkpoint bytes, and mean inference time increases 17.1% (6.179 to 7.233 ms). The small end-to-end timing difference is within likely cache/load/order variability and does not establish that YOLOv8s is faster. Rare Others recall stays zero; dense-scene class confusion can worsen. Single next experiment recommendation: a controlled 640-versus-960 resolution comparison on E1, with the same frozen data/seed/budget and matched latency measurement, to test remaining small/distant and occluded-vehicle weaknesses. This recommendation is conditional on compute feasibility and further authorization; no new training was started.

## Tests, artifacts and delivery

........................................................................ [ 98%]
.                                                                        [100%]
73 passed in 0.64s

Comparison tests verify percentage-point arithmetic, immutable validation identity, per-class alignment, finite metrics and common benchmark order. Generated tables/plots are validated against source artifacts. E1 checkpoints are local:

- `runs/E1_yolov8s_uvh26_mv_640_seed42/weights/best.pt`: 22502762 bytes; SHA-256 `9f1045381024445b50e33539791da224b732d61781c73b347013477ebb077fab`.
- `runs/E1_yolov8s_uvh26_mv_640_seed42/weights/last.pt`: 22502762 bytes; SHA-256 `49a460b4c2ce7abc417b5f598ed4cdab83fca0257fdfc87c9caf3bc72a2ca3b5`.

Eligible source, config, summaries and documentation are committed normally on the existing branch; raw/processed images, labels, weights, full runs, secrets and local paths are excluded. See `reports/audit/E1_delivery.json` for the actual commit/push result. Stop boundary: E1 only; no higher-resolution run, augmentation/loss experiment, tracking or counting was started.
