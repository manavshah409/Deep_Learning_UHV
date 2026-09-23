# Accuracy Stage E: frozen matched E1/E3 comparison protocol

Stage E is complete: fresh predictions, common evaluation and threshold calibration used exactly **calibration500** (6,148 GT objects). Reserved1500 images/annotations were not opened and no reserved predictions or performance were produced. Its SHA below is copied from the existing Stage A protocol, not recomputed by opening the manifest. No fusion, model training or Phase 3 work occurred.

## Scope and checkpoints

E1 YOLOv8s epoch22, SHA `9f1045381024445b50e33539791da224b732d61781c73b347013477ebb077fab`; E3 Faster R-CNN ResNet-50-FPN epoch13, SHA `2d035a95206475b1e9939c5686a731a2427f84918389a2aa4d59148ed5ba5df9`. Checkpoints remain unchanged. Historical E1 val2000 and E3 Stage D metrics are not directly compared here. This is a matched calibration experiment, not independent test evidence.

Calibration SHA `a436b78d561516d61ae2f7eedb4a8acb4a58222d2c13df80411ae807f288c502`; mapping SHA `6fd0b458fed2cb174c924d4fa9ead86db0a292dc7c55fd0f4d4b3b9b20d08ea8`; recorded reserved manifest SHA `6fc3e8928e5e90c3390ab298a234f90ad883d413f292dc04f38b2e63dbe5d0e0`.

## Common schema and access guard

`uvh-common-predictions-1` is JSONL: one image envelope carries image ID/key, original width/height, model ID/checkpoint SHA, model preprocessing/inference configuration and actual tensor shape. Its `detections` hold class ID/name, confidence, original-image xyxy, COCO xywh and clipping flag. Shared classes are 0–13 in frozen UVH order; Faster R-CNN foreground1–14 explicitly maps to0–13 and background0 is counted/excluded. No inferred remapping is allowed.

Coordinates: reject non-finite scores/boxes, invalid classes or non-positive **raw** boxes as fatal errors. Map to original coordinates, explicitly reject/count valid positive-area boxes wholly outside the image, then clip intersecting boxes identically for both architectures. Degenerate boxes never reach the evaluator. The first attempt stopped correctly on a YOLO padding-only box clipped to zero height; its incomplete 107-image export, log and source snapshot remain unchanged. The v2 adapter retains pre-clip coordinates to distinguish padding-only boxes from invalid raw predictions. No metrics were published for v1. Rejections in v2: **E1 6, E3 0**. [Recovery evidence](../../reports/comparisons/E1_E3_stageE_v2/coordinate_recovery.json).

The Stage E command accepts only the exact hash-verified calibration manifest, before opening annotations. Image/label paths must resolve to this allowlist before loading, then content hashes are checked. Renamed manifests, unknown image paths and reserved paths fail closed. The shared loader has a separate future `authorize_reserved_evaluation` switch, but **Stage E exposes no CLI flag enabling reserved inference**. Historical reproduction commands are preserved; they are not development entry points for this workflow and must not be used to bypass the new guarded runner.

## Detector-specific processing, common evaluation

Both: MPS float32, batch1, ascending image ID, confidence floor .001, maximum300 detections, no augmentation. E1 retains imgsz640, stride32 rectangular letterbox (padding114), OpenCV BGR→RGB/CHW/255 and class-aware NMS IoU .70. E3 uses PIL RGB/255, ImageNet normalization, aspect resize short480/max640, padding32, RPN NMS .70 with pre/post top1000 and ROI NMS .50. These internal preprocessors/NMS are deliberately recorded as detector-specific, **not identical**. Package versions/source hashes are sealed in [pre-inference plan](../../reports/comparisons/E1_E3_stageE_v2/calibration_plan.json).

One independent COCO-style evaluator uses common annotations/categories, original-pixel areas, IoUs .50:.05:.95, maxDets [1,10,300], COCO area ranges and absent-GT-class exclusion. AP always integrates .001-floor predictions; operating thresholds never filter AP input. P/R/F1 use class-aware greedy descending-score IoU≥.50 matching; equal scores retain prediction order and equal IoU chooses lowest GT index. Macro P/R and macro class F1 exclude absent GT classes; harmonic aggregate F1 is 2×macroP×macroR/(macroP+macroR). The class-agnostic confusion matrix uses the same threshold/IoU but can differ from class-aware counts; rows GT, columns predictions, final background row/column.

## Frozen operating thresholds and measured calibration results

Before inference, grid **[.001,.01,.02,…,.99]** was fixed. Maximize macro class F1, round objectives to12 decimal places for ties, select the **higher confidence**. E1 threshold **0.340**; E3 **0.430**. Complete 100-point curves and threshold seal are preserved; these are calibration-selected operating points and optimistic for generalization.

| Model | Threshold | Macro P | Macro R | Harmonic F1 | Macro class F1 | AP50 | AP50:95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 YOLOv8s | 0.340 | 0.614771 | 0.581658 | 0.597756 | 0.591045 | 0.573778 | 0.482829 |
| E3 Faster R-CNN | 0.430 | 0.536270 | 0.568768 | 0.552041 | 0.546462 | 0.563561 | 0.429070 |

[Threshold seal](../../reports/comparisons/E1_E3_stageE_v2/operating_thresholds.json), [E1 curve](../../reports/comparisons/E1_E3_stageE_v2/E1_threshold_curve.csv), [E3 curve](../../reports/comparisons/E1_E3_stageE_v2/E3_threshold_curve.csv), [E1 per-class P/R/F1/AP/support](../../reports/comparisons/E1_E3_stageE_v2/E1_per_class.csv), [E3 per-class P/R/F1/AP/support](../../reports/comparisons/E1_E3_stageE_v2/E3_per_class.csv). GT and raw/fixed prediction counts, confusion matrices and diagnostic errors are exported for both models. Detector forward/export times in receipts are not matched latency measurements.

## Calibration complementarity

At the model-specific frozen thresholds and same-class IoU≥.50: **both 4358; E1-only 431; E3-only 467; missed by both 892**. Each GT object belongs to exactly one category.

| Class | GT support | Both | E1 only | E3 only | Neither | E1 AP50:95 | E3 AP50:95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hatchback | 610 | 332 | 74 | 102 | 102 | 0.5410 | 0.4717 |
| Sedan | 301 | 125 | 35 | 34 | 107 | 0.4629 | 0.4310 |
| SUV | 236 | 104 | 31 | 27 | 74 | 0.4021 | 0.3753 |
| MUV | 147 | 42 | 22 | 18 | 65 | 0.3714 | 0.3824 |
| Bus | 166 | 103 | 17 | 12 | 34 | 0.6636 | 0.5720 |
| Truck | 253 | 163 | 17 | 21 | 52 | 0.5926 | 0.5306 |
| Three-wheeler | 1035 | 873 | 35 | 43 | 84 | 0.7918 | 0.6995 |
| Two-wheeler | 2920 | 2334 | 160 | 177 | 249 | 0.7045 | 0.6066 |
| LCV | 311 | 212 | 19 | 24 | 56 | 0.6135 | 0.5416 |
| Mini-bus | 19 | 4 | 0 | 2 | 13 | 0.1145 | 0.1702 |
| Tempo-traveller | 30 | 12 | 4 | 3 | 11 | 0.5055 | 0.4577 |
| Bicycle | 62 | 36 | 5 | 4 | 17 | 0.5438 | 0.4346 |
| Van | 48 | 18 | 12 | 0 | 18 | 0.4452 | 0.2972 |
| Others | 10 | 0 | 0 | 0 | 10 | 0.0072 | 0.0365 |

Supported E3 recoveries: Two-wheeler: 177 E3-only / 2920 GT; Hatchback: 102 E3-only / 610 GT; Three-wheeler: 43 E3-only / 1035 GT; Sedan: 34 E3-only / 301 GT; SUV: 27 E3-only / 236 GT. Classes with <50 GT objects are explicitly low support: Mini-bus19, Tempo-traveller30, Van48, Others10. No favorable single rare class is used for the recommendation.

Duplicate candidates E1/E3 **159/201**; localization candidates **162/302**. These geometric diagnostics are not exhaustive mutually exclusive FP causes. On objects both detect, mean IoU E1/E3 **0.9018/0.8733**, and Pearson confidence correlation **0.4549**; correlation is conditional on successful detection, not probability calibration. Across E1 operating predictions, mean maximum same-class E3 overlap IoU **0.7389**; fraction ≥.50 **0.8361**. This overlap is many-to-one, not a one-to-one detection match.

**Decision:** E3 merits a calibration-only fusion feasibility experiment because it uniquely recovers supported-class objects. This is an oracle complementarity signal, not evidence that a practical fusion method improves AP; added false positives and computation may outweigh it. E3-only detections cannot simply be added to E1 without false-positive and duplicate costs. No fusion settings were implemented, selected or tuned here.

## Future comparison contract and closed gate

Frozen [machine-readable protocol](../../configs/accuracy/E1_E3_comparison_v1.json): canonical SHA **`2d15fd4edde67eb07b8ddf5b6e1090204fdda28a2b2b4601f366ea8e95f6c199`**, file SHA **`83e9d019fed1d69b2cb1bd2d3104e5d8d9fab25172fd95a5337e8ea33f7123d1`**. [Artifact hash receipt](../../reports/comparisons/E1_E3_stageE_v2/freeze_receipt.json) records the complete calibration evidence. Prediction SHA E1 **`7389243341d9c945e2307ae99cdfd42435bf8949bbe607546db65eda5f7bdbcc`**, E3 **`1db478fb2bfdab67c3378c3cf02d5b6d6f72809c8231be79649485496c08727c`**; large immutable JSONL files remain ignored under their `runs/` IDs.

The future family is E1/E3/E4. E4 must be developed only on calibration500 and frozen in an immutable amendment referencing this protocol hash, with exact algorithm/code, model hashes, fusion parameters and operating threshold. **Reserved gate stays closed until that amendment and separate user authorization.** If E4 is abandoned, amend the family before opening reserved data. Never choose fusion settings, checkpoint or threshold after reserved results.

Predefined uncertainty: image-level paired bootstrap, seed42, **1,000 replicates**, sample1500 images with replacement, identical draws for all family models, remap duplicate COCO image IDs, recompute pooled AP/F1 with frozen thresholds. Report 95% percentile intervals for E3−E1, E4−E1 and E4−E3; pointwise descriptive intervals, no formal statistical-significance claim. Per-class intervals require ≥50 GT across≥20 images; report unstable represented-class sets and valid-replicate counts. Correlated traffic frames may make image-level uncertainty optimistic.

Predefined timing: sort reserved IDs, seeded42 shuffle, take100 without looking at difficulty; freeze sample/hash before image load. MPSfloat32/batch1,10warm-ups,3timedpasses; rotate model order. Synchronized boundaries cover file decode/normalization/resize/transfer; model+proposal/box decode/internal NMS; then original-coordinate mapping/CPU transfer/operating threshold. Include all stages end-to-end, exclude/report model loading separately. E4 must execute both networks plus fusion, without cached outputs. Report per-image/pass and pooled mean/median/p95, inverse-mean inference-only and still-image throughput, environment/memory limitations. No live-video FPS claim.

Failure rules reject wrong provenance/settings, unauthorized paths, invalid predictions, missing/duplicate images and incomplete bundles; preserve failures and stop on NMS timeout/truncation warnings. Reserved predictions are generated once per frozen family, cached for bootstrap/reporting, never reused for tuning. Any necessary technical rerun requires documented failure and preregistered correction.

## Validation and limitations

[Commands](../reproducibility/E1_E3_STAGE_E.md) and `validation.json`/`tests_summary.json` document validation. All **200 deliverable tests passed**, including16 new comparison contracts; Ruff passed for all seven new Python files. Local prediction/checkpoint/protocol validators passed. Earlier Stage D artifacts and frozen training source remain unchanged. The inference comparison is matched; training budgets were not equal (E1 30 epochs, E3 20). Calibration500 was used for E3 checkpoint selection and now both operating thresholds: these are selection-biased descriptive results. Reserved1500 has historical E0/E1/E2 validation exposure and is not pristine independent test data. No reserved access, fusion implementation, retraining or Phase3 work occurred in Stage E.
