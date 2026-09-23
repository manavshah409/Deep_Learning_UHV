# All applied models and measured performance

Updated after Accuracy Improvement Stage E. **Three detector architectures have been trained/applied: YOLOv8n, YOLOv8s and Faster R-CNN ResNet-50-FPN.** E2 is an inference-resolution experiment using the existing YOLOv8s checkpoint, not a fourth trained architecture. E4 fusion is planned but unimplemented and has no score.

Scores below are percentages; they are detection AP/F1, not classification accuracy. Precision measures how many proposed detections are correct; recall measures how many annotated objects are found. AP50 is average precision at IoU .50; AP50:95 averages AP across .50:.05:.95 and is stricter about box localization. Harmonic aggregate F1 combines macro precision/recall; macro F1 averages class-specific F1 and can differ.

## 1. Main detector training

| Experiment | Architecture and purpose | Training | Selected checkpoint | Input policy |
|---|---|---|---|---|
| E0 | YOLOv8n, compact one-stage baseline | 8,000 images, 30 epochs | Epoch 30 | 640 |
| E1 | YOLOv8s, higher-capacity one-stage detector | 8,000 images, 30 epochs | Epoch 22 | 640 |
| E3 | Faster R-CNN ResNet-50-FPN, two-stage proposal detector | 8,000 images, 20 epochs, unweighted | Epoch 13 | Aspect resize short480/max640 |

The historical E0/E1 pair uses the same 2,000 validation images and matched Ultralytics evaluation. E3 was selected using 500 calibration images. Stage E is the first common calibration evaluator for E1 and E3. Never rank models by mixing historical and current rows.

## 2. Historical E0/E1 and E2 evidence — validation2000

| Experiment | Evaluation set | Precision | Recall | Harmonic F1 | Macro F1 | AP50 | AP50:95 |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 baseline | historical validation2000 | 61.00% | 54.35% | 57.49% | 53.94% | 56.00% | 45.84% |
| E1 baseline | historical validation2000 | 65.03% | 61.03% | 62.97% | 59.19% | 62.33% | 52.48% |
| E2 960 inference | historical validation2000 | 64.02% | 61.02% | 62.49% | 58.77% | 62.64% | 52.51% |

E1 improved historical AP50:95 over E0 by **6.635 percentage points**. E2 retained the exact E1 checkpoint and changed inference resolution from640 to960: AP50:95 improved only **0.038 points**, while the preregistered small-object AP fell **7.076 points**. Gate A failed; no960 training occurred and640 was retained. The E2 640 reference is the E1 historical row above.

## 3. Current matched E1/E3 evidence — calibration500

| Experiment | Evaluation set | Precision | Recall | Harmonic F1 | Macro F1 | AP50 | AP50:95 |
|---|---|---:|---:|---:|---:|---:|---:|
| E1 Stage E | calibration500 | 61.48% | 58.17% | 59.78% | 59.10% | 57.38% | 48.28% |
| E3 Stage E | calibration500 | 53.63% | 56.88% | 55.20% | 54.65% | 56.36% | 42.91% |

Same500 images, same annotations/class order and independent COCO-style evaluator. AP uses confidence floor .001 and max300 detections; fixed metrics use IoU .50. The calibrated operating confidence is **E1 .34 / E3 .43**, selected from the same fixed grid by macro class F1. Internal preprocessing and NMS remain model-specific and documented.

**E1 currently leads the matched calibration comparison:** AP50:95 48.28% versus42.91% (about5.38 percentage points), and macro class F1 59.10% versus54.65%. E3 still detects467 GT objects missed by E1 at the respective operating thresholds, including177 two-wheelers and102 hatchbacks. That supports investigating fusion on calibration only; it does not demonstrate a fusion improvement. Both detect4,358 objects;431 are E1-only and892 missed by both.

## 4. Earlier E3 standalone result — same calibration500, fixed .25 threshold

| Experiment | Evaluation set | Precision | Recall | Harmonic F1 | Macro F1 | AP50 | AP50:95 |
|---|---|---:|---:|---:|---:|---:|---:|
| E3 Stage D | calibration500 | 45.62% | 63.49% | 53.09% | 52.63% | 56.36% | 42.91% |

Stage D AP agrees exactly with Stage E because AP uses the same low-confidence predictions/settings. Stage E's selected threshold changes P/R/F1; this is not a different E3 checkpoint or new training.

## 5. Sampling pilots and engineering checks

| Experiment | Evaluation set | Precision | Recall | Harmonic F1 | Macro F1 | AP50 | AP50:95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Stage B unweighted | pilot calibration250 | 23.91% | 44.26% | 31.05% | — | 37.70% | 23.41% |
| Stage B weighted | pilot calibration250 | 27.57% | 44.85% | 34.14% | — | 37.54% | 23.86% |

These pilots used independent official COCO initializers and only1,000 training/250 calibration images for3epochs. The weighted pilot slightly improved overall AP50:95 but did not improve the supported-minority aggregate AP used by the decision rule; unweighted sampling was selected as the lower-risk full-run configuration. These pilot values must not be compared as full models against the20-epoch E3 result. Pilot harmonic F1 above is computed from the recorded macro P/R; macro class F1 was not exported in that historical summary.

Additional applied checks: the initial YOLOv8n smoke/preflight and the Stage A Faster R-CNN128-train/64-calibration one-epoch preflight validated engineering stability. Stage C0 used tiny synthetic resume checks. They are not research accuracy baselines and are excluded from the main ranking. The historical smoke artifact remains available at `reports/tables/smoke_validation_seed42_metrics.json`; the Stage A preflight records no comparable standalone AP result. Official COCO weights were initializers, not separately evaluated UVH models. Phase3 only exercised detector/video infrastructure; no new trained tracking model is claimed.

## 6. Speed and model size — keep benchmark groups separate

| Benchmark group | Detector | Parameters | Mean / median / p95 end-to-end ms | Still images/s |
|---|---|---:|---|---:|
| Historical matched E0/E1, same100-image sample | YOLOv8n | 3,013,578 | 30.364 /31.167 /33.681 | 32.934 |
| Historical matched E0/E1, same100-image sample | YOLOv8s | 11,141,018 | 30.292 /30.537 /32.063 | 33.012 |
| Stage D calibration100, different sample/session/boundaries | Faster R-CNN | 41,365,786 | 114.329 /112.901 /122.450 | 8.747 |

These are batch-one MPS still-image timings, not live-video FPS. Do not infer a measured E1/E3 speed ratio from separate benchmark groups; the future protocol specifies a matched test. E0/E1 checkpoints are stripped inference weights (~6.23/22.50MB); E3's330.60MB recovery checkpoint also includes optimizer/recovery state, so file-size ratios are not equivalent inference-model size comparisons.

## 7. Status and limits

E0/E1/E2, E3 full training, Stage D closeout and Stage E calibration/protocol freeze are complete. E4 fusion, reserved family evaluation and further training remain unstarted. Reserved1500 has historical validation exposure and is not a pristine independent test set. Calibration scores are selection-biased, single-seed evidence; no production readiness is claimed. The next possible research step is separately authorized calibration-only E4 development, followed by freezing its protocol before reserved evaluation.

Sources: [Stage E protocol/results](phase_reports/E1_E3_COMPARISON_PROTOCOL.md), [historical E1 comparison](phase_reports/PHASE_2_E1_MODEL_COMPARISON.md), [E2 resolution study](phase_reports/PHASE_2_E2_RESOLUTION_ABLATION.md), [Stage D report](phase_reports/E3_STANDALONE_CALIBRATION_CLOSEOUT.md), [Stage B pilots](phase_reports/ACCURACY_IMPROVEMENT_STAGE_B.md), and [machine-readable consolidated scores](../reports/comparisons/E1_E3_stageE_v2/all_model_performance.csv). Each CSV row cites its original measured source.
