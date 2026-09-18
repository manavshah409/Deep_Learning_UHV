# UVH-26 Vehicle Detection and Traffic Analytics — Project Update and Developer Handover

**Status date: 18 September 2026**  
**Audience:** faculty reviewers, project team and incoming developers  
**Repository:** https://github.com/manavshah409/Deep_Learning_UHV  
**Current preferred research detector:** E1 YOLOv8s, input size 640, best checkpoint from epoch 22

## 1. Faculty overview

The project has progressed from dataset preparation and pipeline checks to two completed detector-training experiments and a measured input-resolution study. We have established a reproducible vehicle-detection baseline, improved it by increasing model capacity, and tested whether increasing inference resolution provides enough additional benefit to justify another training run.

The main result is that YOLOv8s improves validation mAP@0.5:0.95 from **45.84% to 52.48%**, an increase of **6.64 percentage points**, over the original YOLOv8n baseline. Both models completed 30 epochs on the same frozen 8,000-image training subset and were evaluated on the same 2,000-image validation subset.

The subsequent 640-versus-960 inference study did **not** meet its predefined improvement threshold. Overall mAP@0.5:0.95 increased by only **0.038 percentage points**. Some small vehicle categories improved, but the small-object class-average AP declined. We therefore recommend retaining YOLOv8s at 640 and have not started 960-resolution training.

This is a **subset-validation research result**. It is not an independent test-set result, a full-dataset benchmark, a production-ready detector, or a completed real-time tracking/counting system.

## 2. Completion status at a glance

| Workstream | Status | Evidence / qualification |
| --- | --- | --- |
| Full annotation-catalogue audit | Complete | 26,646 image records and 316,220 boxes checked structurally |
| Selected local dataset integrity | Complete | 10,000 selected images, corresponding labels, paths, hashes and leakage checked |
| Acquisition and pixel verification of unselected images | Incomplete | Catalogue audit does not establish image-file integrity |
| Phase 1 / E0 YOLOv8n baseline | Complete and delivered | 30 epochs; best epoch 30; successful process exit |
| Phase 2 / E1 YOLOv8s comparison | Complete and delivered | 30 epochs; best epoch 22; successful process exit; 73 tests recorded |
| E2 Gate A evaluation and benchmarking | Measurements complete | Both corrected resolution evaluations and matched benchmarks exited successfully |
| E2 integrity checks | Passed | Corrected bundles and original E0/E1 preservation verified |
| E2 scientific decision | Threshold not met | Retain E1 at 640 |
| E2 administrative closeout | Pending | Final report/registry/dashboard synchronization, full test suite and Git delivery remain |
| E2 Gate B memory preflight / 30-epoch training | Not started | Gate A failure prevents progression under the current protocol |
| Tracking, counting and sustained video evaluation | Not completed | Future project work, not represented by detector metrics |

**Status reconciliation:** some E2 files still contain intermediate workflow labels such as `awaiting_gate_decision`, `awaiting_qualitative_and_cost_review` and `awaiting_manual_review`. The numerical comparisons and visual review are available, but these machine-readable records have not yet been finalized. This handover explicitly separates completed measurements from unfinished delivery work.

## 3. Project objective and implemented foundation

The broader objective is vehicle detection and traffic analytics for Indian road scenes, including crowded traffic, small vehicles, occlusion and diverse vehicle types. Work completed so far concentrates on the detector and the evidence needed to assess it fairly.

Implemented capabilities include dataset conversion to YOLO format; deterministic subset selection; image/label and class-map verification; immutable experiment naming; training configuration and checkpoint provenance; fresh standalone checkpoint evaluation; per-class metrics and diagnostic visualizations; synchronized Apple MPS benchmarking; artifact-based reporting; and automated validation/tests. The existing dashboard presents saved research artifacts. It should not be confused with a finished live-camera traffic analytics application.

## 4. Dataset and integrity controls

The experiments use UVH-26 **Majority Voting (MV)** annotations. STAPLE annotations are not mixed into the experiments. The source is documented as IISc AIM UVH-26, revision `59f82c57821e8a54dc40bc1f42e83909dbad0b70`, CC BY 4.0.

| Scope | Images | Boxes | Verification scope |
| --- | ---: | ---: | --- |
| Full annotation catalogue | 26,646 | 316,220 | Annotation schema, geometry, classes and split records |
| Frozen local training subset | 8,000 | 94,609 | Image decoding, dimensions, content hashes, labels and paths |
| Frozen local validation subset | 2,000 | 24,342 | Same integrity checks; no train/validation content overlap |

The full catalogue contains 21,349 training and 5,297 validation records. All 14 classes are retained in the selected subset. The largest selected-versus-full class-share deviation is approximately 0.306 percentage points. Semantic manual annotation review covered 42 selected images; it was not an exhaustive semantic relabelling of the dataset.

### Frozen classes and validation support

| YOLO ID | Class | Validation objects |
| --- | --- | --- |
| 0 | Hatchback | 2402 |
| 1 | Sedan | 1169 |
| 2 | SUV | 1028 |
| 3 | MUV | 541 |
| 4 | Bus | 627 |
| 5 | Truck | 933 |
| 6 | Three-wheeler | 4016 |
| 7 | Two-wheeler | 11624 |
| 8 | LCV | 1322 |
| 9 | Mini-bus | 58 |
| 10 | Tempo-traveller | 130 |
| 11 | Bicycle | 278 |
| 12 | Van | 183 |
| 13 | Others | 31 |

Original annotation category IDs 1–14 map in order to YOLO IDs 0–13. Rare categories have limited statistical support, particularly Others and Mini-bus.

### Frozen identifiers

Dataset version: `uvh26_mv_yolo_v1/subsets/baseline_seed42_v2`.

| Record | SHA-256 |
| --- | --- |
| Combined manifest | `990054a93e300a90321db19b3d0bcd98a488a891cd4e2dbd88425f4eb592c2af` |
| Training manifest | `8e4a72413caecc9defee68e75f498b71160b06fb33f30f6cbe2dc8b3677032cf` |
| Validation manifest | `fd23d2e417d70a8614b6312cb3eff1deb0ed98f0532269c22878269fdc5959d3` |
| Class mapping | `6fd0b458fed2cb174c924d4fa9ead86db0a292dc7c55fd0f4d4b3b9b20d08ea8` |

Three problematic source files were quarantined during preparation, and two original subset candidates were replaced before baseline training. The final replacements preserved class counts. There was no silent coordinate repair or relabelling to improve results. The superseded subset and recovery evidence remain recorded in the Phase 1 report.

## 5. Phase 1: proper YOLOv8n baseline (E0)

Run: `yolov8n_uvh26_mv_baseline_seed42_v1`.

- Completed **30/30 epochs**, best checkpoint **epoch 30**, process exit **0**, no early stopping.
- Recorded training duration: **16,029.859 seconds**, approximately **4.45 hours**.
- Configuration: COCO-pretrained YOLOv8n, image size 640, physical batch 8, seed 42, explicit AdamW, learning rate 0.000556, weight decay 0.0005, patience 10 and warm-up 3 epochs.
- Effective MPS behavior: AMP disabled and workers 0. Startup accumulation was 8 with `nbs=64`; accumulation can vary during warm-up.
- Fresh standalone validation and synchronized checkpoint timing were completed. Metrics were not simply copied from the last training epoch.

The smoke and one-epoch preflight runs were engineering checks only. The completed 30-epoch baseline supersedes them as the scientific reference.

## 6. Phase 2 E1: increasing model capacity

Run: `E1_yolov8s_uvh26_mv_640_seed42`.

E1 asks whether YOLOv8s improves detection over YOLOv8n under the same frozen subset and comparable training controls. Both use image size 640, seed 42, a 30-epoch budget and the explicit AdamW schedule. The intended experimental change is model capacity and its corresponding official pretrained initializer.

- Completed **30/30 epochs**, best checkpoint **epoch 22**, process exit **0**, no early stopping.
- Recorded training timer: **28,459.257 seconds**, approximately **7.91 hours**.
- Calendar timestamp span: **40.53 hours**, affected by documented Mac sleep cycles. This must not be presented as uninterrupted training compute time.
- Evaluation-model parameters: E0 **3,013,578**, E1 **11,141,018**.
- Checkpoint sizes: E0 **6,228,714 bytes**, E1 **22,502,762 bytes**.

### Standalone E0 versus E1 accuracy

| Metric | E0 YOLOv8n at 640 | E1 YOLOv8s at 640 | Change, percentage points |
| --- | ---: | ---: | ---: |
| Precision | 60.9993% | 65.0335% | +4.034 |
| Recall | 54.3547% | 61.0255% | +6.671 |
| Harmonic aggregate F1 | 57.4856% | 62.9658% | +5.480 |
| Macro class F1 | 53.9397% | 59.1856% | +5.246 |
| mAP@0.5 | 56.0049% | 62.3274% | +6.322 |
| mAP@0.5:0.95 | 45.8407% | 52.4760% | +6.635 |

All 14 class AP50:95 values improved in E1 versus E0. The largest gains were Van (+13.81 pp), Bicycle (+12.22 pp), Tempo-traveller (+11.55 pp) and Sedan (+10.32 pp). Three-wheeler was the strongest E1 class; Others and Mini-bus remained weakest. Others recall was still zero at the selected operating point.

**Metric definitions:** precision and recall are macro class means at each model's confidence maximizing smoothed mean class F1. Harmonic aggregate F1 is `2PR/(P+R)`. Macro F1 is the mean of class-specific F1 scores. Neither is micro-F1. AP50:95 averages IoU thresholds from 0.50 to 0.95 in steps of 0.05. An Others precision value of 1.0 at zero recall reflects an evaluator interpolation convention, not perfect detection.

E1's original evaluation export failed because of an incompatible validator-access assumption. A supported validator hook and atomic output publication repaired the export. Both E0 and E1 were reevaluated under the same corrected implementation; E0 reproduced its baseline metrics. No retraining was needed for this recovery.

## 7. Phase 2 E2: 640-versus-960 inference study

### Question and controls

Gate A tests the **same already-trained E1 epoch-22 checkpoint** at two inference resolutions. It does not test a model trained at 960. Both runs use the same frozen 2,000 validation images and order, mapping, checkpoint, MPS device, batch 8, workers 0, confidence floor 0.001, NMS IoU 0.7, max_det 300, corrected evaluator and package versions.

Corrected immutable evaluation IDs:

- `E2_gateA_640_seed42_v2`
- `E2_gateA_960_seed42_v2`

### Overall measured results

| Metric | 640 (%) | 960 (%) | Change (pp) |
| --- | --- | --- | --- |
| precision | 65.0335 | 64.0242 | -1.0094 |
| recall | 61.0255 | 61.0244 | -0.0011 |
| f1 | 62.9658 | 62.4883 | -0.4775 |
| macro_f1 | 59.1856 | 58.7663 | -0.4192 |
| map50 | 62.3274 | 62.6411 | +0.3138 |
| map50_95 | 52.4760 | 52.5139 | +0.0379 |

F1 operating confidences were approximately 0.3143 at 640 and 0.3584 at 960. The common AP confidence floor remains 0.001. Predictions exported at that low threshold numbered **271,658** and **253,260**, respectively; these are not counts of confidently detected real-world vehicles.

Recorded YOLO validation-stage durations were **150.647 seconds** at 640 and **169.885 seconds** at 960. These durations include the validator's JSON evaluation but exclude the subsequent supplementary object-size analysis; they are not total CLI wall times or live-video latency.

### Object-size analysis

Boxes are measured in **original-image pixels**, so changing input resolution does not change their size-bin membership. Supplementary pycocotools COCOeval uses small area up to 32² pixels, medium 32²–96² and large 96²–1e10, standard endpoint behavior, IoU 0.50:0.05:0.95, 101 recall points and maxDets [1,10,300]. AP averages categories with ground truth in the relevant area bin; absent category/size combinations are not applicable, not zero.

This COCO-area AP is a distinct metric family from the Ultralytics overall AP above. Small pixel area is a proxy for small appearance, not an annotated physical distance measurement.

| Area group | GT objects | AP50:95 at 640 (%) | AP50:95 at 960 (%) | Change (pp) |
| --- | --- | --- | --- | --- |
| all | 24342 | 52.438 | 52.484 | +0.045 |
| small | 338 | 27.347 | 20.271 | -7.076 |
| medium | 8577 | 31.889 | 37.077 | +5.188 |
| large | 15427 | 57.423 | 56.765 | -0.658 |

The small-object result needs careful interpretation. There are only **338 small objects**, including **297 two-wheelers** and **28 three-wheelers**. Small two-wheeler AP50:95 improves from **38.54% to 46.07%** (+7.53 pp); small three-wheeler AP improves from **28.30% to 41.37%** (+13.07 pp). However, the macro average gives equal weight to each of the eight represented categories. Sparse categories, including a single small SUV whose AP falls from 60% to 0%, drive the aggregate decline. Small Bicycle gains are based on only three objects. This is evidence of category-specific gains and unstable rare-bin estimates, not uniform small-object deterioration or improvement.

### Per-class E2 AP50:95

| Class | GT objects (all sizes) | 640 (%) | 960 (%) | Change (pp) |
| --- | --- | --- | --- | --- |
| Hatchback | 2402 | 59.712 | 60.234 | +0.521 |
| Sedan | 1169 | 59.158 | 59.680 | +0.522 |
| SUV | 1028 | 46.075 | 47.716 | +1.640 |
| MUV | 541 | 43.047 | 42.947 | -0.100 |
| Bus | 627 | 70.437 | 69.412 | -1.025 |
| Truck | 933 | 57.877 | 54.729 | -3.149 |
| Three-wheeler | 4016 | 78.138 | 79.328 | +1.190 |
| Two-wheeler | 11624 | 69.254 | 70.417 | +1.163 |
| LCV | 1322 | 61.985 | 62.175 | +0.190 |
| Mini-bus | 58 | 17.363 | 17.187 | -0.176 |
| Tempo-traveller | 130 | 68.496 | 67.020 | -1.476 |
| Bicycle | 278 | 54.414 | 54.703 | +0.289 |
| Van | 183 | 45.647 | 45.949 | +0.302 |
| Others | 31 | 3.060 | 3.698 | +0.638 |

Three-wheeler (including the auto-rickshaw category) and Two-wheeler improve overall AP by about 1.19 and 1.16 points. Truck drops 3.15 points; Tempo-traveller drops 1.48 points. Van AP rises slightly while precision and recall fall. Mini-bus recall falls from 31.03% to 24.14%, and Others recall remains zero. Full P/R/F1/AP50/AP50:95 values are available in the per-class CSV, rather than being inferred from AP alone.

### Qualitative comparison with ground truth

The same six diagnostic scenes were inspected at both resolutions, using confidence 0.10 and class-agnostic greedy IoU 0.5 matching. These selected examples are not unbiased population error estimates.

| Validation image | Observed finding |
| --- | --- |
| 4232 | Both correctly match all 10 GT objects; unmatched predictions increase from 4 to 5. A spurious vehicle box on a pedestrian persists. |
| 10518 | Both have 16 correct-class matches of 17 GT. The distant two-wheeler is already detected at 640; 960 adds unmatched/duplicate boxes rather than recovering additional GT. LCV/Truck confusion remains. |
| 1364 | Dense junction: correct matches fall 50→49 of 56 GT; unmatched GT rises 2→3. Dense overlap and subtype confusion remain. |
| 21621 | Dense, occluded queue: correct matches fall 31→30 of 40 GT; unmatched predictions rise 32→52. The Others handcart remains misclassified. |
| 4711 | Correct matches rise 7→8 of 9 GT, but Mini-bus/Bus confusion remains and unmatched predictions rise 7→10. |
| 954 | Both correctly match 8 of 9 GT; the construction vehicle labelled Others is still predicted as Truck. |

Annotation limitations are recorded separately from model failures: apparent missing boundary vehicles, fine-grained subtype ambiguity, uncertain occluded-object extents and source redactions. The frozen ground truth has not been changed after observing predictions. No population occlusion-stratified AP is available.

### Recovery and integrity

An initial supplementary size-analysis export did not invert the validator's 1-based JSON category IDs to the frozen 0-based YOLO mapping. The implausible AP values exposed this issue. The adapter was corrected and a synthetic regression test added. Both resolutions were rerun under new v2 IDs using identical corrected source captured at process start. Original v1 evidence is preserved locally and excluded from the scientific comparison.

The corrected bundles pass integrity checks covering hashes, counts, class mapping, benchmark image order and preservation of E0/E1 artifacts. No installed Ultralytics or Torch code was patched.

## 8. E2 latency and memory measurements

Protocol: same checkpoint and deterministic 100-image sample/order, batch 1, MPS float32, 10 warm-up iterations, confidence 0.25, NMS IoU 0.7 and max_det 300. Timing is synchronized around stages. Preprocessing includes rectangular letterboxing, BGR-to-RGB conversion and normalized CHW tensors; per-image actual tensor shapes are recorded.

| Stage | Statistic | 640 (ms) | 960 (ms) |
| --- | --- | --- | --- |
| preprocess_ms | mean_ms | 1.939 | 2.244 |
| preprocess_ms | median_ms | 1.922 | 2.256 |
| preprocess_ms | p95_ms | 2.507 | 2.654 |
| inference_ms | mean_ms | 17.304 | 18.057 |
| inference_ms | median_ms | 17.693 | 17.543 |
| inference_ms | p95_ms | 20.660 | 21.568 |
| postprocess_ms | mean_ms | 2.623 | 2.124 |
| postprocess_ms | median_ms | 2.692 | 1.797 |
| postprocess_ms | p95_ms | 3.630 | 3.256 |
| end_to_end_ms | mean_ms | 69.094 | 65.272 |
| end_to_end_ms | median_ms | 70.032 | 64.932 |
| end_to_end_ms | p95_ms | 75.926 | 72.184 |

| Throughput definition | 640 | 960 |
| --- | ---: | ---: |
| Inference-only images/s | 57.789 | 55.381 |
| End-to-end still images/s | 14.473 | 15.321 |

FPS is sample count divided by total elapsed stage time, equivalent to 1000/mean milliseconds. End-to-end time wraps `predict(path)`, including local read/decode, preprocessing, inference, postprocessing, API overhead and synchronization. Model load, warm-up, video capture, drawing and display are excluded.

Mean inference is approximately **4.35% slower at 960**, while measured total latency is lower. A single sequential timing pass can be affected by caching, background load and thermal conditions; this does not prove 960 is faster. Earlier E1 comparison timings were approximately 30 ms end-to-end in a different measurement session. Do not mix those historical numbers with this 69/65 ms E2 session to claim a model or resolution speedup. Neither session establishes sustained live-video FPS.

### Sampled memory, not absolute peak

| Context | Resolution | Max sampled allocated bytes | Max sampled driver bytes |
| --- | --- | --- | --- |
| Validation batch 8 | 640 | 69803008 | 1209729024 |
| Validation batch 8 | 960 | 100288000 | 1142620160 |
| Benchmark batch 1 | 640 | 44587520 | 144375808 |
| Benchmark batch 1 | 960 | 44655872 | 177930240 |

Memory was sampled after each validation batch or timed image. These are maxima of observed MPS allocations, not true hardware peak-memory measurements. Driver caching can make the 640 validation driver sample larger than 960; it does not show that 960 requires less peak memory. Successful batch-8 validation does not establish a stable batch-8 training configuration.

## 9. Gate A decision and training boundary

The predefined gate requires either **+2.0 percentage points small-object AP50:95**, or **+1.0 percentage point overall mAP50:95 with clear relevant class gains**, together with protocol integrity and acceptable cost.

Observed changes are **−7.076 pp small-object macro AP** and **+0.038 pp overall mAP**. Neither threshold is met. Relevant class gains are documented, but substituting a favorable class metric after observing results would change the decision rule.

**Recommendation: retain E1 YOLOv8s at 640.** A 960 training run is not justified under this gate. This does not prove that a separately trained 960 model could never help; it means this protocol does not authorize that compute expenditure on the available evidence.

Gate B was not entered. Therefore no stable 960 training batch, verified optimizer accumulation, training-time estimate or proposed training command has been established. **No E2 30-epoch training job has started.**

For future experiment design, E1's physical batch was 8 and recorded startup accumulation was 8, implying a nominal optimizer batch of 64 outside warm-up. Physical batch 8 must not be described as effective optimizer batch 8. Actual accumulation and optimizer-step behavior would require verification in any future training preflight.

## 10. Developer handover

### Environment and architecture

Recorded measurement environment: Apple M5 with 24 GiB memory; macOS 26.6.2 arm64; Python 3.12.14; PyTorch 2.14.0; Ultralytics 8.4.146. E2 supplements evaluation with pycocotools 2.0.11, pinned in `requirements-e2-evaluation.txt`. MPS nondeterministic operations limit bitwise numerical replay despite a fixed seed.

| Location | Purpose |
| --- | --- |
| `configs/` | Frozen experiment configurations and registry |
| `src/data/` | Conversion, dataset validation and provenance utilities |
| `src/evaluation/evaluate_e2.py` | Resolution-specific standalone validation and complete output bundles |
| `src/evaluation/e2_size_ap.py` | Original-pixel-area COCO evaluation and category mapping |
| `src/evaluation/benchmark_e2.py` | Matched synchronized MPS timing and sampled memory |
| `src/evaluation/compare_e2.py` | Protocol equality checks and comparison tables |
| `src/evaluation/paired_e2_predictions.py` | Paired GT/640/960 diagnostic exports |
| `scripts/validate_e2_artifacts.py` | Bundle consistency and E0/E1 preservation checks |
| `scripts/render_e2_comparison.py` | Scientific comparison chart |
| `tests/test_e2_resolution.py` | E2 mapping, bins, protocol, ID and publication checks |
| `reports/evaluations/` | Complete summarized evaluation bundles and figures |
| `reports/comparisons/` | Derived comparisons and diagnostic summaries |
| `app.py`, `components/` | Artifact-based Streamlit dashboard |
| `runs/`, `data/processed/` | Local-only execution and dataset files; excluded from Git |

### Checkpoints to preserve

| Checkpoint | Bytes | SHA-256 |
| --- | ---: | --- |
| E0 best | 6,228,714 | `85b9e089ec3dfaa676e30a6191b7e5726f320d5bf85391f1675922b88f9778b3` |
| E0 last | 6,228,714 | `19cf82c3848fafae370fa36dd16e845a26dc38eb83bd80a473bb24691268551a` |
| E1 best | 22,502,762 | `9f1045381024445b50e33539791da224b732d61781c73b347013477ebb077fab` |
| E1 last | 22,502,762 | `49a460b4c2ce7abc417b5f598ed4cdab83fca0257fdfc87c9caf3bc72a2ca3b5` |

Project-relative checkpoint paths are `runs/yolov8n_uvh26_mv_baseline_seed42_v1/weights/{best,last}.pt` and `runs/E1_yolov8s_uvh26_mv_640_seed42/weights/{best,last}.pt`. Weights are local and intentionally absent from Git. A repository clone alone cannot reproduce evaluations without the frozen dataset and verified checkpoints.

### Tests and Git status

The last completed, recorded full test suite is **73 passed** at E1 closeout. Five E2 tests have been added, but the full suite has not yet been executed for E2; do not report a new full-suite pass count. The separate E2 artifact validator reports `passed` and confirms that E0/E1 artifacts remain unchanged.

Current branch is `master`; current HEAD is `50462e39553b20b126900f78bdfcb0663e0433fd` (`Record successful E1 comparison delivery`). E1 results commit is `f2f51a464a4971167f62d16881e515e545c63e1d`; E1 delivery records a successful push to origin/master. E2 source/results are currently local and uncommitted. This handover is a new local document, not evidence that E2 has been pushed.

Do not commit datasets, generated labels, checkpoints, complete run directories, caches, raw prediction imagery, secrets or machine-specific paths. Existing measured bundles and run IDs must not be overwritten.

### Useful review commands

Run from the repository root in the existing environment. These commands review or validate existing work; they do not launch training. Full tests and data validation below are recommended next checks, not claimed completed E2 checks.

```bash
# Review repository state and actual branch/remote.
git status --short
git branch --show-current
git remote -v

# Validate the current E2 evidence and preservation snapshot.
.venv/bin/python scripts/validate_e2_artifacts.py

# Run the complete suite before E2 delivery.
.venv/bin/python -m pytest -q

# Recheck the frozen selected data if needed.
.venv/bin/python -m src.data.validate_yolo --dataset-version uvh26_mv_yolo_v1/subsets/baseline_seed42_v2

# Open the existing artifact dashboard (E2 integration still pending).
.venv/bin/python -m streamlit run app.py
```

Do not rerun the one-time Gate A supervisor or evaluation commands with their existing IDs. Any deliberate repeat must have a fresh resolution-specific ID and the same documented controls. No training command is supplied because the gate did not permit progression.

## 11. Deliverables and evidence index

Links below are repository-relative so the document remains portable with the project.

- [Phase 1 baseline closeout](phase_reports/PHASE_1_BASELINE.md): data recovery, E0 training, evaluation, timing and subset gate.
- [E1 model comparison](phase_reports/PHASE_2_E1_MODEL_COMPARISON.md): completed E0/E1 comparison and decision.
- [E1 evaluation recovery](phase_reports/E1_EVALUATION_RECOVERY.md): failure diagnosis and recovery method.
- [E2 protocol/report draft](phase_reports/PHASE_2_E2_RESOLUTION_ABLATION.md): currently requires final status and result integration.
- [E2 overall comparison](../reports/comparisons/E2_gateA_v2/overall.csv), [per-class comparison](../reports/comparisons/E2_gateA_v2/per_class.csv), [size AP](../reports/comparisons/E2_gateA_v2/size_ap.csv), [latency](../reports/comparisons/E2_gateA_v2/latency.csv).
- [E2 comparison chart](../reports/comparisons/E2_gateA_v2/area_accuracy_latency.png).
- [640 metrics](../reports/evaluations/E2_gateA_640_seed42_v2/metrics.json), [960 metrics](../reports/evaluations/E2_gateA_960_seed42_v2/metrics.json): complete settings and numerical provenance. Their directories contain confusion matrices, curves, size summaries and completeness hashes.
- [E2 artifact validation](../reports/audit/E2_artifact_validation.json), [execution exits](../reports/audit/E2_gateA_execution.json), [superseded-export audit](../reports/audit/E2_superseded_size_exports.json).
- [Faculty presentation script](faculty_review/PRESENTATION_SCRIPT.md) and [faculty start guide](faculty_review/START_HERE.md): historical Phase 1 faculty materials.

The existing faculty PDF, portable project ZIP and presentation script represent the Phase 1 deliverable. They have not been regenerated to incorporate this E2 update. Use this document for the current status instead of presenting the older faculty pack as an E2-complete package.

## 12. Remaining work and recommended next steps

1. Finalize E2's recorded gate decision and qualitative review status; retain the numerical failure and class-specific nuance.
2. Synchronize the E2 report, experiment registry, README, CHANGELOG, reproduction instructions and artifact dashboard.
3. Run the full test suite and final artifact checks; review eligible changes; commit and push normally to origin/master.
4. Refresh faculty materials if a single consolidated E0/E1/E2 presentation pack is required.
5. With a separately agreed scope, progress the traffic-analytics application around the retained E1 detector: define tracking/counting requirements, video sources, ground-truth evaluation and an end-to-end latency criterion before implementation.
6. If small-vehicle improvement is revisited, preregister a better-supported target analysis and cost budget. Do not retroactively change Gate A to pass or launch 960 training automatically.

Remaining research limitations include a single seed, validation-based checkpoint selection, no independent test-set generalization estimate, severe class imbalance, sparse size/class bins, incomplete unselected-image acquisition, source-annotation uncertainty, qualitative review on selected scenes and non-laboratory sequential timing. No production or real-time certification is claimed.

## 13. Short faculty presentation script

> Our project now has a verified dataset pipeline and two completed vehicle-detector training experiments. We audited the full annotation catalogue and verified a frozen local subset of 8,000 training and 2,000 validation images across 14 classes.
>
> We first trained YOLOv8n for 30 epochs to establish a proper baseline. We then trained YOLOv8s under comparable settings. YOLOv8s increased mAP at IoU 0.50–0.95 from 45.84% to 52.48%, so it is our preferred research detector. Its best checkpoint was produced at epoch 22.
>
> Next, we tested the same YOLOv8s checkpoint at input resolutions 640 and 960. This was an inference experiment, not another training run. Higher resolution helped small two-wheelers and three-wheelers, but overall accuracy increased by only 0.038 percentage points, and the small-object class-average metric declined. Therefore the experiment did not meet our predefined gate for spending time on 960-resolution training.
>
> We also measured latency on Apple MPS, checked predictions against ground truth, preserved checkpoint and dataset hashes, and recorded the results in reproducible artifacts. Current limitations include rare-class errors, dense-scene confusion and the absence of an independent test or sustained video benchmark. We are retaining YOLOv8s at 640. The immediate remaining work is E2 documentation, testing and Git closeout, followed by a separately scoped tracking and counting stage. No E2 30-epoch run has started.
