# Phase 2 E2: inference-only 640 versus 960 resolution ablation

Status: Gate A measurements and review are complete; threshold not met. Retain E1 at 640. Gate B was not entered. No 30-epoch E2 training is authorized or started. E0/E1 checkpoints and existing result artifacts remain unchanged.

The same E1 YOLOv8s epoch-22 checkpoint is evaluated at 640 and 960 on the frozen 2,000-image validation set. Batch8/MPS/workers0, confidence floor .001, NMS .7/max_det300, evaluator implementation and package versions are controlled. Final corrected outputs use E2_gateA_640_seed42_v2 and E2_gateA_960_seed42_v2.

## Original-area object-size method

Auxiliary pycocotools 2.0.11 COCOeval uses bounding boxes reconstructed from the frozen YOLO validation labels in original-image pixel coordinates. It uses the COCO small/medium/large area ranges 0–1024, 1024–9216, and 9216–1e10 pixels squared, including COCO's exact endpoint rules, IoU .50:.05:.95, 101 recall points and maxDets [1,10,300]. Category IDs exported as 1–14 are explicitly remapped to frozen YOLO IDs 0–13. Ignored GT and out-of-area unmatched predictions follow COCOeval rules. Each area AP is averaged over categories with GT in that area; absent category/size combinations are explicitly not applicable.

This size-specific COCO AP is a separate metric family from Ultralytics overall AP, which has different matching/integration details. Resolution changes never change the original-area bin assignment. Small area is a reproducible proxy, not a measured distance/depth label. Occlusion and density are reviewed qualitatively on the same six diagnostic images; no annotated population occlusion/distance stratification is available.

The initial v1 size exports had a 1-based versus 0-based prediction-category mapping error, detected by the implausible all-area AP check. They are preserved and excluded. Both v2 evaluations rerun with identical corrected source pinned at process start; see reports/audit/E2_superseded_size_exports.json. No library files were patched and no Torch/Ultralytics versions changed.

## Gate and cost controls

Gate A requires small-object AP50:95 +2.0 percentage points, or overall mAP50:95 +1.0 point with clear relevant small/distant class gains, plus valid protocol and cost justification. The benchmark uses the identical checkpoint and deterministic 100-image order, batch1, MPS, 10 warm-ups, conf .25/NMS .7/max_det300 and synchronized stages. Memory values are maxima of sampled current/driver allocations, not absolute hardware peaks.

If Gate A fails, retain E1 at 640 and stop. If it passes, only a short uniquely named 960 training memory preflight and a proposed command may be prepared. E1 physical batch8 had startup accumulation8 with nbs64; physical and optimizer batch are distinct. A preflight must inspect actual optimizer/accumulation behavior, including warm-up. No full E2 run may start without explicit approval.

## Results and decision



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


Detailed machine-readable results: `reports/comparisons/E2_gateA_v2/`. Reproduction: `docs/reproducibility/E2_EVALUATION.md`. Full test result: `reports/audit/E2_pytest.txt`.
