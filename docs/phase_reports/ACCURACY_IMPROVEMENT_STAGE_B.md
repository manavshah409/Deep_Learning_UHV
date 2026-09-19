# Accuracy Improvement Stage B: paired sampling pilot

**Complete: two independent three-epoch MPS pilots. Provisional full-run sampling choice: unweighted. Full E3, E4 fusion and reserved-split evaluation remain unstarted.**

These measurements select a training configuration. They are not final detector results, a test-set evaluation, or evidence of superiority over E1. E1 remains the selected research detector.

## Shared configuration and provenance

- Train: deterministic 1,000-image subset of frozen 8,000; evaluation: 250 images drawn exclusively from Stage A calibration500. Seed 42 and all 14 classes represented in both. Rare-first class-covering anchors are followed by seeded-shuffle fill; this is not an unbiased random benchmark.
- Both runs independently initialize official COCO_V1 Faster R-CNN ResNet-50-FPN, followed by the same seed-42 15-output head. Neither starts from Stage A or from the other pilot. Physical batch 1, workers 0, MPS, no additional augmentation; aspect-preserving min480/max640 resize.
- Shared SGD: base LR .001, momentum .9, weight decay .0005; default pretrained backbone freezing retained. Sampling is the only configuration difference. B1 visits every selected training image once per epoch in a deterministic shuffled order. B2 uses Stage A image weights (mean capped inverse-square-root class weights) with replacement, exactly 1,000 draws and at most three repetitions per image per epoch.
- COCO bbox AP at IoU .50:.95, maxDet300, score floor .001 and NMS .5. AP is macro-averaged over represented classes. P/R uses confidence .25 and IoU .5, macro-averaged over represented GT classes. It differs from YOLO operating-point metrics.
- Identical initialized model state SHA-256: `d1e6d7c12c066be9f3be743461926f63874bc000ed2e9cbd6961eb84e306242f`.
- Initializer SHA-256: `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`. Frozen parent, mapping and selected-manifest hashes are in `reports/accuracy_stage_b/protocol.json`; full effective configurations and source hashes remain with each run.
- Reserved1500 manifest and images were not opened. The full 26,646-image annotation-catalogue audit remains distinct from the 10,000-image local subset integrity audit. Acquisition/integrity checks for unselected images remain incomplete. Historical validation reuse means the reserved pool is not an unseen test set.

## Learning rate fixed before either pilot

Stage A established finite loss at .005 for only 128 updates; it did not establish an optimal batch-one LR. A fivefold reduction to .001 with a two-epoch warm-up is a conservative safety choice. No optimizer or schedule was selected using the reserved split. The later decays remain a proposal, not something three epochs validated.

`LR(e) = .001 × min(1,e/2) × .1**(I[e>=13] + I[e>=18])`, with one-based epoch e.

| Epoch | LR |
|---:|---:|
| 1 | 0.00050 |
| 2 | 0.00100 |
| 3 | 0.00100 |
| 4 | 0.00100 |
| 5 | 0.00100 |
| 6 | 0.00100 |
| 7 | 0.00100 |
| 8 | 0.00100 |
| 9 | 0.00100 |
| 10 | 0.00100 |
| 11 | 0.00100 |
| 12 | 0.00100 |
| 13 | 0.00010 |
| 14 | 0.00010 |
| 15 | 0.00010 |
| 16 | 0.00010 |
| 17 | 0.00010 |
| 18 | 0.00001 |
| 19 | 0.00001 |
| 20 | 0.00001 |

## Measured epoch results

Times include synchronized MPS training; total additionally includes sampling export, loss export, checkpoint save/reload/hash verification and logging. Train timing includes image decoding, integrity checking and optimizer work; validation includes decoding and COCO metric calculation. These are training durations, not inference FPS.

| Pilot | Epoch | LR | Train s | Val s | Total s | Cumulative s | Mean loss | AP50 | AP50:95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unweighted | 1 | 0.0005 | 325.65 | 44.75 | 371.87 | 371.87 | 1.0941 | 0.3033 | 0.1802 |
| unweighted | 2 | 0.0010 | 317.43 | 41.27 | 360.06 | 731.94 | 0.7997 | 0.3425 | 0.2134 |
| unweighted | 3 | 0.0010 | 326.02 | 46.49 | 373.99 | 1105.93 | 0.6595 | 0.3770 | 0.2341 |
| weighted | 1 | 0.0005 | 308.87 | 41.69 | 352.00 | 352.00 | 1.0747 | 0.3047 | 0.1755 |
| weighted | 2 | 0.0010 | 306.98 | 41.42 | 349.88 | 701.89 | 0.7805 | 0.3313 | 0.2032 |
| weighted | 3 | 0.0010 | 301.51 | 41.32 | 344.27 | 1046.16 | 0.6398 | 0.3754 | 0.2386 |

All 6,000 training-step losses and gradients were finite. Component losses and macro P/R are preserved for every epoch in `epoch_comparison.csv`; per-step losses remain in each ignored run, and extrema/p95/p99 are summarized in `loss_diagnostics.json`. Endpoint loss stability does not establish long-run convergence.

## Sampling coverage

| Pilot | Epoch | Draws | Unique | Coverage % | Max repetitions | Histogram (repetitions: images) |
|---|---:|---:|---:|---:|---:|---|
| unweighted | 1 | 1000 | 1000 | 100.0 | 1 | {'1': 1000} |
| unweighted | 2 | 1000 | 1000 | 100.0 | 1 | {'1': 1000} |
| unweighted | 3 | 1000 | 1000 | 100.0 | 1 | {'1': 1000} |
| weighted | 1 | 1000 | 618 | 61.8 | 3 | {'0': 382, '1': 327, '2': 200, '3': 91} |
| weighted | 2 | 1000 | 625 | 62.5 | 3 | {'0': 375, '1': 350, '2': 175, '3': 100} |
| weighted | 3 | 1000 | 620 | 62.0 | 3 | {'0': 380, '1': 335, '2': 190, '3': 95} |

Zero-repeat images are included in weighted histograms. Effective per-class object and image exposures for each epoch are in `sampling_comparison.json`. These count repeat presentations, not new independent annotations.

## Per-class epoch-three calibration AP and exposure

| Class | Calibration objects | B1 AP50 | B2 AP50 | B1 AP50:95 | B2 AP50:95 | B1 objects drawn (3 epochs) | B2 objects drawn (3 epochs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hatchback | 305 | 0.4448 | 0.4510 | 0.2956 | 0.2932 | 3297 | 3428 |
| Sedan | 154 | 0.2651 | 0.2850 | 0.1819 | 0.1927 | 1755 | 1894 |
| SUV | 118 | 0.2565 | 0.2492 | 0.1769 | 0.1694 | 1419 | 1554 |
| MUV | 67 | 0.1526 | 0.1554 | 0.1154 | 0.1162 | 684 | 756 |
| Bus | 80 | 0.5541 | 0.5257 | 0.3551 | 0.3626 | 1038 | 1058 |
| Truck | 122 | 0.4805 | 0.4399 | 0.2878 | 0.2619 | 1266 | 1365 |
| Three-wheeler | 484 | 0.8172 | 0.8359 | 0.4521 | 0.5648 | 5586 | 5743 |
| Two-wheeler | 1538 | 0.7970 | 0.7935 | 0.4424 | 0.4652 | 16713 | 16621 |
| LCV | 180 | 0.5743 | 0.5893 | 0.3667 | 0.3487 | 1845 | 2039 |
| Mini-bus | 9 | 0.0406 | 0.0371 | 0.0253 | 0.0209 | 93 | 103 |
| Tempo-traveller | 16 | 0.3515 | 0.4148 | 0.2413 | 0.3031 | 183 | 229 |
| Bicycle | 28 | 0.4201 | 0.4031 | 0.2668 | 0.1979 | 387 | 404 |
| Van | 23 | 0.0949 | 0.0491 | 0.0660 | 0.0330 | 297 | 326 |
| Others | 8 | 0.0284 | 0.0265 | 0.0040 | 0.0107 | 54 | 58 |

Every epoch’s per-class precision, recall, AP50 and AP50:95 is in `per_class_metrics.csv`. Mini-bus (9 objects) and Others (8 objects) are reported but excluded from the supported-minority AP decision. Minority membership was fixed from the five lowest full-training class counts: Others, Mini-bus, Tempo-traveller, Van and Bicycle.

## Predeclared decision

Weighted sampling requires finite, stable endpoint losses, no AP50 or AP50:95 regression exceeding 1.0 percentage point, and better aggregate minority exposure or supported minority macro AP. An exposure-only benefit also requires non-regressing supported minority macro AP. Otherwise use unweighted as the lower-coverage-risk choice. This rule and support threshold were frozen before either pilot.

- Epoch-three B2−B1 AP50: **-0.157 pp**; AP50:95: **+0.449 pp**.
- Supported-minority macro AP50:95: B1 **0.1914**, B2 **0.1780** (Tempo-traveller, Van, Bicycle collectively).
- Three-epoch aggregate minority object exposure: B1 **1014**, B2 **1120**.
- Recommendation: **unweighted**. Provisional configuration selection from one seed and three epochs; no statistical superiority claim. Unsupported rare classes are excluded from AP selection. Unweighted is the lower-coverage-risk default when the weighted rule does not pass.
- Both pilots are evaluated at the same final epoch; no favorable-epoch or single-class cherry-picking. Three epochs are insufficient to establish a reliable final-accuracy advantage or optimal optimizer. No full training is authorized by this result alone.

## Checkpoint and process integrity

Both actual child processes exited zero and each completed exactly three epochs. All six epoch checkpoints passed SHA-256, size, safe CPU deserialization, finite parameter/optimizer state and epoch/LR metadata checks. All logs and checkpoints remain unchanged locally.

| Pilot | Epoch | Bytes | SHA-256 |
|---|---:|---:|---|
| unweighted | 1 | 330580683 | `a21b2757c74315592f310be9af6a67001007b190ea7f1c62bfbc8ce33ed70a93` |
| unweighted | 2 | 330580683 | `295ab9760c68735d7a0194e4841d37a1bbd99d385ce61d4cf2f15af990636b02` |
| unweighted | 3 | 330580683 | `a3951d74c6b50fc2b60193c472b3b611b51bbcb64f01548387f74cef2da560f0` |
| weighted | 1 | 330580683 | `986b51aefd563438ce8cecb4807473301f0e38532b77864cf90b2a68615cdd5e` |
| weighted | 2 | 330580683 | `b2114ba49513242565e9e62cc2642d78780bb8e05330f502e6aa5e61c903d6bb` |
| weighted | 3 | 330580683 | `bd1f3703dd21c43b7209b11530769ca6581b1464ae4c6cda0e2119842bd1f407` |

Local relative checkpoint paths are recorded in `checkpoint_integrity.json`, under `runs/E3_stageB_{unweighted,weighted}_3ep_seed42_v1/epoch_00{1,2,3}.pth`. Actual exit records are in `process_status.json`. Full logs are the corresponding `.log` files directly under `runs/`.

Warnings: none found in the captured pilot logs.

## Full 20-epoch proposal — not executed

- unweighted: **14.86 hours**, observed-epoch-scaled planning range **14.57–15.01 hours**.
- weighted: **14.06 hours**, observed-epoch-scaled planning range **13.87–14.20 hours**.

20 * mean(8*1000-image training + 2*250-image calibration + checkpoint/log overhead); planning estimate, not measured full training; no early stopping assumed. Thermal load, contention and sampling distribution can change this estimate; the range is not a confidence interval.

Proposed command, only after separate full-run authorization (8,000 training / 500 calibration images):

```bash
caffeinate -i .venv/bin/python -m src.experiments.stage_b --sampling unweighted --run-id E3_fasterrcnn_unweighted_20ep_seed42_v1 --allow-full-training
```

## Validation and delivery

The Stage B validator passed: all 1,250 selected image/label pairs, parent/mapping/protocol hashes, dataset membership, disjointness, replayed draw/exposure reports, identical initializer/model state, paired configurations, process exits and six checkpoints. 356 historical tracked files retained their recorded hashes; the reserved manifest was excluded from reads.
Tests passed: **136 local / 132 deliverable**. Results are stored in `reports/accuracy_stage_b/tests.txt` and `git_deliverable_tests.txt`. The latter excludes the four pre-existing untracked Phase 3 tests. Git delivery is reported in the task response; checkpoints, full runs, raw/processed data, generated labels and caches are ignored.

Reproduction: [commands and protocol](../reproducibility/ACCURACY_STAGE_B.md). Detailed evidence: `reports/accuracy_stage_b/`. Existing E0/E1/E2, Phase 3 and Stage A artifacts remain preserved. No fusion, weighted YOLO fine-tuning, full E3 or reserved-split evaluation was started.
