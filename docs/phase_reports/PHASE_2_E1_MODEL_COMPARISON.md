# Phase 2 E1: controlled model-capacity comparison

Status: registered; recovery preflight v2 running. No E1 accuracy comparison is available yet. Preflight metrics will be treated only as pipeline diagnostics.

Research question: Does COCO-pretrained YOLOv8s improve UVH-26 vehicle detection, especially difficult categories, enough to justify additional training time, checkpoint size and synchronized still-image latency?

E0 remains the completed `yolov8n_uvh26_mv_baseline_seed42_v1` with frozen best checkpoint SHA-256 `85b9e089ec3dfaa676e30a6191b7e5726f320d5bf85391f1675922b88f9778b3`. E1 is registered as `E1_yolov8s_uvh26_mv_640_seed42`; preflights have separate `_preflight_v1` and `_preflight_v2` suffixes.

Both use `uvh26_mv_yolo_v1/subsets/baseline_seed42_v2`: 8,000 training and 2,000 validation images, unchanged 14-class mapping. These are subset validation results, not independent test-set or live-video results. All 10,000 image/label hashes and E0 run-file hashes were reverified before E1.

## Preregistered controls

Model capacity (n to s) and the corresponding official COCO initialization are the intended independent variable. Image size 640, batch 8, nbs 64, seed 42, 30-epoch budget, patience 10, AdamW lr0 0.000556, weight decay 0.0005, lrf .01, warm-up and augmentations match E0. Explicit AMP=false and workers=0 match E0 effective behavior, although E0 requested true/4. No memory-driven adjustment has been made.

YOLOv8s initializer: official Ultralytics assets v8.4.0, 22,588,772 bytes, SHA-256 `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`. E1 runner derives from the E0 runner in a new source file; the frozen E0 training source is unchanged. It adds an epoch-level finite-loss guard and records the correct initializer source.

## Comparison protocol

Fresh best-checkpoint validation will use the same 2,000-image manifest and evaluator settings as E0: MPS, batch8, imgsz640, conf floor .001, NMS .7, max_det300. Report absolute percentage-point changes in macro Precision/Recall, harmonic aggregate F1, macro per-class F1, AP50 and AP50:95. Each model chooses its own max mean-F1 operating confidence; harmonic aggregate F1 is not micro-F1.

Per-class AP/P/R and confusion-matrix changes will be reported, with explicit attention to Mini-bus (58 validation objects), Others (31), Van, Two-wheeler and Three-wheeler. These are single-seed descriptive comparisons, without significance or independent-generalization claims.

The same diagnostic validation IDs 4232,10518,1364,21621,4711,954 will be rendered as GT/E0/E1 comparisons at confidence .10, NMS .7 and diagnostic matching IoU .5. Source omissions and ambiguous subtype/occlusion boxes remain separate from model errors.

Fresh E0 and E1 batch-one MPS benchmarks will use the same seed42 100-image order, 10 warm-ups, rectangular imgsz640 policy, float32, conf .25, NMS .7 and explicit synchronization per stage. E0's original Phase 1 timing stays unchanged. End-to-end includes local read/decode and predict API overhead, excluding capture/display/model loading; it is still-image throughput only.

The completed report will include observed training duration/storage, actual/best epochs, early stopping, checkpoint hashes, measured comparison tables, qualitative findings, tests and Git delivery. No other Phase 2 experiment is authorized in this task.

## Preflight recovery

Preflight v1 completed 1,000 training batches with finite displayed losses, then exited 1 because the added finite-loss guard expected a tensor rather than the loss dictionary supplied by this Ultralytics version. It did not reach validation or save a checkpoint. Its run, log and failure provenance are preserved. The guard now supports both forms and has regression tests.

Recovery preflight v2 uses one epoch on 800 training images (`fraction=0.1`) and all 2,000 validation images to verify the remaining pipeline. It is not an accuracy result. Proper E1 remains 30 epochs on the full frozen 8,000/2,000 subset.
