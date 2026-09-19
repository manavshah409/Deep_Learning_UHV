# Accuracy Stage B reproduction

Stage B is configuration selection, not a final detector result. It uses 1,000 training images selected from the frozen 8,000 and 250 calibration images selected exclusively from Stage A's 500 calibration images. It never opens the reserved 1,500-image manifest or its images. Seed 42; rare-first class-covering anchors in a seeded shuffle, then fill to the requested size. All 14 classes are represented. This class-covering subset is not an unbiased random benchmark.

This proposal supersedes Stage A’s initial `.005` / no-scheduler full-training proposal. Stage A documents remain unchanged as historical evidence.

## Frozen shared configuration

`configs/accuracy/E3_stage_b.json` is hash-frozen in `reports/accuracy_stage_b/protocol.json` before either pilot. Official torchvision Faster R-CNN ResNet-50-FPN COCO_V1 initializer, independently loaded for each run; seed-42 random 15-output head (background + 14 classes). No Stage A checkpoint is used. Aspect-preserving min480/max640 resize, no added augmentation, batch 1 without gradient accumulation, workers 0, MPS float32 without AMP, SGD momentum .9 and weight decay .0005. No silent device fallback. The pretrained detector's default first two backbone stages remain frozen. Evaluation uses score floor .001, NMS .5, maxDet300 and COCO AP at IoU .50:.95; macro P/R uses confidence .25 and IoU .5.

SGD base LR is .001, lowered fivefold from the finite but very short Stage A .005 preflight. This is a conservative batch-one choice, not a demonstrated optimum. For one-based epoch `e`:

```text
LR(e) = .001 × min(1,e/2) × .1**(I[e>=13] + I[e>=18])
epoch 1: .0005
epoch 2–12: .001
epoch 13–17: .0001
epoch 18–20: .00001
```

Stage B uses the first three entries; later decays have not been empirically validated. The full proposal keeps the same schedule and optimizer. There is no scheduler search, reserved-split evaluation or early stopping in these pilots. Seeded sampling and initial weights are reproducible; bitwise numerical determinism of all MPS kernels is not claimed.

## Commands

Run from the repository root with its existing virtual environment. These commands reject existing run IDs and protocol files; do not rerun the preparation over the delivered protocol.

```bash
# Validate existing frozen inputs; does not open the reserved split.
.venv/bin/python -m src.experiments.stage_b --validate

# Historical preparation command, already executed; intentionally refuses to overwrite:
.venv/bin/python -m src.experiments.stage_b --prepare

# Original paired jobs (already-used IDs must not be reused):
.venv/bin/python -m src.experiments.stage_b --sampling unweighted --run-id E3_stageB_unweighted_3ep_seed42_v1
.venv/bin/python -m src.experiments.stage_b --sampling weighted --run-id E3_stageB_weighted_3ep_seed42_v1

# This wrapper ran the original pair sequentially, preserving process exits/logs:
caffeinate -i .venv/bin/python scripts/run_accuracy_stage_b.py

# Only after both process records indicate exit zero:
.venv/bin/python scripts/close_accuracy_stage_b.py
.venv/bin/python -m pytest -q
```

The individual commands and wrapper are alternative launch routes, not steps to run one after another. For a new reproduction, use new immutable run IDs and a corresponding new process record; never erase the originals. The original wrapper intentionally refuses to run twice. Detailed draw image IDs, per-step component losses, epoch checkpoints and full stdout/stderr remain under ignored `runs/`. Portable summaries, per-class metrics, all coverage diagnostics and checkpoint hashes are under `reports/accuracy_stage_b/`.

The full-run command is recorded in the Stage B report after the predeclared comparison. It is a proposal requiring separate authorization, not part of Stage B execution. Its evaluation pool is calibration500 only; reserved1500 remains excluded. Full training duration is projected by scaling measured training time ×8 and validation time ×2, preserving measured checkpoint/logging overhead; it is not a completed-run measurement.
