# Accuracy Stage A reproduction

Use the existing frozen environment plus pinned pycocotools from `requirements-e2-evaluation.txt`. Official Torchvision 0.29.0 and Torch 2.14.0 remain unchanged. Initializer is recorded in `reports/accuracy_stage_a/initializer.json`; checkpoint and datasets are local-only. Protocol v2 contains 500 calibration / 1,500 reserved evaluation and 128/64 preflight manifests. Do not regenerate into an existing protocol/run directory.

Executed one-epoch command:

```bash
.venv/bin/python -m src.experiments.train_frcnn --config configs/accuracy/E3_fasterrcnn.yaml --run-id E3_stageA_preflight_seed42_v1 --preflight
```

Do not rerun this immutable ID. A deliberate repeat requires a new run ID.

Verification:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_accuracy_stage_a.py
```

**Proposed later 20-epoch command — not executed and requires new authorization:**

```bash
.venv/bin/python -m src.experiments.train_frcnn --config configs/accuracy/E3_fasterrcnn.yaml --run-id E3_fasterrcnn_weighted_seed42_full_v1 --allow-full-training
```

The proposal uses 8,000 training images and 500 calibration images for per-epoch validation; it does not read the reserved 1,500 evaluation images. 20 epochs is an initial budget proposal, not evidence that 20 epochs are sufficient. No full-training approval is implied by documenting the command. A scheduler/unweighted control and the scientific budget should be reviewed before authorizing that later run. Do not select fusion parameters or compare checkpoints on the 1,500 reserved split until the final protocol is frozen.
