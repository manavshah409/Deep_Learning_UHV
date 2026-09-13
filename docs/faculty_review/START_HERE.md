# Faculty review pack - start here

This is a **verified Phase 1 progress deliverable**, not a claim that the complete baseline has finished.

## What to show

1. `output/pdf/UVH26_Faculty_Progress_Report.pdf`: polished faculty-facing progress report.
2. `docs/faculty_review/PRESENTATION_SCRIPT.md`: a 6-7 minute explanation, demo sequence and likely viva questions.
3. `reports/figures/class_distribution.png`: the main data finding.
4. `reports/audit/pytest.txt`: saved test evidence.
5. `reports/tables/yolov8n_uvh26_mv_smoke_seed42_provenance.json`: executed training evidence.

## Offline evidence demo

From the extracted project directory, run:

```bash
python3 scripts/show_progress.py
```

This command only reads the included reports. It needs Python 3 but no dataset, weights or additional packages.

For the actual project environment and tests:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Dependencies must be installed before an offline classroom demo. A new installation may require network access.

## What is included

Source code, synthetic tests, portable configurations, exact package versions, the EDA notebook, generated charts and tables, annotation audit summaries, smoke training/evaluation evidence, technical documentation and the presentation materials.

## What remains local

Raw images and annotations, processed training data, checkpoints, caches, logs, local paths, virtual environments and Git credentials. No model weights or original dataset images are inside the ZIP. The image download and proper baseline continue separately in the working project.

## Current experimental boundary

The executed smoke test used 64 training and 32 validation images for one epoch. It verifies loading, finite losses, checkpoint saving and evaluation. Its detection quality is poor and it is not the final model.

The intended proper baseline uses 8,000 training and 2,000 validation images for 30 epochs. Its final metrics must be added only after that run completes. All remaining completion criteria are documented in `docs/phase_reports/PHASE_1_BASELINE.md`.
