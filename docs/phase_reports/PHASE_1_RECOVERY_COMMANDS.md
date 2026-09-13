# Phase 1 subset recovery command record

Run from the repository root using Python 3.12 and the project virtual environment. Raw inputs remain immutable. Commands that freeze existing versions intentionally refuse to overwrite them.

Executed initial candidate audit:

```bash
.venv/bin/python -m src.data.audit_baseline_subset
.venv/bin/python -m src.data.investigate_dimensions
.venv/bin/python -m src.data.recover_baseline plan
.venv/bin/python -m src.data.audit_baseline_subset --selection data/interim/subset_plans/baseline_seed42_recovery_v1/selection.json --prefix baseline_subset_final
.venv/bin/python -m src.data.recover_baseline freeze
.venv/bin/python -m src.data.validate_yolo --dataset-version uvh26_mv_yolo_v1/subsets/baseline_seed42
.venv/bin/python -m src.data.review_baseline
```

Manual visual review superseded that candidate before training. Conservative visual recovery:

```bash
HF_HOME="$PWD/data/interim/hf_home" HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 .venv/bin/python -m src.data.revise_visual_subset
.venv/bin/python -m src.data.audit_baseline_subset --selection data/interim/subset_plans/baseline_seed42_recovery_v2/selection.json --prefix baseline_subset_final_v2
```

The candidate and replacement-selection files remain local under ignored `data/interim/`. Summarized decisions and checksums are preserved under `reports/audit/`. Exact final image IDs are present in the audit records and final manifests; generated labels and image symlinks are excluded from Git.
