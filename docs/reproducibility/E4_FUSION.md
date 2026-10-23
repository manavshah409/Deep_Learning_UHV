# Stage F reproduction and validation

From the repository root:

```bash
# Read-only summaries, protocol/code hashes, folds, gates and historical preservation:
.venv/bin/python scripts/validate_stage_f.py
# Also verify selected checkpoints, cached predictions and exact OOF hashes:
.venv/bin/python scripts/validate_stage_f.py --local
# Entire local suite, including pre-existing local Phase3 tests:
.venv/bin/python -m pytest -q
# Fusion-specific tests:
.venv/bin/python -m pytest -q tests/test_fusion.py
# Artifact-only dashboard:
.venv/bin/python -m streamlit run app.py
```

The scientific commands used were:

```bash
.venv/bin/python scripts/stage_f_fusion.py
.venv/bin/python scripts/benchmark_stage_f.py
```

Both are recorded for reproducibility, not rerun instructions against existing outputs. The CV command refuses existing `reports/accuracy_stage_f` or `runs/E4_stageF_v1`; benchmark refuses existing latency results. Never delete or overwrite completed artifacts. A deliberate repeat requires fresh preregistered output IDs and preservation of all earlier evidence. The protocol seals source hashes before CV; no expanded search or reserved flag exists.

CV uses only cached low-floor predictions, verified calibration labels and training-fold-only parameter selection. The latency command alone performs actual sequential detector inference on the existing calibration100 sample. Large OOF predictions/logs/checkpoints remain local under ignored `runs/`. A checkpoint/manifest/protocol/prediction hash mismatch creates a failure record and stops execution.

The delivered outcome rejects fusion. `E4_fusion.yaml` describes only the strongest-family fold0 benchmark representative and must not be treated as a selected production model. No training, E5 command or reserved-comparison command is supplied here.
