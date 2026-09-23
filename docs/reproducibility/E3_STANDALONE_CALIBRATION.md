# E3 Stage D reproduction and read-only validation

Run commands from the repository root with the pinned `.venv` dependencies. No command below starts training or reads the reserved split.

```bash
# Read-only validation of delivered summaries plus local checkpoint/manifest receipts:
.venv/bin/python scripts/validate_e3_stage_d.py --local
# Report-only validation, works without datasets/checkpoints:
.venv/bin/python scripts/validate_e3_stage_d.py
# Dashboard:
.venv/bin/python -m streamlit run app.py
# Deliverable suite (the untracked Phase 3 test is unrelated work):
.venv/bin/python -m pytest -q tests --ignore=tests/test_event_evaluation.py
```

## Independent evaluation reproduction

On a fresh output workspace with the completed original run, calibration500 and official initializer available:

```bash
.venv/bin/python scripts/evaluate_e3_stage_d.py
.venv/bin/python scripts/render_e3_stage_d.py
```

The fixed `E3_fasterrcnn_best_calibration500_v1` ID refuses overwrite. The delivered bundle already exists locally: do not delete it or rerun these commands against it. For a deliberate new independent repetition, assign a new ID without changing the evaluator science settings:

```bash
.venv/bin/python - <<'PYCODE'
import scripts.evaluate_e3_stage_d as e
# Choose a never-used ID. Existing output bundles always raise FileExistsError.
e.ID = 'E3_fasterrcnn_best_calibration500_v2'
e.BUNDLE = e.ROOT / 'runs' / e.ID
e.REPORT = e.ROOT / 'reports/evaluations' / e.ID
e.main()
PYCODE
```

This reproduction command includes audit, calibration-only evaluation and the 100-image benchmark; it is not a routine progress command. MPS access must be available. The renderer defaults to the original v1 bundle; its output image directory also refuses overwrite. Raw predictions/qualitative JPEGs are local under `runs/`; committed reports contain only compact summaries and plots. AP settings, fixed-threshold F1/confusion matching and timing boundaries are documented in the technical report and `protocol.json`.
