# Stage E reproduction

From the repository root, inspect or validate without opening reserved data:

```bash
.venv/bin/python scripts/validate_stage_e.py
.venv/bin/python scripts/validate_stage_e.py --local
.venv/bin/python -m pytest -q tests --ignore=tests/test_event_evaluation.py
```

`--local` hashes the two selected checkpoints and validates only the calibration prediction bundles against the allowed calibration500 manifest. It never opens reserved data.

On a fresh output workspace only, with the existing selected checkpoints and frozen calibration inputs present:

```bash
.venv/bin/python scripts/calibrate_e1_e3_stage_e.py
.venv/bin/python scripts/freeze_e1_e3_protocol.py
```

The runner uses v2 IDs and refuses existing report/run directories. Existing local outputs must not be deleted or overwritten. It deliberately does not expose a reserved authorization flag. To repeat scientifically, first preregister new IDs/source hashes and preserve this frozen protocol; do not edit existing bundles to force agreement. The v1 failed coordinate-conversion attempt remains archived locally and is documented in `coordinate_recovery.json`.

Large JSONL predictions and original logs stay under ignored `runs/`. Common evaluator, schema, fixed calibration plan, full threshold curves, summaries, confusion/error evidence, paired recovery counts and future protocol are committed. Future reserved access requires a separately authorized command and a completed E4 family seal (or an explicit pre-access family amendment). No E4 implementation or reserved execution command is provided in Stage E.
