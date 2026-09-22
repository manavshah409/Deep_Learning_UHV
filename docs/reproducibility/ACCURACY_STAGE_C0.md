# Stage C0 reproduction

Engineering evidence only. No dataset or detector training is required for the resume tests.

```bash
.venv/bin/python -m pytest tests/test_epoch_resume.py tests/test_accuracy_stage_a.py tests/test_accuracy_stage_b.py -q
.venv/bin/python -m src.experiments.stage_b --validate
.venv/bin/python -m src.experiments.stage_b --help
```

The synthetic CPU demonstration was run with `scripts/validate_resume_smoke.py`; the optional MPS demonstration used `scripts/validate_mps_resume_smoke.py`. They refuse existing engineering result directories. Preserve the delivered artifacts; use a new engineering run ID in a reviewed copy if another execution is needed. Test-suite fixtures use fresh temporary directories automatically.

The full-run and same-run resume commands, checkpoint schema, recovery rules and limitations are in [the Stage C0 report](../phase_reports/ACCURACY_IMPROVEMENT_STAGE_C0.md). They are proposals for a separately authorized Stage C launch, not instructions to start full training during C0.

Frozen hyperparameters remain SGD base LR .001, momentum .9, weight decay .0005; epoch 1 LR .0005, epochs 2–12 .001, 13–17 .0001, 18–20 .00001. Full-mode inputs remain training8000 and calibration500 only, batch1, seed42, MPS float32. Resume derives sampling from the frozen run configuration; do not pass `--sampling` with `--resume-run-id`.

Original Stage B runs retain their historical checkpoint format and are complete; they cannot be resumed using the new schema. No migration or historical checkpoint modification is performed.

## Committed detached launcher

The launch-only helper is `scripts/launch_e3.py`. Its supervisor starts in a new session with stdin detached and stdout/stderr redirected. It runs `/usr/bin/caffeinate -i -s` directly without a shell, waits for the actual command, and atomically records its exit code. Harmless commands returning 0 and 7 verified status preservation. A one-shot launch-directory reservation plus inherited supervisor flock prevents duplicate launches; the training runner independently owns its scientific run flock.

```bash
# Only with full-run launch authorization and passing resource/provenance checks:
.venv/bin/python scripts/launch_e3.py --launch

# Read-only, bounded status inspection:
.venv/bin/python scripts/launch_e3.py --status

# Only after the original process has stopped and released its run lock:
caffeinate -i -s .venv/bin/python -m src.experiments.stage_b \
  --resume-run-id E3_fasterrcnn_unweighted_20ep_seed42_v1 --allow-full-training
```

Launcher evidence is ignored under `runs/launches/E3_fasterrcnn_unweighted_20ep_seed42_v1/`: request/commit, supervisor PID, command PID/UTC launch time, stdout/stderr, supervisor log and atomic `exit.json`. The immutable scientific result directory is created only by the runner. The status helper prefers the runner's PID, distinguishes active locks from stale PID records, and reports epoch metrics, best metric, active time, remaining-time estimate and any recorded launcher exit code. A manually resumed session has its own run lock/session PID; the original launcher's exit code remains historical evidence.

Keep the Mac plugged in with its lid open. `caffeinate -s` applies on AC power; `caffeinate` does not make lid-closed training reliable. Training is only **launched** until completed epochs, completion marker and process exit evidence prove otherwise. Supervisor SIGKILL/power loss can prevent an exit artifact; absence must never be interpreted as success.
