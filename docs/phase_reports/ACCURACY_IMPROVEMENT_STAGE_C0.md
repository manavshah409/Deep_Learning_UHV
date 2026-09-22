# Accuracy Stage C0 — safe epoch-boundary resume

Status: engineering implementation and tests complete. **No full E3 training was launched.** Stage B's selected unweighted method, physical batch 1, seed 42, optimizer and 20-epoch schedule remain unchanged. No reserved comparison data, fusion, weighted detector training, YOLO fine-tuning or Phase 3 work was performed.

## Original blocker and implementation

The previous runner created only new directories, reset the optimizer and epoch counter, and could load model weights without restoring training state. The current `src/experiments/stage_b.py` integrates `src/training/epoch_resume.py` for new runs and resumes. Existing Stage A/B checkpoints and reports are preserved; their older checkpoint schema is intentionally not upgraded or used for recovery.

New-run and resume IDs are mutually exclusive CLI arguments. New mode refuses an existing directory. Resume requires the original incomplete directory and refuses `COMPLETE.json`. It restores the last verified epoch inside that directory; it never clones the experiment or starts another scientific run. A persisted run UUID and run name prevent importing another run's checkpoint even when its hyperparameters match.

A POSIX kernel `flock` is held throughout initialization, restoration, training and persistence. The `.run.lock` PID/session metadata is informational. Lock ownership, rather than `kill(pid,0)` alone, establishes concurrent execution: dead-process locks are automatically released and a reused live PID in an old file does not cause a false refusal. No process is killed by recovery.

## Version 2 checkpoint schema

Each completed checkpoint contains:

- Model and optimizer state, including momentum; optional scheduler object's state (the approved runner uses a declarative schedule, not a scheduler object).
- Completed/next epoch, current/next LR, exact schedule definition and hash, per-epoch LR table.
- Full frozen configuration and canonical hash: architecture, 15 output classes, optimizer definition, sampling, seed, dtype/device, resize, input manifest hashes, class mapping, initializer provenance, original configuration hash, library versions and source hashes.
- Run UUID/name, checkpoint format version, process-session ID.
- Cumulative completed-epoch active time, best metric/epoch, completed timing row and complete preceding timing history.
- Python `random`, NumPy (safe primitive representation), Torch CPU RNG state; MPS RNG state when getter and setter APIs exist. The installed MPS APIs were exercised by the smoke test.
- Sampler policy/seed and next-epoch seed. Both samplers use a private `Random(seed + epoch - 1)` and workers remain zero; there is no separate persistent loader generator to restore.

An initial epoch-zero checkpoint permits recovery if the first epoch is interrupted. Training checkpoints are retained for every completed epoch. Verified `last.pth` and `best.pth` are atomically replaced hard-link pointers to immutable canonical checkpoint files; they do not duplicate model storage. Best means maximum calibration AP50:95, retaining the earlier epoch on ties. No historical epoch checkpoint is overwritten.

## Publication order and durability

1. Finish training and calibration, export diagnostics, and construct the complete timing row.
2. Write checkpoint to a unique temporary file, flush/fsync it, safely deserialize it with `weights_only=True`, and validate provenance, epoch/LR position, timing history and finite model/optimizer tensors.
3. Atomically rename the checkpoint; fsync the directory; verify the published SHA-256. Atomically write its hash/size/epoch receipt.
4. Append the completed CSV row and flush/fsync both file and directory.
5. Publish verified last/best pointers and atomically update `run_state.json`.

The canonical checkpoint plus its receipt is authoritative. State JSON and last/best pointers may lag a crash and are rebuilt from it. Temporary or unsealed future checkpoint files are not accepted as completed epochs. Partial-epoch sampling/loss/metric files and unsealed files are moved into `partial_attempts/<recovery-session>/` before that epoch is retried, preserving evidence rather than silently deleting it.

A corrupt latest sealed checkpoint is refused; recovery does not silently fall back to a more favorable or older checkpoint. `COMPLETE.json` is published only after all configured epochs have been committed. Caught failures write `FAILURE.json` and a separate immutable session failure record. Prior failure evidence remains after a successful resume; current status is obtained from run state and completion evidence, not the mere presence of historical `FAILURE.json`.

## Timing reconciliation

| CSV versus verified checkpoint | Action |
|---|---|
| Same completed epoch and matching history | Continue with next epoch |
| Checkpoint exactly one epoch ahead | Restore its embedded missing row, flush/fsync; repeat recovery is idempotent |
| CSV ahead | Refuse |
| Checkpoint more than one epoch ahead | Refuse |
| Duplicate, missing, malformed or mismatched rows | Refuse |

Failed/partial epochs never receive a completed row. A torn partial CSV write is refused for manual evidence review rather than silently truncated.

Cumulative active time is the sum of **completed epoch work**, including training/calibration/export but excluding checkpoint commit overhead, discarded partial work and downtime between sessions. This definition is explicit because the row must exist before checkpoint persistence. On this Mac, `CLOCK_UPTIME_RAW` excludes system sleep. UTC endpoints and per-session wall/uptime measurements are retained separately in `sessions/`; partial work is therefore distinguishable from completed cumulative time. Other platforms fall back to monotonic time with its platform-specific suspend semantics recorded.

## Resume provenance policy

The new invocation recomputes the frozen train/calibration, mapping, initializer and configuration hashes before loading recovery state. The checkpoint must match the original configuration, run identity and supported format. Changed sampling, seed, architecture/classes, optimizer, schedule, resize, library versions or source hashes are rejected. There is no CLI override that silently changes scientific inputs.

Only runtime metadata (PID, session identifier, timestamps and elapsed session time) may differ. Relocating the repository with unchanged relative paths and bytes is possible; renaming the scientific run is refused. Logging overrides are not exposed; even a logging-only source edit currently changes the strict source hash and requires review rather than being silently permitted.

The input allowlist remains the frozen training manifest and calibration500 (or the original Stage B pilot manifests for pilot mode). The reserved1500 manifest and images are never read by this implementation or these tests.

## Measured engineering evidence

- **39 new resume test cases**, covering all 20 requested behaviors plus decay-boundary LR recovery, cross-run checkpoint refusal, partial CSV refusal, non-finite optimizer state, initial-epoch recovery and fsync ordering.
- **62 targeted tests passed** (resume, Stage A and Stage B).
- **175 local tests passed** in the complete suite; this includes four pre-existing, unrelated Phase 3 tests.
- CPU interruption experiment: complete epochs 1–2, interrupt during epoch 3, resume the same run, finish epoch 3. Timing rows are exactly `[1,2,3]`; model and optimizer tensors match the uninterrupted reference **exactly**. Best epoch remains 2; synthetic cumulative durations restore to `1+2+3=6` seconds (not a measured performance result).
- Tiny MPS engineering smoke: one synthetic Linear-model epoch, simulated stop, one resumed epoch; optimizer momentum restored to `mps:0`, and the saved MPS RNG stream reproduced. No detector dataset or real E3 training was used.
- Frozen Stage B input validator passed without opening the reserved split.

Pre-launch verification additionally passed **42 resume/launcher tests** and **178 full local tests**, including detached harmless-command exit codes 0 and 7. No full-training success is implied.

Portable evidence: `reports/accuracy_stage_c0/` contains test outputs, CPU integration JSON, MPS smoke JSON, CLI help and protocol validation. Local ignored engineering artifacts are under `runs/E3_stageC0_cpu_resume_v2/` and `runs/E3_stageC0_mps_resume_v1/`. Original Stage A/B experiment results are unchanged.

## Commands — documented only, not executed

Run from the repository root after a separate Stage C launch authorization and its resource/process checks.

```bash
# New immutable scientific run
caffeinate -i .venv/bin/python -m src.experiments.stage_b \
  --sampling unweighted \
  --run-id E3_fasterrcnn_unweighted_20ep_seed42_v1 \
  --allow-full-training

# Resume that same incomplete run (sampling comes from its frozen configuration)
caffeinate -i .venv/bin/python -m src.experiments.stage_b \
  --resume-run-id E3_fasterrcnn_unweighted_20ep_seed42_v1 \
  --allow-full-training
```

These are foreground CLI examples. A tested detached supervisor is now available; see the reproduction instructions for launch and status commands. Stage C must still validate disk, process state and MPS resources, then use a detached launcher with PID/log/exit-status artifacts. This task does not launch or monitor a 20-epoch job.

## Remaining risks

- Recovery is at epoch boundaries; partial epoch work is deliberately repeated. MPS numerical bitwise equivalence is not claimed by a tiny engineering smoke.
- SIGKILL, power loss or a filesystem failure may prevent Python from writing a final failure/session record. The kernel lock is still released; sealed checkpoint/CSV reconciliation determines safe progress on the next attempt. An unfinalized session's end time is unknown, not fabricated.
- A damaged receipt/checkpoint, torn CSV or provenance disagreement requires investigation, not automatic repair. If interruption precedes a valid initial checkpoint, automatic recovery is unavailable.
- Local POSIX locking and same-filesystem atomic rename/hard links are assumed. Network/distributed filesystems and concurrent non-cooperating writers are unsupported.
- Hashes detect accidental changes; they are not cryptographic authentication against an attacker who can rewrite both payload and receipt.
- Full training stability, final accuracy and the later LR decay behavior remain untested. Full E3, fusion and reserved evaluation remain unstarted.
