# E3 checkpoint alias repair and epoch-7 recovery

The original E3 process exited 1 after completing and sealing epoch 6. Its CSV contains epochs 1–6; the state JSON lagged at epoch 5 because alias publication failed. The checkpoint's authoritative best epoch is 6, calibration AP50:95 0.4014405759 (configuration/calibration evidence, not reserved evaluation).

## Cause and correction

On POSIX filesystems, renaming one hard link onto another link to the same inode may do nothing, leaving the source link in place. The old code reused one temporary alias filename throughout a session. A non-improving epoch left that temporary best-checkpoint link behind; the next publication failed with `FileExistsError`.

The repair skips an alias already pointing to the canonical checkpoint, otherwise uses a unique temporary name for every publication and removes only its own redundant temporary link if a same-inode rename leaves it behind. Existing stale links are preserved and archived by the normal recovery process. All epoch checkpoints remain immutable.

## Verified recovery boundary

All six checkpoint SHA-256 values match their receipts. Epoch 6 safely deserializes, contains finite model and optimizer tensors, and its full timing history matches all six CSV rows exactly. Completed epoch is 6; next epoch is 7; current/next LR is .001; cumulative completed active time is 11,867.904832 seconds. Existing configuration, provenance, checkpoints and failure evidence are preserved.

The original strict provenance rule would correctly reject any source change. `configs/accuracy/resume_source_compatibility.json` now records one explicit, reviewed code-only exception for this exact run and original configuration hash. Every source hash must match the exact original/approved maps; all non-source configuration fields must remain identical. Only `src/training/epoch_resume.py` changes in the frozen source map. Different hyperparameters, inputs, versions, seeds or unapproved code remain refused.

The original `config.json` and checkpoint configuration hashes are not rewritten. Actual execution source hashes and the approved repair ID are recorded separately in each new session and future checkpoints. This is an explicit repair trail, not a silent provenance bypass or a new scientific experiment.

## Detached resumption and status

The launcher now supports `--resume`, reserves an independent attempt directory beneath the original launcher directory, refuses an active run/supervisor lock, and preserves earlier logs and exit codes. The scientific run ID and directory remain unchanged. The status helper follows the latest attempt and avoids mixing newer CSV metrics with stale best-epoch/state metadata.

```bash
# Authorized recovery only; original process must be stopped and lock-free:
.venv/bin/python scripts/launch_e3.py --resume

# Read-only status:
.venv/bin/python scripts/launch_e3.py --status
```

Evidence: `reports/accuracy_stage_c_recovery/`. Tests cover repeated best aliases, a later improving best, explicit source-repair acceptance, rejection of changed scientific inputs/unapproved code, detached resume locking and stale status metadata. This report records repair eligibility, not successful completion of full training. No reserved-split data, new experiment, fusion, weighted training, YOLO fine-tuning or Phase 3 work is involved.
