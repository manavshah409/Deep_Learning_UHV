# Local model artifacts

The completed proper baseline is `runs/yolov8n_uvh26_mv_baseline_seed42_v1/weights/best.pt`, produced at epoch 30 of 30. `last.pt` is alongside it. Both are readable, finite, 14-class checkpoints of 6,228,714 bytes.

- best.pt SHA-256: `85b9e089ec3dfaa676e30a6191b7e5726f320d5bf85391f1675922b88f9778b3`
- last.pt SHA-256: `19cf82c3848fafae370fa36dd16e845a26dc38eb83bd80a473bb24691268551a`

Fresh frozen-subset validation: mAP50 0.560049, mAP50:95 0.458407. See `docs/phase_reports/PHASE_1_BASELINE.md` for scope, full metrics and proper MPS timing. The original COCO initializer remains `models/yolov8n.pt`; smoke/preflight weights are historical engineering checks.

Weights and complete runs remain local, excluded from Git, Git LFS and the faculty ZIP. Preserve the completed run unchanged; use new IDs for later controlled experiments. No production deployment claim is made.
