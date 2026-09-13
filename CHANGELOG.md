# Changelog

## 0.1.0 — Phase 1 in progress (2026-09-11)

- Initialized the project environment and pinned the official UVH-26 revision.
- Inspected all four consensus annotation files and audited Majority Voting annotation geometry and references.
- Generated annotation-based exploratory analysis for 26,646 images and 316,220 objects.
- Implemented versioned COCO-to-YOLO conversion, strict validation and coverage-aware subset selection.
- Added training, evaluation, diagnostic matching and CLI inference tools.
- Added synthetic unit and integration tests; see the saved test output.
- Completed a separately audited 64/32 pilot and genuine one-epoch MPS smoke training (35.7 seconds total wall time).
- Inspected early annotation and smoke prediction images, documenting imperfect labels and heavily redacted imagery.
- Full image acquisition, full image integrity audit and proper baseline training are pending. This release is not Phase 1 completion.


## Phase 1 subset recovery and training checkpoint

- Audited and froze the 8,000/2,000 MV subset with deterministic quarantine/replacement records and no content leakage.
- Verified 42 annotation overlays, full-subset preflight checkpoints and 42 preflight prediction images.
- Froze explicit AdamW settings and started the separate 30-epoch baseline.
- Added audit/recovery, latency and artifact-driven faculty dashboard support; 58 tests pass.
- This is an in-progress Git checkpoint requested during training. Final baseline evaluation, latency and updated faculty results remain pending.
