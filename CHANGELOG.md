## Accuracy Improvement Stage E — 2026-09-23

- Added fail-closed calibration manifest/path guards, common prediction schema and independent COCO evaluator for E1/E3.
- Generated fresh immutable calibration500 prediction bundles; preserved a failed padding-coordinate export and repaired the adapter under new v2 IDs with explicit out-of-frame rejection accounting.
- Selected operating thresholds on a preregistered 100-point macro-F1 grid; exported all curves, class support, confusion/errors and object-level complementarity evidence.
- Sealed the future matched E1/E3/E4 protocol, paired bootstrap and latency procedure; reserved gate remains closed pending E4 calibration freeze and separate authorization. No reserved access, fusion or training.

## Accuracy Improvement Stage D — 2026-09-23

- Audited completed 20-epoch E3 and preserved the original/recovery session evidence; selected epoch 13 with finite hash-verified checkpoints.
- Independently reproduced calibration500 AP50 0.563561 / AP50:95 0.429070 exactly; exported class support, fixed-threshold F1, PR/confusion and qualitative error evidence.
- Measured deterministic 100-image batch-one MPS timing, training curves and active/calendar time; updated the faculty briefing, dashboard artifacts and reproducibility commands.
- Reserved1500, matched E1 comparison, fusion and further training remain unstarted. Earlier entries below describe historical stage status.

## E3 checkpoint-publication recovery — 2026-09-23

- Fixed same-inode hard-link alias publication after a best-metric plateau.
- Added an exact run/config/source-hash repair allowlist while preserving original checkpoint/configuration provenance.
- Added detached same-run resumption with independent attempt logs and consistent status reporting.

## Accuracy Improvement Stage C0 — 2026-09-21

- Added a detached E3 supervisor, atomic exit-status evidence and read-only status helper; harmless success/nonzero-exit and duplicate-launch tests passed.

- Added versioned epoch-boundary recovery with optimizer/RNG/schedule restoration, run locks, provenance checks, atomic checkpoint publication and CSV reconciliation.
- Preserved best/last checkpoints and partial-attempt/failure evidence; new and resume CLI modes are mutually exclusive.
- Passed 39 new recovery cases, 62 targeted tests and 175 local tests; CPU resumed/uninterrupted states match exactly, and tiny MPS resume smoke passed. No full E3 launch or reserved-split access.

## Accuracy Improvement Stage B — 2026-09-19

- Completed paired three-epoch COCO-initialized Faster R-CNN MPS pilots on deterministic 1000/250 train/calibration subsets.
- Frozen batch-one SGD .001 and two-epoch warm-up with proposed epoch13/18 decays; exported per-class metrics, timings, coverage and all checkpoint hashes.
- Provisional sampling choice: unweighted; full E3, E4 fusion and reserved1500 evaluation remain unstarted.

## Accuracy Improvement Stage A — 2026-09-18

- Added official Faster R-CNN ResNet-50-FPN pipeline, explicit background mapping, capped inverse-square-root image sampling and durable epoch CSV logging.
- Prepared deterministic 500-calibration/1,500-reserved manifests and distribution diagnostics. Historical validation reuse remains disclosed.
- Completed one-epoch MPS preflight on 128/64 images in 52.00 s; 125 local tests / 121 repository-deliverable tests passed; no full training or fusion.

# Phase 3 initial delivery — 2026-09-18

- Added hash-verified streaming video detector smoke, immutable outputs, timing and failure records.
- Added frozen proposed ByteTrack/counting configuration and architecture; tracking/counting are not yet implemented.
- 90 tests pass; actual MPS checkpoint processed 20 synthetic frames. Real-video evaluation awaits supplied clips.

# E2 closeout — 2026-09-18

- Finalized inference-only resolution study: retain YOLOv8s 640; Gate B not entered.
- Preserved superseded exports and corrected paired evaluations; added size/protocol integrity tests and dashboard comparison.

# Changelog

## Phase 2 E1 evaluation and comparison closeout - 2026-09-17

- Confirmed 30 successful epochs, best epoch 22, no early stop, readable/hash-verified best and last checkpoints; preserved both completed training runs.
- Fixed the project evaluator: retain the local validator through the supported hook instead of accessing a nonexistent wrapper attribute. Publish complete evaluation bundles atomically and reject missing/non-finite metrics.
- Reevaluated both checkpoints with identical source/settings and new IDs; E0 exactly reproduces the Phase 1 overall metrics. E1 mAP50:95 is 0.524760 (+6.635 pp), aggregate F1 0.629658.
- Completed matched MPS timing, six-scene GT/E0/E1 visual review, all-class comparisons and preferred-model decision. E1 selected for detection quality, with size/inference cost and rare-class limitations explicit.
- Added evaluator regression tests, artifact validation, measured report, reproduction commands and dashboard results. No new training or Phase 3 work.

## Phase 2 E1 preparation and recovery - 2026-09-15

- Registered the controlled YOLOv8s comparison while preserving E0 and the frozen data.
- Preserved failed preflight v1 (exit 1 before validation/checkpoint saving); fixed the added loss guard to accept Ultralytics named-loss dictionaries.
- Added comparison and loss-guard tests: 68 tests pass. Recovery preflight v2 uses 800 training and all 2,000 validation images; proper E1 keeps its 30-epoch, 8,000/2,000 budget.

## Phase 1 baseline closeout - 2026-09-13

- Verified successful exit 0, 30 completed epochs, best epoch 30 and no early stopping; preserved the training run and both checkpoints.
- Reverified all 10,000 selected image/label hashes, dataset paths, manifests, class mapping, frozen optimizer and training-source provenance.
- Executed fresh best-checkpoint validation on 2,000 images: P 0.609993, R 0.543547, mAP50 0.560049, mAP50:95 0.458407; recorded both F1 aggregations, per-class scores, confusion matrix and curves.
- Added explicitly synchronized MPS batch-one stage timing: 31.732 ms median / 34.359 ms p95 end-to-end still-image latency, 31.891 images/s; no video FPS claim.
- Reviewed six paired validation scenes from 101 diagnostic cases; separated source annotation limitations from model errors.
- Rebuilt Phase 1 report, artifact dashboard, faculty PDF, speaking script, offline demo and portable ZIP; retained historical smoke/preflight evidence separately.
- Re-ran complete tests, selected-data/evaluation validation and the subset Phase 2 gate. Delivery commit/push evidence lives in reports/audit/closeout_delivery.json. Phase 2 training remains unstarted.

## Preparation and recovery - 2026-09-11 to 2026-09-13

- Audited the full 26,646-image MV annotation catalogue, implemented strict COCO-to-YOLO conversion, EDA and deterministic subset selection.
- Completed smoke and full-subset preflight checks as engineering diagnostics.
- Quarantined inconsistent/degraded source candidates, preserved originals and froze the audited 8,000/2,000 v2 subset with all 14 classes.
- Initial source checkpoint c82e277 recorded the project while proper training was active; later closeout records supersede that progress snapshot.

Unselected-image acquisition/integrity verification remains incomplete. Catalogue metadata audit and selected-image pixel audit are separate claims.
