# Changelog

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
