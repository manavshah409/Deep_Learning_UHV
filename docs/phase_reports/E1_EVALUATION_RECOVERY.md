# E1 evaluation export recovery

Training succeeded: 30 epochs, exit 0, best epoch 22, no early stop. Checkpoint verification also exited 0. The original standalone validation completed image processing, then export exited 1. Its output directory, partial metrics/CSV and log are preserved; they are not the final comparison source.

Confirmed cause: project evaluator accessed `model.validator.confusion_matrix_conf`. Runtime types are high-level `YOLO` and internal `DetectionModel`. Installed Ultralytics 8.4.146 `YOLO.val()` constructs a local validator, runs it, stores only its metrics on the wrapper, and returns `DetMetrics`. The wrapper has no saved `validator`; its attribute fallback reaches `DetectionModel`, causing the reported AttributeError. The wrapper was not overwritten with `.model`.

The repair retains an explicit `DetectionValidator` subclass through supported `YOLO.val(validator=...)`. Standard metrics come from returned `DetMetrics`; confusion threshold and prediction counts come from the retained validator. No package/dependency files were edited. Missing/non-finite metrics raise errors. All result files are prepared in a temporary sibling directory and exposed with a single directory rename only after export succeeds; COMPLETE.json lists file hashes. Existing output IDs are refused.

Both E0 and E1 are evaluated again with identical repaired source and protocol, because the export implementation changed. New IDs: `yolov8n_uvh26_mv_e0_validation_seed42_v2` and `yolov8s_uvh26_mv_e1_validation_seed42_v2`. Their complete bundles live in `reports/evaluations/`. Original Phase 1 metrics remain historical evidence.

Protocol: frozen 2,000 validation images; MPS, batch 8, 640, workers 0, conf .001, NMS .7, max_det 300, AP IoU .50:.05:.95. Confusion matching IoU .45 is read from the installed API signature; its confidence .001 is read from the retained validator. These are different from AP's IoU integration range and each model's max-F1 confidence.

Regression tests cover returned metrics without any model/validator attribute, missing/non-finite field rejection, absence of `.validator` accesses in evaluator source, rollback of failed exports, and refusal to overwrite complete bundles. Source/hash/environment preservation evidence: `reports/audit/E1_evaluation_recovery_preservation.json`.
