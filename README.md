# UVH-26 vehicle detection: Phase 1 subset baseline

**Phase 1 is complete under the subset-based entry gate.** The baseline checkpoint commit `a4eabf1475d479332f1332a99ae24fac94b2b908` is verified on GitHub. The subsequent E1-only controlled YOLOv8s comparison is registered and its recovery preflight has passed; full training is ready.

The proper YOLOv8n run completed **30 epochs**, best epoch **30**, exit **0**, no early stopping. Fresh best-checkpoint evaluation, qualitative review and synchronized MPS timing are complete. E1 YOLOv8s recovery preflight has passed; see the [entry gate](reports/audit/phase2_subset_gate.json) and [delivery status](reports/audit/closeout_delivery.json).

**Scope:** 8,000 training / 2,000 validation images, 14 Majority Voting classes. These are subset validation results, not full-dataset or test-set scores.

| Checkpoint / evaluation | Precision | Recall | F1 (harmonic aggregate) | Macro F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n best epoch 30 / frozen 2,000 val | 0.609993 | 0.543547 | 0.574856 | 0.539397 | 0.560049 | 0.458407 |

F1 above is the harmonic mean of macro Precision/Recall; macro F1 averages class F1 at confidence 0.3093. Others has zero recall; its reported precision 1 is an evaluator convention. Three-wheeler AP50:95 is 0.7403; Others 0.0242 and Mini-bus 0.1383 remain weak.

Batch-one Apple M5 MPS timing (640, float32, 10 warm-ups, 100 images): median **31.73 ms**, p95 **34.36 ms**, **31.89 end-to-end still images/s**; inference-only **201.77 FPS**. Includes file decode/preprocess/inference/postprocess and synchronized measurement, excludes capture/display. No real-time video or production-readiness claim.

The **26,646-image annotation-catalogue audit** covers metadata. The separate **10,000-image local subset integrity audit** covers actual decoded images, dimensions, hashes, labels and split leakage. Acquisition/integrity verification of unselected images remains incomplete. Frozen subset has 94,609 train and 24,342 validation objects. Raw sources and the completed run are unchanged.

## Faculty demonstration

```bash
python3 scripts/show_progress.py
# Optional dashboard, after installing dependencies:
python -m pip install -r requirements.txt -r requirements-dashboard.txt
python -m streamlit run app.py
python -m pytest -q
```

- [Final Phase 1 report](docs/phase_reports/PHASE_1_BASELINE.md): measured results, configuration, hashes, recovery policy and commands.
- [Presentation script](docs/faculty_review/PRESENTATION_SCRIPT.md) and [start guide](docs/faculty_review/START_HERE.md).
- Faculty PDF: `output/pdf/UVH26_Faculty_Progress_Report.pdf`.
- Portable ZIP (local, generated): `deliverables/UVH26_Faculty_Review_Project.zip`; build with `python scripts/build_faculty_pack.py`.
- [Checkpoint/integrity evidence](reports/audit/baseline_closeout_integrity.json); weights remain local under `runs/yolov8n_uvh26_mv_baseline_seed42_v1/weights/`.
- [Measured per-class results](reports/tables/yolov8n_uvh26_mv_baseline_seed42_v1_validation_per_class.csv), [latency](reports/tables/yolov8n_uvh26_mv_baseline_seed42_v1_latency.json), [prediction review](reports/error_analysis/baseline_prediction_review.json).

## Reproducibility and boundaries

Python 3.12.14 / PyTorch 2.14.0 / Ultralytics 8.4.146, frozen training dependencies in `requirements.txt`. Dashboard dependencies are separate. Follow [data documentation](data/README.md) for the pinned official dataset and immutable preparation policy. The portable ZIP intentionally contains no dataset images, generated YOLO labels or checkpoints. Offline reporting works without training dependencies; inference needs the original local data and weights.

Do not rerun training to demonstrate results. Existing run/evaluation IDs refuse overwrite. No tracking/counting, Phase 2 experiments, independent test scores or full pixel-audit completion is claimed. Initial smoke and full-subset preflight evidence remains historical and separate from this measured baseline.

## Phase 2 E1

E1 compares YOLOv8s with the frozen YOLOv8n reference. Same 8,000/2,000 subset, imgsz640, batch8, seed42 and 30-epoch budget. [Preregistered protocol and current status](docs/phase_reports/PHASE_2_E1_MODEL_COMPARISON.md). Preflight is a pipeline check; no E1 comparison result is claimed until proper training and standalone evaluation complete.
