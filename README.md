**E2 resolution study closed:** Gate A failed: overall AP50:95 +0.038 pp; small-object macro AP −7.076 pp. Retain E1 YOLOv8s at 640. No 960 training. [E2 report](docs/phase_reports/PHASE_2_E2_RESOLUTION_ABLATION.md).

# UVH-26 vehicle detection: Phase 1 subset baseline

**Phase 1 and the Phase 2 E1 comparison are complete.** YOLOv8s trained for 30 epochs (best epoch 22; exit 0; no early stop). Its initial evaluation export failed; the evaluator was repaired and both checkpoints were evaluated successfully under new IDs. No retraining or other experiment was started.

**Preferred research detector: E1 YOLOv8s at 640.** mAP50:95 improves from 0.458407 to 0.524760 (+6.635 percentage points); harmonic aggregate F1 improves from 0.574856 to 0.629658. All 14 class AP50:95 values improve, but Others recall remains zero and some dense-scene class errors worsen.

Matched MPS batch-one timing: E0/E1 mean end-to-end 30.364/30.292 ms; median 31.167/30.537 ms; p95 33.681/32.063 ms; 32.934/33.012 still images/s. E1 mean inference is 17.1% slower and weights are 3.61x larger. The tiny end-to-end difference is not evidence of a speed advantage or live-video performance. See the [E1 technical report](docs/phase_reports/PHASE_2_E1_MODEL_COMPARISON.md), [recovery note](docs/phase_reports/E1_EVALUATION_RECOVERY.md), [comparison table](reports/comparisons/E1_vs_E0/comparison.csv) and [reproduction instructions](docs/reproducibility/E1_EVALUATION.md).

The Phase 1 results and faculty pack below remain the historical E0 deliverable; the E1 report and dashboard provide the completed model comparison.

**Scope:** 8,000 training / 2,000 validation images, 14 Majority Voting classes. These are subset validation results, not full-dataset or test-set scores.

| Checkpoint / evaluation | Precision | Recall | F1 (harmonic aggregate) | Macro F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n best epoch 30 / frozen 2,000 val | 0.609993 | 0.543547 | 0.574856 | 0.539397 | 0.560049 | 0.458407 |

F1 above is the harmonic mean of macro Precision/Recall; macro F1 averages class F1 at confidence 0.3093. Others has zero recall; its reported precision 1 is an evaluator convention. Three-wheeler AP50:95 is 0.7403; Others 0.0242 and Mini-bus 0.1383 remain weak.

Batch-one Apple M5 MPS timing (640, float32, 10 warm-ups, 100 images): median **31.73 ms**, p95 **34.36 ms**, **31.89 end-to-end still images/s**; inference-only **201.77 FPS**. Includes file decode/preprocess/inference/postprocess and synchronized measurement, excludes capture/display. No real-time video or production-readiness claim.

The **26,646-image annotation-catalogue audit** covers metadata. The separate **10,000-image local subset integrity audit** covers actual decoded images, dimensions, hashes, labels and split leakage. Acquisition/integrity verification of unselected images remains incomplete. Frozen subset has 94,609 train and 24,342 validation objects. Raw sources and the completed run are unchanged.

## Phase 1 faculty demonstration (historical E0 pack)

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

Do not rerun training to demonstrate results. Existing run/evaluation IDs refuse overwrite. No tracking/counting, independent test scores or full pixel-audit completion is claimed. Initial smoke and full-subset preflight evidence remains historical and separate from this measured baseline.

## Phase 2 E1

E1 compares YOLOv8s with the frozen YOLOv8n reference. Same 8,000/2,000 subset, imgsz640, batch8, seed42 and 30-epoch budget. [Preregistered protocol and current status](docs/phase_reports/PHASE_2_E1_MODEL_COMPARISON.md). E1 training and standalone evaluation are complete; preflight remains a separate pipeline check.
