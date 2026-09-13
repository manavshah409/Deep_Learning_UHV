# Real-Time Vehicle Detection and Traffic Analytics for Indian Urban Roads Using YOLOv8 and the UVH-26 Dataset

A fourth-year undergraduate deep learning project studying vehicle detection in heterogeneous Indian urban traffic. Phase 1 establishes a reproducible UVH-26 Majority Voting data pipeline and a YOLOv8n transfer-learning baseline. Vehicle counting, density estimation and an application interface are future phases. Real-time performance is a project objective, not an established result.

**Status: audited subset and preflight complete; proper baseline training is running.** The frozen `baseline_seed42_v2` subset contains 8,000 training images (94,609 objects) and 2,000 validation images (24,342 objects). All selected image and label integrity checks passed, with zero split content overlap. A 42-image visual review passed with documented source limitations. The full-subset one-epoch preflight completed and its checkpoints and predictions were verified. The separate 30-epoch run is `yolov8n_uvh26_mv_baseline_seed42_v1`; its current provenance is in `reports/tables/`. Final evaluation and latency measurements remain pending. Full acquisition of unselected images is a separate future task, not a prerequisite for this subset experiment. The faculty PDF and speaking script are historical progress snapshots awaiting the final measured results.

## Observed annotation counts

| Official split | Images in MV JSON | Objects |
|---|---:|---:|
| Train | 21,349 | 252,723 |
| Validation | 5,297 | 63,497 |
| Total | 26,646 | 316,220 |

These are annotation-file counts, not a claim that all image files have been downloaded or decoded. See `reports/audit/annotation_audit.json` and `reports/audit/schema.json`.

## Structure

- `configs/`: portable examples, model settings and generated class mapping.
- `src/data/`: schema inspection, audits, conversion, validation, subsets and EDA.
- `src/training/`, `src/evaluation/`, `src/inference/`: experiment and prediction tools.
- `tests/`: synthetic unit and integration fixtures.
- `notebooks/`: notebook reading generated EDA artifacts.
- `reports/`: measured audits, figures, tables and local diagnostic images.
- `docs/phase_reports/`: technical status and final Phase 1 report.
- `data/`, `models/`, `runs/`: ignored local data, weights and experiments.

## Environment and acquisition

Tested setup: Apple M5 MacBook Pro, 24 GB memory, Python 3.12.14. MPS requires GPU access; it was available outside the execution sandbox. `requirements.txt` pins the installed Python dependencies.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp configs/paths.example.yaml configs/paths.local.yaml
```

Follow `data/README.md` to download the official dataset after checking free disk space. The pinned revision is `59f82c57821e8a54dc40bc1f42e83909dbad0b70`. Never commit datasets or checkpoints.

## Data preparation

Run from the repository root after the download completes:

```bash
python -m src.data.inspect_uvh26
python -m src.data.validate_raw
python -m src.data.convert_to_yolo --dataset-version uvh26_mv_yolo_v1
python -m src.data.validate_yolo --dataset-version uvh26_mv_yolo_v1
python -m src.data.visualize_annotations --dataset-version uvh26_mv_yolo_v1
python -m src.data.eda
python -m pytest -q
```

Conversion rejects invalid boxes and records a ledger; it never clips or repairs boxes implicitly. All 14 observed classes remain separate. Empty retained labels are background images. Structural errors and unresolved content leakage block conversion. Review rendered annotations before recording a passed manual review in `reports/audit/visual_review.json`.

## Baseline procedure

The proper baseline uses 8,000 training and 2,000 validation images, selected with seed 42 inside the official splits. The full-subset preflight passed before launch. The planned baseline has 30 epochs, image size 640, batch 8 and MPS. Any changes must be documented.

```bash
python -m src.data.build_subset --name smoke_seed42 --train 64 --val 32
python -m src.data.build_subset --name baseline_seed42 --train 8000 --val 2000
python -m src.training.train_baseline \
  --data data/processed/uvh26_mv_yolo_v1/subsets/smoke_seed42/dataset.yaml \
  --name yolov8n_uvh26_mv_smoke_seed42 --smoke
python -m src.training.train_baseline \
  --data data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2/dataset.yaml \
  --name yolov8n_uvh26_mv_baseline_seed42_v1
python -m src.evaluation.evaluate_baseline \
  --weights runs/yolov8n_uvh26_mv_baseline_seed42_v1/weights/best.pt \
  --data data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2/dataset.yaml \
  --name yolov8n_uvh26_mv_validation_seed42
python -m src.inference.predict_image \
  --weights runs/yolov8n_uvh26_mv_baseline_seed42_v1/weights/best.pt \
  --source path/to/image.png --confidence 0.25 --device mps
```

The full-dataset preparation commands are optional for a future full-dataset experiment. The audited subset recovery commands are recorded in `docs/phase_reports/PHASE_1_RECOVERY_COMMANDS.md`. The active baseline is already running; do not launch a duplicate. Runs and subset manifests are never overwritten. `--device cpu` provides a training fallback.

## Baseline results

| Metric | Measured UVH-26 baseline result |
|---|---|
| Precision, recall, F1 | Not yet measured |
| mAP@0.5, mAP@0.5:0.95 | Not yet measured |
| Training duration, best epoch | Not yet measured |
| Inference throughput | Not yet measured for a fine-tuned model |

## Limitations and roadmap

Class imbalance is substantial. Annotation quality and distinctions between similar car types require visual review. Filename and content checks cannot establish camera-level or temporal independence without suitable metadata. Validation throughput must not be described as end-to-end real-time video FPS.

Finish Phase 1 before Phase 2: compare imbalance strategies and consensus variants under controlled evaluation, improve small-object recall, then add class-wise counting and density analytics.

Dataset: [IISc AIM UVH-26](https://huggingface.co/datasets/iisc-aim/UVH-26), CC BY 4.0. Citation and raw-data policy are in `data/README.md`.
