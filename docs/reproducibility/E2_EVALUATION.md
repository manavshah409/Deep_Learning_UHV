# E2 reproduction

Frozen E1 epoch-22 checkpoint, frozen 8000/2000 subset, MPS, batch8, conf .001, NMS .7, max_det300. Install the pinned auxiliary `requirements-e2-evaluation.txt` in the existing environment. Do not overwrite the completed v2 bundles or rerun the one-time supervisor with existing IDs.

For a deliberate new comparison, use unique IDs containing `_640_` and `_960_` and run sequentially:

```bash
.venv/bin/python -m src.evaluation.evaluate_e2 --weights runs/E1_yolov8s_uvh26_mv_640_seed42/weights/best.pt --data data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2/dataset.yaml --name E2_640_repeat_UNIQUE --imgsz 640 --device mps --batch 8
.venv/bin/python -m src.evaluation.evaluate_e2 --weights runs/E1_yolov8s_uvh26_mv_640_seed42/weights/best.pt --data data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2/dataset.yaml --name E2_960_repeat_UNIQUE --imgsz 960 --device mps --batch 8
.venv/bin/python scripts/validate_e2_artifacts.py
.venv/bin/python -m pytest -q
```

Benchmark module `src.evaluation.benchmark_e2` accepts the same weights/data/device/imgsz/name arguments plus `--count 100 --warmup 10`; omit `--batch` (fixed batch1). New output names are mandatory. Current comparison tooling intentionally binds the final v2 artifacts. Original-area COCO bins, F1 definitions, timing limitations and superseded category-export recovery are documented in the E2 report. No training command is authorized by this failed gate.
