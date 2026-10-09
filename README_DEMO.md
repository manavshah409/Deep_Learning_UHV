# Indian Urban Traffic Analytics System

A faculty-facing Streamlit application using verified UVH-26 research evidence and the selected E1 YOLOv8s / 640 checkpoint. The historical root `app.py` showcase is preserved. This is a research prototype, not a production or live-video system.

## Quick start

From the repository or extracted demonstration package:

```bash
python3.12 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install -r requirements-demo.txt
streamlit run dashboard/app.py
```

Saved Evidence Mode is the default. It requires no PyTorch, model weights or datasets. All eight pages open; Image Detection displays saved evidence and an explicit media-unavailable state, Traffic Analytics waits for session predictions, and Recorded Video displays the existing synthetic smoke summary with a planned integration notice.

For optional local image inference, use the existing project `.venv` with its training dependencies and dashboard packages:

```bash
.venv/bin/python -m streamlit run dashboard/app.py
```

Place the existing verified checkpoint at `runs/E1_yolov8s_uvh26_mv_640_seed42/weights/best.pt`. Do not download a replacement model automatically. Expected SHA-256:

`9f1045381024445b50e33539791da224b732d61781c73b347013477ebb077fab`

Select Live Inference Mode, upload your own permitted JPG/PNG, set thresholds and click Detect vehicles. CPU is available; unavailable MPS falls back to CPU. A corrupt or different checkpoint is refused. Missing optional inference dependencies show an unavailable message.

## Faculty sequence

1. Overview: explain Indian heterogeneous traffic, audit scope and E1 selection.
2. Model Comparison: show the separate validation2000 and calibration500 tables; explain the unmet precision/recall target.
3. Training Analysis: show convergence and E3 epoch 13 selection.
4. Dataset Insights: discuss imbalance, small objects and annotation limitations.
5. Optional Image Detection with your own permitted image; inspect boxes and export CSV/JSON.
6. Traffic Analytics: explain visible detections and the prototype density rule.
7. Recorded Video: distinguish the saved synthetic smoke from road-video validation.
8. Documentation: explain reproducibility and remaining work.

Preflight on presentation day: start Saved Evidence Mode offline; inspect every page; if demonstrating inference, verify weights and run a permitted image before the session. No permitted dataset photographs or annotated photo outputs are included in this package. Existing aggregate plots are included; no arbitrary media were downloaded.

## Troubleshooting and limitations

- Missing/corrupt artifacts display Not available; restore the original small report files. Never fabricate replacements.
- If inference fails on MPS, choose CPU. Model loading is lazy and cached; image outputs remain session-local. Shared model calls are locked.
- Uploaded filenames are never used as filesystem paths. Image bytes are capped at 10 MB and decoded images at 20 MP. Restart/close the session to clear memory.
- Timing is a measured single request, not a throughput benchmark; excludes model load, hash checks and browser/network time.
- Video upload integration is deliberately disabled: current CLI lacks UI progress, persisted per-class detections and validated browser codec support. Tracking/counting are not added.
- 70–80% precision/recall has not been achieved. Fusion feasibility was rejected; E1 remains selected.
- Never commit checkpoints, uploads, datasets, generated videos, run directories, caches or logs. Portable ZIP lives under ignored `deliverables/`.

Detailed evidence: `docs/phase_reports/FACULTY_STREAMLIT_DELIVERABLE.md`.

## New data evaluation workflow

Use the eight-page navigation beginning with Executive Overview. New Data Evaluation accepts images alone, YOLO image/label ZIPs or images plus annotations.json. Faculty Presentation Mode uses saved evidence without weights. See [preparation and CLI instructions](docs/reproducibility/NEW_DATA_EVALUATION.md) and [technical protocol](docs/phase_reports/NEW_DATA_EVALUATION_DASHBOARD.md). The older portable ZIP predates this extension; use the repository checkout for the new workflow.
