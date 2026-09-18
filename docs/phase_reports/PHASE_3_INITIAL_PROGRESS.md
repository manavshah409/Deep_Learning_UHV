# Phase 3 initial progress — 18 September 2026

## Objective and boundary

Close E2, freeze Phase 3 design/configuration, implement recorded-video detector smoke and synthetic tests, then stop if no suitable evaluation clip exists. This boundary is complete; Phase 3 as a whole is not complete.

## Stage 0: E2 closeout

Finalized intermediate status records using measured evidence: Gate A failed, retain E1 at 640, Gate B not entered, no 960 training. Updated registry/report/README/CHANGELOG/reproduction/dashboard. E2 measurements remain unchanged. All 78 tests passed; artifact validator passed; Streamlit AppTest passed. Commit `7c05701` pushed successfully to origin/master before Phase 3 implementation.

The dated project handover remains a historical snapshot of the state before this closeout; its pending statements are superseded here.

## Stage 1: design/configuration

Completed architecture and configuration, hash-locked checkpoint and 14-class adapter. Defaults MPS with CPU fallback, imgsz640, conf.25, NMS.7, max_det300. ByteTrack and bidirectional bottom-center/hysteresis settings are proposed and recorded, not implemented or measured. Architecture documents tracking/class history, event publication, evaluation and timing boundaries.

## Stage 2: detector-only smoke

Implemented streaming MP4/MOV/AVI inference, immutable run directories, source/model hashes, config snapshots, annotated MP4, per-frame timing CSV, summary JSON, frame/time bounds, explicit no-render mode and failure evidence/handle cleanup. Ordinary tests use generated video and a fake detector, never requiring real weights.

A separate actual-checkpoint MPS smoke used `data/videos/phase3_synthetic_v1.avi`: locally generated white moving rectangle,320×240, 20 frames, 10 FPS. Output `runs/phase3/phase3_synthetic_detector_v1/annotated.mp4` was reopened and all 20 frames decoded at 10 FPS. Source/output videos remain excluded from Git.

Checkpoint SHA256 verified: `9f1045381024445b50e33539791da224b732d61781c73b347013477ebb077fab`.

Measured smoke: 2.2903 seconds processing, 8.7325frames/s; mean 114.473ms,median 17.242ms,p95 140.932ms; no dropped frames, no failed reads. Rendering/encoding enabled, first-frame startup included. Model load/source hashing excluded. This tiny synthetic sample is not sustained road-video performance and does not support real-time claims. Summary is preserved in `reports/audit/phase3_synthetic_smoke.json`.

## Validation

Full suite: 90 passed; Ruff F checks passed. Tests cover generated-video frame/FPS output, immutable run refusal, time windows, frame limits, no-render mode, malformed detections, invalid videos/configuration and checkpoint mismatch. Real-checkpoint smoke exited 0. E0/E1 preservation validator still passes; Phase 3 did not change E2 result files.

## Files and use

New `src/video/`, `configs/phase3_video.yaml`, `configs/phase3_bytetrack.yaml`, `tests/test_video_pipeline.py`, architecture, GT template and this report. Updated README, CHANGELOG and video ignore rules. No source or generated videos, weights or run files are committed.

From repository root, after placing a permitted source clip locally:

```bash
.venv/bin/python -m src.video.pipeline --input-video data/videos/input.mp4 --run-id phase3_user_clip_smoke_v1 --smoke
.venv/bin/python -m pytest -q
```

Use a fresh run ID every time. Config or CLI supports confidence, NMS IoU, device, start/end seconds and max frames. `--no-render` disables drawing and encoding. These are detector-only commands and produce no track or crossing counts.

## Remaining work / exact next action

Supply 2–5 legally usable Indian urban-road clips, stable camera, 30–120seconds each, clear crossing line/direction, multiple vehicle classes; include permission/licence/attribution. No face or number-plate analysis is needed. No suitable local clip was found and none was downloaded. Then implement tracking/counting/analytics/UI stages, manually annotate crossings, measure count errors and benchmark full pipeline with/without encoding. Counting errors, tracking accuracy and road-video latency remain unmeasured. No detector retraining occurred.
