"""Streaming detector-only video smoke. Tracking/counting are later stages."""
import argparse
import csv
import json
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.data.common import ROOT, sha256

EXPECTED_SHA = "9f1045381024445b50e33539791da224b732d61781c73b347013477ebb077fab"
CLASS_NAMES = ["Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", "Two-wheeler", "LCV", "Mini-bus", "Tempo-traveller", "Bicycle", "Van", "Others"]


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_config(cfg):
    if cfg["checkpoint_sha256"] != EXPECTED_SHA or cfg["imgsz"] != 640:
        raise ValueError("Phase 3 requires frozen E1 checkpoint and imgsz640")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg["run_id"]):
        raise ValueError("Run ID must be a simple unique name")
    if Path(cfg["output_video"]).name != cfg["output_video"] or not cfg["output_video"].endswith('.mp4'):
        raise ValueError("output_video must be an MP4 basename")
    for key in ["confidence", "nms_iou"]:
        if not 0 <= cfg[key] <= 1:
            raise ValueError(f"Invalid {key}")
    if cfg['max_det'] < 1 or cfg['device'] not in ['mps', 'cpu']:
        raise ValueError('Invalid device or max_det')
    if cfg['start_seconds'] < 0 or (cfg.get('end_seconds') is not None and cfg['end_seconds'] <= cfg['start_seconds']):
        raise ValueError('Invalid time interval')
    if cfg.get('max_frames') is not None and cfg['max_frames'] < 1:
        raise ValueError('max_frames must be positive')


class FrozenDetector:
    def __init__(self, cfg):
        import torch
        from ultralytics import YOLO
        checkpoint = ROOT / cfg['checkpoint']
        if not checkpoint.is_file():
            raise FileNotFoundError('Frozen E1 checkpoint is missing')
        if sha256(checkpoint) != EXPECTED_SHA:
            raise ValueError('Frozen checkpoint SHA-256 mismatch')
        self.cfg = cfg
        self.device = cfg['device']
        self.warnings = []
        if self.device == 'mps' and not torch.backends.mps.is_available():
            if not cfg['cpu_fallback']:
                raise RuntimeError('MPS unavailable and CPU fallback disabled')
            self.device = 'cpu'
            self.warnings.append('MPS unavailable; explicit CPU fallback used')
        torch.manual_seed(cfg['seed'])
        self.model = YOLO(checkpoint)
        if [self.model.names[i] for i in range(14)] != CLASS_NAMES:
            raise ValueError('Frozen 14-class mapping mismatch')

    def sync(self):
        if self.device == 'mps':
            import torch
            torch.mps.synchronize()

    def __call__(self, frame):
        result = self.model.predict(frame, imgsz=640, conf=self.cfg['confidence'], iou=self.cfg['nms_iou'], max_det=self.cfg['max_det'], device=self.device, verbose=False)[0]
        return result.boxes.data.cpu().numpy()


def process_video(cfg, detector_factory=FrozenDetector):
    """Injection is for tests; CLI always uses the hash-verified frozen detector."""
    validate_config(cfg)
    source = ROOT / cfg['input_video']
    if source.suffix.lower() not in ['.mp4', '.mov', '.avi'] or not source.is_file():
        raise ValueError('Input must be an existing MP4, MOV or AVI')
    run = ROOT / cfg['output_root'] / cfg['run_id']
    run.mkdir(parents=True, exist_ok=False)
    cap = writer = None
    timings = []
    warnings = []
    failed = 0
    try:
        atomic_json(run / 'config.json', cfg)
        input_hash = sha256(source)
        detector = detector_factory(cfg)
        warnings.extend(detector.warnings)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError('Video decoder could not open input')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
            raise ValueError('Invalid FPS or dimensions')
        start = int(cfg['start_seconds'] * fps)
        if start >= total and total > 0:
            raise ValueError('Start time exceeds video length')
        # Decode sequentially to preserve frame numbering; no approximate codec seeks.
        for _ in range(start):
            if not cap.grab():
                raise ValueError('Could not decode to requested start')
        if cfg['render']:
            writer = cv2.VideoWriter(str(run / cfg['output_video']), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError('MP4 encoder unavailable')
        frame_number = start
        wall_start = time.perf_counter()
        with (run / 'frame_timings.csv').open('w', newline='') as f:
            csv_out = csv.DictWriter(f, fieldnames=['frame', 'timestamp_seconds', 'decode_ms', 'detector_ms', 'render_encode_ms', 'total_ms', 'detections'])
            csv_out.writeheader()
            while True:
                if cfg.get('max_frames') and len(timings) >= cfg['max_frames']:
                    break
                if cfg.get('end_seconds') is not None and frame_number / fps >= cfg['end_seconds']:
                    break
                begin = time.perf_counter()
                ok, frame = cap.read()
                decoded = time.perf_counter()
                if not ok:
                    if total > 0 and frame_number < total:
                        failed += 1
                        warnings.append('Premature decoder termination; unread remainder not processed')
                    break
                if frame is None or frame.shape[:2] != (height, width):
                    raise ValueError('Malformed or changing-size video frame')
                detector.sync()
                infer_start = time.perf_counter()
                boxes = np.asarray(detector(frame))
                detector.sync()
                inferred = time.perf_counter()
                if boxes.ndim != 2 or boxes.shape[1] != 6 or not np.isfinite(boxes).all():
                    raise ValueError('Malformed detector output; expected finite Nx6')
                for x1,y1,x2,y2,conf,cls in boxes:
                    if cls != int(cls) or not 0 <= cls < 14 or not 0 <= conf <= 1 or x2 < x1 or y2 < y1:
                        raise ValueError('Invalid detection geometry/class/confidence')
                    if writer:
                        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,220,0),1)
                        cv2.putText(frame,f'{CLASS_NAMES[int(cls)]} {conf:.2f}',(int(x1),max(12,int(y1))),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,220,0),1)
                if writer:
                    writer.write(frame)
                end = time.perf_counter()
                row = dict(frame=frame_number,timestamp_seconds=frame_number/fps,decode_ms=1000*(decoded-begin),detector_ms=1000*(inferred-infer_start),render_encode_ms=1000*(end-inferred),total_ms=1000*(end-begin),detections=len(boxes))
                csv_out.writerow(row)
                timings.append(row['total_ms'])
                frame_number += 1
        if writer:
            writer.release()
            writer = None
        elapsed = time.perf_counter() - wall_start
        if not timings:
            raise ValueError('No frames processed')
        summary = dict(status='partial_decode_failure' if failed else 'complete',scope='detector_only_smoke_no_tracking_or_counting',input_sha256=input_hash,checkpoint_sha256=cfg['checkpoint_sha256'],detector_backend=type(detector).__name__,checkpoint_verified=isinstance(detector,FrozenDetector),device=detector.device,source_fps=fps,output_fps=fps if cfg['render'] else None,source_frame_count=total,processed_frame_count=len(timings),source_duration_seconds=total/fps,processed_duration_seconds=len(timings)/fps,resolution=[width,height],processing_seconds=elapsed,processing_fps=len(timings)/elapsed,latency_ms=dict(mean=float(np.mean(timings)),median=float(np.median(timings)),p95=float(np.percentile(timings,95))),failed_reads=failed,dropped_frames=0,unprocessed_frames_after_decode_failure=max(0,total-frame_number) if failed else 0,warnings=warnings,encoding_included=cfg['render'],timing_note='Decode through detection/render/encode; wall includes writer finalization. Model loading and source hashing excluded. No warm-up excluded: first-frame startup included. Detector time includes preprocessing and postprocessing, not inference-only.')
        atomic_json(run/'summary.json',summary)
        if failed:
            raise RuntimeError('Premature decode failure; partial evidence preserved')
        return run,summary
    except Exception as error:
        atomic_json(run/'FAILURE.json',dict(error_type=type(error).__name__,message=str(error),processed_frames=len(timings)))
        raise
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/phase3_video.yaml')
    parser.add_argument('--input-video')
    parser.add_argument('--run-id')
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--no-render',action='store_true')
    parser.add_argument('--confidence',type=float)
    parser.add_argument('--nms-iou',type=float)
    parser.add_argument('--device',choices=['mps','cpu'])
    parser.add_argument('--start-seconds',type=float)
    parser.add_argument('--end-seconds',type=float)
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    for key in ['input_video','run_id','max_frames','confidence','nms_iou','device','start_seconds','end_seconds']:
        if getattr(args,key) is not None:
            cfg[key]=getattr(args,key)
    if args.smoke:
        cfg['max_frames']=min(cfg.get('max_frames') or 30,30)
    if args.no_render:
        cfg['render']=False
    run,summary=process_video(cfg)
    print(json.dumps({'run':str(run.relative_to(ROOT)),**summary},indent=2))


if __name__ == '__main__':
    main()
