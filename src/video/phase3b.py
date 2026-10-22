"""Versioned protocol, four pipeline measurements and human-gated closeout."""

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.data.common import ROOT, sha256
from src.video.counting import LineCounter, export_events
from src.video.phase3b_annotations import load_revision
from src.video.phase3b_evaluation import evaluate
from src.video.pipeline import CLASS_NAMES, EXPECTED_SHA, FrozenDetector, atomic_json
from src.video.tracking import VehicleTracker

DEFAULT_PROTOCOL = "reports/phase3b/protocol_v2.json"
MODES = ["detection", "tracking", "counting", "complete"]


def digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def load_protocol(path=DEFAULT_PROTOCOL):
    value = json.loads(Path(path).read_text())
    if value["sha256"] != digest(value["payload"]):
        raise ValueError("Protocol hash mismatch")
    return value


def metadata(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Video cannot be opened")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height, frames = [int(cap.get(k)) for k in [3, 4, 7]]
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        if not np.isfinite(fps) or fps <= 0 or min(width, height, frames) <= 0:
            raise ValueError("Invalid video metadata")
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise ValueError("Video resolution changed")
            decoded += 1
        if decoded != frames:
            raise ValueError("Decoded frame count mismatch")
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frames": frames,
            "duration_seconds": frames / fps,
            "codec": "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)),
        }
    finally:
        cap.release()


def freeze(path, device="mps", line=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    cfg = yaml.safe_load((ROOT / "configs/phase3a_bangalore01.yaml").read_text())
    source = ROOT / cfg["input_video"]
    if (
        sha256(source) != cfg["input_sha256"]
        or sha256(ROOT / cfg["checkpoint"]) != EXPECTED_SHA
    ):
        raise ValueError("Frozen video or checkpoint changed")
    counting = yaml.safe_load((ROOT / "configs/phase3_counting.yaml").read_text())
    if line is not None:
        points = np.asarray(line, float).reshape(2, 2)
        if (
            not np.isfinite(points).all()
            or np.any(points < 0)
            or np.any(points > 1)
            or np.array_equal(points[0], points[1])
        ):
            raise ValueError("Invalid normalized endpoints")
        counting["line_normalized"] = points.tolist()
    cfg.update(device=device, cpu_fallback=False)
    payload = {
        "schema": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "video": dict(
            filename=cfg["input_video"],
            video_id=cfg["video_id"],
            sha256=cfg["input_sha256"],
            **metadata(source),
            permission="Gpkp, Ramana Maharshi Road Bangalore 2025 01, CC BY-SA 4.0; excerpt, no audio, re-encoded; retain attribution and share-alike for adapted footage.",
            source_url="https://commons.wikimedia.org/wiki/File:Ramana_Maharshi_Road,_Bangalore_(2025)_01.webm",
            camera="Mostly stable elevated handheld view; minor movement, night glare; limited fixed-image-line pilot",
        ),
        "detector": {
            k: cfg[k]
            for k in [
                "checkpoint",
                "checkpoint_sha256",
                "imgsz",
                "confidence",
                "nms_iou",
                "max_det",
                "device",
                "cpu_fallback",
                "seed",
            ]
        },
        "tracker": yaml.safe_load((ROOT / "configs/phase3_tracking.yaml").read_text()),
        "counting": counting,
        "class_names": CLASS_NAMES,
        "directions": {
            "A_to_B": "negative to positive signed side; downwards for left-to-right horizontal line",
            "B_to_A": "positive to negative; upwards for this line",
        },
        "matching": {
            "rule": "Global closest absolute timestamp pair first, same video/class/direction, one-to-one; ties GT timestamp then GT ID then prediction ID. Not maximum-cardinality matching.",
            "tolerance_seconds": 0.5,
            "timestamp": "frame/fps of first observed point beyond opposite dead-band; interpolation diagnostic only",
        },
        "timing": {
            "warmup_iterations": 10,
            "warmup_excluded": True,
            "measured_frames": "all 900, decoded anew for every mode",
            "stages": "decode, internal preprocess/inference/postprocess, detector outer call, tracking, counting plus evidence publication, rendering, encoding, table writes",
            "wall": "After startup/warmup/encoder initialization through all frames, encoder finalization and detailed artifact export; summary/completion hashing separately outside processing wall.",
            "realtime_target": "Complete pipeline effective FPS >= source FPS, zero dropped/failed frames; no live capture/display or GUI network included",
        },
        "acceptance": {
            "event_f1_minimum": 0.8,
            "wape_percent_maximum": 20,
            "status": "preregistered engineering pilot targets, not deployment certification",
            "human_review": "all frames reviewed before predictions; all uncertainty resolved; zero-event confirmation if empty",
            "real_time_fps": 30,
            "no_failed_or_dropped_frames": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {
                p: version(p)
                for p in [
                    "torch",
                    "torchvision",
                    "ultralytics",
                    "opencv-python",
                    "streamlit",
                ]
            },
        },
        "launch_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "sources": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / "src/video").glob("*.py"))
            if p.name != "evaluate_events.py"
        },
        "configs": {
            p: sha256(ROOT / p)
            for p in ["configs/phase3_tracking.yaml", "configs/phase3_counting.yaml"]
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump({"payload": payload, "sha256": digest(payload)}, f, indent=2)
    return load_protocol(path)


def stats(values):
    return {
        k: float(v)
        for k, v in zip(
            ["mean", "median", "p90", "p95"],
            [
                np.mean(values),
                np.median(values),
                np.percentile(values, 90),
                np.percentile(values, 95),
            ],
        )
    }


class TimedDetector(FrozenDetector):
    def predict(self, frame):
        result = self.model.predict(
            frame,
            imgsz=640,
            conf=self.cfg["confidence"],
            iou=self.cfg["nms_iou"],
            max_det=300,
            device=self.device,
            verbose=False,
            quantize=32,
        )[0]
        self.sync()
        return result.boxes.data.cpu().numpy(), result.speed


def run(protocol_path, run_id, mode, limit=None, detector_factory=TimedDetector):
    if mode not in MODES or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid mode/run ID")
    protocol = load_protocol(protocol_path)
    p = protocol["payload"]
    source = ROOT / p["video"]["filename"]
    if sha256(source) != p["video"]["sha256"]:
        raise ValueError("Video changed")
    for name, expected in p["sources"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("Source changed after protocol freeze: " + name)
    dest = ROOT / "runs/phase3b" / run_id
    dest.mkdir(parents=True, exist_ok=False)
    cap = writer = None
    rows, tracks_rows = [], []
    start = time.perf_counter()
    try:
        atomic_json(dest / "protocol.json", protocol)
        detector = detector_factory(p["detector"])
        cap = cv2.VideoCapture(str(source))
        ok, first = cap.read()
        if not ok:
            raise ValueError("Cannot decode warmup frame")
        load_seconds = time.perf_counter() - start
        warm = time.perf_counter()
        for _ in range(p["timing"]["warmup_iterations"]):
            detector.predict(first)
        detector.sync()
        warm_seconds = time.perf_counter() - warm
        cap.release()
        cap = cv2.VideoCapture(str(source))
        v, cc = p["video"], p["counting"]
        tracker = (
            VehicleTracker(
                p["tracker"], cc["maximum_absence_frames"], reidentify_retired=True
            )
            if mode != "detection"
            else None
        )
        endpoints = np.array(cc["line_normalized"]) * [v["width"], v["height"]]
        counter = LineCounter(
            endpoints,
            minimum_age=cc["minimum_confirmed_age"],
            hysteresis=cc["hysteresis_pixels"],
            maximum_gap=cc["maximum_crossing_gap_frames"],
            minimum_displacement=cc["minimum_displacement_pixels"],
        )
        identity = {
            "run_id": run_id,
            "video_id": v["video_id"],
            "video_sha256": v["sha256"],
            "line_id": "L1",
            "tracker": "ByteTrack",
            "checkpoint_sha256": EXPECTED_SHA,
        }
        if mode == "complete":
            writer = cv2.VideoWriter(
                str(dest / "annotated.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                v["fps"],
                (v["width"], v["height"]),
            )
            if not writer.isOpened():
                raise ValueError("Encoder failed")
        setup_seconds = time.perf_counter() - start - load_seconds - warm_seconds
        (dest / "evidence").mkdir()
        count = min(limit or v["frames"], v["frames"])
        begin = time.perf_counter()
        for frame_id in range(count):
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok or frame.shape[:2] != (v["height"], v["width"]):
                raise ValueError(f"Decode failure frame {frame_id}")
            detector.sync()
            detections, speed = detector.predict(frame)
            detector.sync()
            if (
                detections.ndim != 2
                or detections.shape[1] != 6
                or not np.isfinite(detections).all()
            ):
                raise ValueError("Malformed detections")
            t2 = time.perf_counter()
            observed = (
                tracker.update(detections, frame_id, (v["height"], v["width"]))
                if tracker
                else []
            )
            t3 = time.perf_counter()
            if mode in ["counting", "complete"]:
                for track in observed:
                    event = counter.update(track, frame_id, v["fps"], identity)
                    if event and not cv2.imwrite(
                        str(dest / "evidence" / f"{event['event_id']}.jpg"), frame
                    ):
                        raise ValueError("Evidence write failed")
            t4 = time.perf_counter()
            if writer:
                cv2.line(
                    frame,
                    tuple(endpoints[0].astype(int)),
                    tuple(endpoints[1].astype(int)),
                    (0, 0, 255),
                    3,
                )
                for track in observed:
                    x1, y1, x2, y2 = map(int, track.box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(
                        frame,
                        f"{track.track_id} {CLASS_NAMES[track.class_id]}",
                        (x1, max(20, y1)),
                        0,
                        0.55,
                        (0, 220, 0),
                        1,
                    )
                cv2.putText(
                    frame,
                    f"Predicted crossings {len(counter.events)} | UNREVIEWED",
                    (15, 30),
                    0,
                    0.65,
                    (0, 255, 255),
                    2,
                )
            t5 = time.perf_counter()
            if writer:
                writer.write(frame)
            t6 = time.perf_counter()
            observed_ids = {t.track_id for t in observed}
            if tracker:
                for track in tracker.active.values():
                    tracks_rows.append(
                        {
                            "frame": frame_id,
                            "timestamp_seconds": frame_id / v["fps"],
                            "track_id": track.track_id,
                            "class_id": track.class_id,
                            "class_name": CLASS_NAMES[track.class_id],
                            "confidence": track.confidence,
                            "bbox": track.box,
                            "bottom_center": track.anchor,
                            "state": "observed"
                            if track.track_id in observed_ids
                            else "lost_last_observed_geometry",
                        }
                    )
            rows.append(
                {
                    "frame": frame_id,
                    "timestamp_seconds": frame_id / v["fps"],
                    "detections": len(detections),
                    "active_tracks": len(observed),
                    "cumulative_crossings": len(counter.events),
                    "decode_ms": 1000 * (t1 - t0),
                    "detector_ms": 1000 * (t2 - t1),
                    "preprocessing_ms": speed["preprocess"],
                    "inference_ms": speed["inference"],
                    "postprocessing_ms": speed["postprocess"],
                    "tracking_ms": 1000 * (t3 - t2),
                    "counting_publication_ms": 1000 * (t4 - t3),
                    "rendering_ms": 1000 * (t5 - t4),
                    "encoding_ms": 1000 * (t6 - t5),
                    "total_ms": 1000 * (time.perf_counter() - t0),
                }
            )
        finalize_start = time.perf_counter()
        if writer:
            writer.release()
            writer = None
        encoding_finalize_seconds = time.perf_counter() - finalize_start
        export_events(dest, counter.events)

        def csv_file(name, values, fields):
            with (dest / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(values)

        csv_file("frame_timings.csv", rows, list(rows[0]))
        csv_file(
            "per_frame_tracks.csv",
            tracks_rows,
            [
                "frame",
                "timestamp_seconds",
                "track_id",
                "class_id",
                "class_name",
                "confidence",
                "bbox",
                "bottom_center",
                "state",
            ],
        )
        summaries = []
        for t in tracker.all_tracks() if tracker else []:
            summaries.append(
                {
                    "track_id": t.track_id,
                    "first_frame": t.first_frame,
                    "last_frame": t.last_frame,
                    "duration_seconds": (t.last_frame - t.first_frame + 1) / v["fps"],
                    "observations": t.observations,
                    "final_class": t.class_id,
                    "history": t.history,
                    "class_scores": t.scores,
                    "counted_directions": sorted(t.counted),
                }
            )
        atomic_json(dest / "tracks.json", summaries)
        elapsed = time.perf_counter() - begin
        class_counts = Counter(e["class_name"] for e in counter.events)
        direction_counts = Counter(e["direction"] for e in counter.events)
        summary = {
            "status": "processing_complete_human_review_pending",
            "run_id": run_id,
            "mode": mode,
            "protocol_sha256": protocol["sha256"],
            "actual_checkpoint_verified": isinstance(detector, TimedDetector),
            "device": detector.device,
            "frames": count,
            "source_fps": v["fps"],
            "full_clip": count == v["frames"],
            "dropped_frames": 0,
            "failed_frames": 0,
            "model_load_and_first_decode_seconds": load_seconds,
            "warmup_seconds": warm_seconds,
            "setup_seconds": setup_seconds,
            "processing_wall_seconds": elapsed,
            "encoding_finalize_seconds": encoding_finalize_seconds,
            "effective_fps": count / elapsed,
            "realtime_target_pass": mode == "complete"
            and count == v["frames"]
            and count / elapsed >= v["fps"],
            "achieved_source_fps_ratio": (count / elapsed) / v["fps"],
            "latency_ms": {
                k: stats([r[k] for r in rows]) for k in rows[0] if k.endswith("_ms")
            },
            "predicted_crossings": len(counter.events),
            "detection_instances": sum(r["detections"] for r in rows),
            "maximum_observed_active_tracks": max(r["active_tracks"] for r in rows),
            "class_counts": {c: class_counts[c] for c in CLASS_NAMES},
            "direction_counts": {c: direction_counts[c] for c in ["A_to_B", "B_to_A"]},
            "class_direction": [
                {
                    "class_name": c,
                    "direction": di,
                    "count": sum(
                        e["class_name"] == c and e["direction"] == di
                        for e in counter.events
                    ),
                }
                for c in CLASS_NAMES
                for di in ["A_to_B", "B_to_A"]
            ],
            "identity_transitions": tracker.identity_transitions if tracker else [],
            "track_id_policy": "New public ID after maximum absence; raw ByteTrack revival is recorded",
            "ground_truth_reviewed": False,
            "counting_metrics": None,
            "timing_boundary": p["timing"],
        }
        atomic_json(dest / "summary.json", summary)
        atomic_json(
            dest / "COMPLETE.json",
            {
                "files": {
                    str(f.relative_to(dest)): sha256(f)
                    for f in dest.rglob("*")
                    if f.is_file()
                }
            },
        )
        return summary
    except Exception as e:
        atomic_json(
            dest / "FAILURE.json",
            {
                "error": type(e).__name__,
                "message": str(e),
                "completed_frames": len(rows),
            },
        )
        raise
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


def evaluate_run(protocol_path, run_id, revision, output):
    protocol = load_protocol(protocol_path)
    document = load_revision(revision, protocol)
    run_dir = ROOT / "runs/phase3b" / run_id
    complete = json.loads((run_dir / "COMPLETE.json").read_text())
    for name in ["crossing_events.json", "summary.json", "protocol.json"]:
        if sha256(run_dir / name) != complete["files"][name]:
            raise ValueError("Run integrity failure")
    summary = json.loads((run_dir / "summary.json").read_text())
    if (
        not summary["full_clip"]
        or not summary["actual_checkpoint_verified"]
        or summary["mode"] != "complete"
        or summary["protocol_sha256"] != protocol["sha256"]
    ):
        raise ValueError(
            "Evaluation requires matching full actual-checkpoint complete-pipeline run"
        )
    result = evaluate(
        json.loads((run_dir / "crossing_events.json").read_text()), document, protocol
    )
    result.update(
        protocol_sha256=protocol["sha256"],
        annotation_sha256=sha256(Path(revision) / "annotations.json"),
        run_id=run_id,
    )
    result["acceptance_pass"] = (
        result["total"]["f1"] >= protocol["payload"]["acceptance"]["event_f1_minimum"]
        and result["weighted_absolute_percentage_error"] is not None
        and result["weighted_absolute_percentage_error"]
        <= protocol["payload"]["acceptance"]["wape_percent_maximum"]
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "metrics.json", result)
    atomic_json(output / "COMPLETE.json", {"sha256": sha256(output / "metrics.json")})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run", "evaluate"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--line", type=float, nargs=4)
    parser.add_argument("--mode", choices=MODES, default="complete")
    parser.add_argument("--run-id")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--annotations")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze(args.protocol, args.device, args.line)
        print(result["sha256"])
    elif args.action == "run":
        print(json.dumps(run(args.protocol, args.run_id, args.mode, args.max_frames)))
    else:
        print(
            json.dumps(
                evaluate_run(args.protocol, args.run_id, args.annotations, args.output)[
                    "total"
                ]
            )
        )


if __name__ == "__main__":
    main()
