"""Prepare a separately identified H264 browser copy, never mutate a measured run."""

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.common import sha256
from src.video.pipeline import atomic_json


def main():
    benchmark = json.loads(
        (ROOT / "reports/phase3b/benchmark_summary.json").read_text()
    )
    summary = next(x for x in benchmark["runs"] if x["mode"] == "complete")
    source = ROOT / "runs/phase3b" / summary["run_id"] / "annotated.mp4"
    marker = json.loads((source.parent / "COMPLETE.json").read_text())
    assert sha256(source) == marker["files"]["annotated.mp4"]
    directory = ROOT / "runs/phase3b_previews" / summary["run_id"]
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "annotated_browser.mp4"
    cap = cv2.VideoCapture(str(source))
    writer = None
    try:
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"avc1"),
            cap.get(5),
            (int(cap.get(3)), int(cap.get(4))),
        )
        if not writer.isOpened():
            raise ValueError("H264 encoder unavailable; use original local video")
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            n += 1
        writer.release()
        writer = None
        if n != summary["frames"]:
            raise ValueError("Preview frame count mismatch")
        info = {
            "run_id": summary["run_id"],
            "protocol_sha256": summary["protocol_sha256"],
            "filename": str(output.relative_to(ROOT)),
            "sha256": sha256(output),
            "frames": n,
            "codec": "avc1",
            "source_sha256": sha256(source),
            "timing_note": "Presentation-only transcode after benchmark; not included in measured pipeline FPS",
            "licence": "Adapted Gpkp Commons footage, CC BY-SA 4.0; boxes, IDs and predicted counts added; no audio.",
        }
        atomic_json(directory / "COMPLETE.json", info)
        with (ROOT / "reports/phase3b/video_preview.json").open("x") as f:
            json.dump(info, f, indent=2)
        print("Browser preview validated:", n, "frames")
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
