"""Human-only annotations, immutable revisions and strict evaluation eligibility."""

import csv
import io
import json
import math
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.data.common import sha256
from src.video.pipeline import CLASS_NAMES, atomic_json

DIRECTIONS = ["A_to_B", "B_to_A"]
FIELDS = [
    "event_id",
    "video_id",
    "frame",
    "timestamp_seconds",
    "class_id",
    "class_name",
    "direction",
    "review_status",
    "annotator",
    "uncertain",
    "notes",
]


def validate(document, protocol, final=False):
    if document.get("protocol_sha256") != protocol["sha256"]:
        raise ValueError("Annotation protocol mismatch")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", document.get("annotator", "")):
        raise ValueError("Use an anonymous reviewer code of 1–32 letters/digits/_/-")
    ids = set()
    video = protocol["payload"]["video"]
    for event in document["events"]:
        if not event["event_id"] or event["event_id"] in ids:
            raise ValueError("Duplicate or missing event ID")
        ids.add(event["event_id"])
        if event["video_id"] != video["video_id"]:
            raise ValueError("Wrong annotation video")
        frame = event["frame"]
        if type(frame) is not int or not 0 <= frame < video["frames"]:
            raise ValueError("Invalid annotation frame")
        timestamp = event["timestamp_seconds"]
        if not math.isfinite(timestamp) or abs(timestamp - frame / video["fps"]) > 1e-6:
            raise ValueError("Frame/timestamp mismatch")
        cls = event["class_id"]
        if (
            type(cls) is not int
            or not 0 <= cls < 14
            or event["class_name"] != CLASS_NAMES[cls]
        ):
            raise ValueError("Invalid annotation class")
        if event["direction"] not in DIRECTIONS or type(event["uncertain"]) is not bool:
            raise ValueError("Invalid direction or uncertainty")
        if event["annotator"] != document["annotator"] or event[
            "review_status"
        ] not in ["draft", "reviewed"]:
            raise ValueError("Invalid event review identity/status")
        if not isinstance(event["notes"], str) or len(event["notes"]) > 2000:
            raise ValueError("Invalid notes")
    if final:
        if (
            document.get("human_review_complete") is not True
            or document.get("whole_video_reviewed") is not True
        ):
            raise ValueError("Final human review of the whole video is required")
        if document.get("predictions_seen_before_annotation") is not False:
            raise ValueError(
                "Independent annotation must precede prediction inspection"
            )
        if any(
            e["uncertain"] or e["review_status"] != "reviewed"
            for e in document["events"]
        ):
            raise ValueError("Resolve uncertain/draft events before final evaluation")
        if not document["events"] and document.get("confirmed_zero_events") is not True:
            raise ValueError("An empty template is not reviewed zero ground truth")
    return document


def as_csv(document):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(document["events"])
    return out.getvalue()


def save_revision(parent, document, protocol):
    validate(document, protocol, final=document.get("human_review_complete", False))
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    # Unique directories preserve earlier work even across interrupted writes.
    dest = parent / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12]
    )
    dest.mkdir(exist_ok=False)
    atomic_json(dest / "annotations.json", document)
    (dest / "annotations.csv").write_text(as_csv(document))
    atomic_json(
        dest / "COMPLETE.json",
        {
            "files": {
                n: sha256(dest / n) for n in ["annotations.json", "annotations.csv"]
            }
        },
    )
    return dest


def load_revision(path, protocol):
    path = Path(path)
    marker = json.loads((path / "COMPLETE.json").read_text())
    for name in ["annotations.json", "annotations.csv"]:
        if sha256(path / name) != marker["files"].get(name):
            raise ValueError("Annotation revision integrity failure")
    document = json.loads((path / "annotations.json").read_text())
    return validate(
        document, protocol, final=document.get("human_review_complete", False)
    )
