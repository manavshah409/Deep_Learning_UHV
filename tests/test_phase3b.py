import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.video import phase3b
from src.video.counting import LineCounter
from src.video.phase3b_annotations import load_revision, save_revision, validate
from src.video.phase3b_evaluation import counts, evaluate
from src.video.pipeline import CLASS_NAMES
from src.video.tracking import Track, VehicleTracker


def protocol():
    payload = {
        "video": {"video_id": "test", "frames": 30, "fps": 10, "duration_seconds": 3},
        "matching": {"tolerance_seconds": 0.5},
    }
    return {"payload": payload, "sha256": phase3b.digest(payload)}


def event(i=1, frame=10, cls=7, direction="A_to_B"):
    return {
        "event_id": str(i),
        "video_id": "test",
        "frame": frame,
        "timestamp_seconds": frame / 10,
        "class_id": cls,
        "class_name": CLASS_NAMES[cls],
        "direction": direction,
        "annotator": "reviewer01",
        "uncertain": False,
        "notes": "",
        "review_status": "reviewed",
    }


def document(events=None):
    return {
        "protocol_sha256": protocol()["sha256"],
        "annotator": "reviewer01",
        "events": events if events is not None else [event()],
        "human_review_complete": True,
        "whole_video_reviewed": True,
        "predictions_seen_before_annotation": False,
        "confirmed_zero_events": False,
    }


def test_nearest_pair_and_one_to_one():
    r = evaluate([event(1, 6), event(2, 9), event(3, 12)], document(), protocol())
    assert r["matches"][0]["prediction_index"] == 1
    assert (r["total"]["tp"], r["total"]["fp"], r["total"]["fn"]) == (1, 2, 0)
    assert r["crossing_time_mae_seconds"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "wrong", [{"class_id": 6}, {"direction": "B_to_A"}, {"timestamp_seconds": 2.0}]
)
def test_wrong_class_direction_time(wrong):
    e = event()
    e.update(wrong)
    assert evaluate([e], document(), protocol())["total"]["tp"] == 0


def test_ties_and_tolerance():
    r = evaluate([event("b", 15), event("a", 5)], document(), protocol())
    assert len(r["matches"]) == 1 and r["matches"][0]["prediction_index"] == 1


def test_zero_groups_and_errors():
    d = document([])
    d["confirmed_zero_events"] = True
    r = evaluate([event()], d, protocol())
    assert (
        r["total"]["percentage_error"] is None
        and r["weighted_absolute_percentage_error"] is None
    )
    assert r["mae_28_class_direction_groups"] == pytest.approx(1 / 28)
    assert counts(8, 10, 7)["signed_error"] == -2
    assert counts(8, 10, 7)["undercount_rate"] == 0.2
    assert counts(12, 10, 7)["overcount_rate"] == 0.2


@pytest.mark.parametrize(
    "field,value",
    [
        ("human_review_complete", False),
        ("whole_video_reviewed", False),
        ("predictions_seen_before_annotation", True),
    ],
)
def test_review_gate(field, value):
    d = document()
    d[field] = value
    with pytest.raises(ValueError):
        evaluate([], d, protocol())


def test_uncertainty_empty_and_protocol_gate():
    for d in [document([]), document()]:
        if d["events"]:
            d["events"][0]["uncertain"] = True
        with pytest.raises(ValueError):
            validate(d, protocol(), True)
    d = document()
    d["protocol_sha256"] = "wrong"
    with pytest.raises(ValueError):
        validate(d, protocol())


def test_revision_recovery_tampering(tmp_path):
    a = save_revision(tmp_path, document(), protocol())
    b = save_revision(tmp_path, document(), protocol())
    assert a != b and load_revision(a, protocol())["events"]
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    (interrupted / "annotations.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        load_revision(interrupted, protocol())
    (b / "annotations.json").write_text("{}")
    with pytest.raises(ValueError):
        load_revision(b, protocol())
    assert load_revision(a, protocol())["human_review_complete"]


def test_track_history_and_retired_id():
    track = Track(1, 0, 0)
    track.observe(0, [0, 0, 10, 20], 0.8, 7)
    assert track.anchor == [5, 20] and track.history[0]["observed_class"] == 7
    t = VehicleTracker(
        {
            "track_buffer": 1,
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.25,
            "match_thresh": 0.8,
            "fuse_score": True,
        },
        1,
    )
    row = np.array([[0, 0, 10, 20, 1, 0.8, 7, 0]])
    t.accept(row, 0)
    t.accept(np.empty((0, 8)), 2)
    with pytest.raises(ValueError, match="reused"):
        t.accept(row, 3)


def test_displacement_and_jitter():
    c = LineCounter(
        [[0, 0], [10, 0]], minimum_age=1, hysteresis=1, minimum_displacement=10
    )
    t = Track(1, 0, 0)
    for f, y in enumerate([-3, 0, 3, -3, 3]):
        t.observe(f, [0, y - 2, 4, y], 0.8, 7)
        c.update(t, f, 10, {"run_id": "test", "video_id": "test"})
    assert not c.events


def synthetic_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(phase3b, "ROOT", tmp_path)
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    for _ in range(4):
        writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    p = {
        "video": dict(
            video_id="test",
            filename="source.avi",
            sha256=phase3b.sha256(source),
            **phase3b.metadata(source),
        ),
        "detector": {},
        "sources": {},
        "timing": {"warmup_iterations": 1},
        "counting": {
            "maximum_absence_frames": 3,
            "line_normalized": [[0, 0.5], [1, 0.5]],
            "minimum_confirmed_age": 1,
            "hysteresis_pixels": 2,
            "maximum_crossing_gap_frames": 1,
            "minimum_displacement_pixels": 4,
        },
        "tracker": {
            "track_buffer": 3,
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.25,
            "match_thresh": 0.8,
            "fuse_score": True,
        },
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"payload": p, "sha256": phase3b.digest(p)}))
    return path


class FakeDetector:
    device = "cpu"

    def __init__(self, cfg):
        pass

    def sync(self):
        pass

    def predict(self, frame):
        return np.array([[5, 5, 20, 20, 0.9, 7]], float), {
            "preprocess": 0.1,
            "inference": 0.2,
            "postprocess": 0.1,
        }


@pytest.mark.parametrize("mode", phase3b.MODES)
def test_pipeline_exports_and_immutable(tmp_path, monkeypatch, mode):
    path = synthetic_protocol(tmp_path, monkeypatch)
    r = phase3b.run(path, "test", mode, detector_factory=FakeDetector)
    assert r["frames"] == 4 and r["effective_fps"] > 0 and r["counting_metrics"] is None
    run = tmp_path / "runs/phase3b/test"
    assert (run / "COMPLETE.json").exists() and (run / "per_frame_tracks.csv").exists()
    with pytest.raises(FileExistsError):
        phase3b.run(path, "test", mode, detector_factory=FakeDetector)


def test_malformed_detection_failure_record(tmp_path, monkeypatch):
    path = synthetic_protocol(tmp_path, monkeypatch)

    class Bad(FakeDetector):
        def predict(self, frame):
            return np.full((1, 6), np.nan), {}

    with pytest.raises(ValueError, match="Malformed"):
        phase3b.run(path, "bad", "complete", detector_factory=Bad)
    assert (tmp_path / "runs/phase3b/bad/FAILURE.json").exists()
    assert not (tmp_path / "runs/phase3b/bad/COMPLETE.json").exists()


def test_invalid_video(tmp_path):
    p = tmp_path / "bad.mp4"
    p.write_bytes(b"bad")
    with pytest.raises(ValueError):
        phase3b.metadata(p)


def test_protocol_tamper(tmp_path):
    p = tmp_path / "p.json"
    v = protocol()
    v["payload"]["video"]["fps"] = 99
    p.write_text(json.dumps(v))
    with pytest.raises(ValueError):
        phase3b.load_protocol(p)


def test_saved_empty_dashboard_app(tmp_path, monkeypatch):
    from dashboard import video_analytics

    monkeypatch.setattr(video_analytics, "ROOT", tmp_path)
    app = AppTest.from_string(
        "from dashboard.video_analytics import saved_evidence\nsaved_evidence()"
    ).run(timeout=20)
    assert not app.exception


def test_dashboard_navigation(tmp_path, monkeypatch):
    from dashboard import video_analytics

    monkeypatch.setattr(video_analytics, "ROOT", tmp_path)
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard/app.py")
    ).run(timeout=20)
    app.radio(key="navigation").set_value("Video Traffic Analytics").run(timeout=20)
    assert not app.exception and any(
        x.value == "Video Traffic Analytics" for x in app.header
    )


def test_reidentified_late_reappearance_is_logged():
    t = VehicleTracker(
        {
            "track_buffer": 1,
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.25,
            "match_thresh": 0.8,
            "fuse_score": True,
        },
        1,
        reidentify_retired=True,
    )
    row = np.array([[0, 0, 10, 20, 1, 0.8, 7, 0]])
    first = t.accept(row, 0)[0]
    t.accept(np.empty((0, 8)), 2)
    second = t.accept(row, 3)[0]
    assert first.track_id != second.track_id
    assert t.identity_transitions[0]["previous_public_id"] == first.track_id


def test_saved_evidence_populated_without_road_data(tmp_path, monkeypatch):
    from dashboard import video_analytics

    monkeypatch.setattr(video_analytics, "ROOT", tmp_path)
    report = tmp_path / "reports/phase3b"
    report.mkdir(parents=True)
    row = {
        "mode": "complete",
        "frames": 4,
        "device": "cpu",
        "effective_fps": 10.0,
        "latency_ms": {"total_ms": {"p95": 100.0}},
        "detection_instances": 8,
        "maximum_observed_active_tracks": 2,
        "predicted_crossings": 1,
        "source_fps": 10.0,
        "realtime_target_pass": True,
        "class_direction": [{"class_name": "Sedan", "direction": "A_to_B", "count": 1}],
        "direction_counts": {"A_to_B": 1, "B_to_A": 0},
        "run_id": "synthetic",
    }
    (report / "benchmark_summary.json").write_text(
        json.dumps({"protocol_sha256": "test", "runs": [row]})
    )
    app = AppTest.from_string(
        "from dashboard.video_analytics import saved_evidence\nsaved_evidence()"
    ).run(timeout=20)
    assert not app.exception and any("pending" in x.value for x in app.warning)


def test_annotation_interface_synthetic_only(tmp_path, monkeypatch):
    from dashboard import video_analytics

    monkeypatch.setattr(video_analytics, "ROOT", tmp_path)
    path = synthetic_protocol(tmp_path, monkeypatch)
    p = json.loads(path.read_text())
    # Render from a known small synthetic clip, with no model or road data.
    script = (
        "from dashboard.video_analytics import annotation_tool\nannotation_tool("
        + repr(p)
        + ")"
    )
    app = AppTest.from_string(script).run(timeout=20)
    assert not app.exception
    assert any(b.label == "Finalize human review as a new revision" for b in app.button)
