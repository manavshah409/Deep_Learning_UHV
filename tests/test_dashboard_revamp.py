"""User journeys and filter/export consistency, using synthetic records only."""

import copy
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from dashboard.new_data import analysis
from dashboard.new_data.explorer import overlay, select_rows

APP = str(Path(__file__).resolve().parents[1] / "dashboard/app.py")


def sample():
    r = {
        "image_id": 1,
        "name": "image.png",
        "width": 100,
        "height": 100,
        "gt_boxes": [[10, 10, 30, 30]],
        "gt_classes": [0],
        "boxes": [[10, 10, 30, 30], [10, 10, 30, 30], [70, 70, 80, 80]],
        "classes": [0, 0, 1],
        "scores": [0.9, 0.8, 0.2],
    }
    errors = analysis.fixed(r)[3]
    for e in errors:
        e["image_id"] = 1
    return r, errors


def test_filter_matches_overlay_and_export():
    r, e = sample()
    rows = select_rows(r, e, kind="duplicate detection")
    assert len(rows) == 1 and rows[0]["prediction"] == 1
    assert rows[0]["class_name"] == "Hatchback"
    im = overlay(
        Image.new("RGB", (100, 100), "white"), r, rows, "Predictions only", True
    )
    assert im.getpixel((10, 25)) != (255, 255, 255)
    assert im.getpixel((70, 75)) == (255, 255, 255)
    assert not select_rows(r, e, kind="duplicate detection", confidence=(0.85, 1.0))
    assert not select_rows(r, e, size="large")
    assert not select_rows(r, e, iou=(0.0, 0.4))


def test_unlabeled_no_correctness_labels():
    r, _ = sample()
    rows = select_rows(r, confidence=(0.0, 1.0))
    assert len(rows) == 3 and all(x["kind"] == "prediction" for x in rows)
    assert len(select_rows(r, cls="Sedan", confidence=(0.0, 1.0))) == 1


def test_missed_survives_confidence_filter():
    r, _ = sample()
    r["scores"] = [0.1, 0.1, 0.1]
    errors = analysis.fixed(r)[3]
    for e in errors:
        e["image_id"] = 1
    rows = select_rows(r, errors, kind="missed object", confidence=(0.9, 1.0))
    assert len(rows) == 1 and rows[0]["prediction"] is None


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_latency_rejected(value):
    row = {
        "preprocessing_ms": 1,
        "inference_ms": 2,
        "postprocessing_ms": 1,
        "end_to_end_ms": value,
    }
    with pytest.raises(ValueError):
        analysis.latency([row])


def test_empty_latency_rejected():
    with pytest.raises(ValueError):
        analysis.latency([])


def test_navigation_call_to_action():
    app = AppTest.from_file(APP).run()
    next(b for b in app.button if b.label == "Evaluate new images").click().run()
    assert app.session_state["navigation"] == "New Data Evaluation"
    assert app.session_state["application_mode"] == "Live Inference Mode"
    assert not app.exception
    assert next(b for b in app.button if b.label == "Validate dataset").disabled
    assert not any(b.label == "Run predictions" for b in app.button)


def test_faculty_guided_navigation():
    app = AppTest.from_file(APP).run()
    app.sidebar.radio[0].set_value("Faculty Presentation Mode").run()
    next(b for b in app.button if b.label.startswith("Next:")).click().run()
    assert app.session_state["navigation"] == "Historical Model Comparison"
    assert not app.exception and not app.error


def test_synthetic_inference_does_not_mutate_prepared_records(monkeypatch):
    import sys
    from types import SimpleNamespace

    import numpy as np

    monkeypatch.setitem(
        sys.modules, "ultralytics", SimpleNamespace(__version__="synthetic-test")
    )
    from dashboard.new_data import runner

    class Boxes:
        @property
        def data(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([[10, 10, 30, 30, 0.9, 0]])

    class Prediction:
        def __init__(self):
            self.boxes = Boxes()
            self.speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 1.0}

    class Detector:
        device = "cpu"

        def __init__(self):
            self.model = self

        def sync(self):
            pass

        def __call__(self, _):
            pass

        def predict(self, *args, **kwargs):
            return [Prediction()]

    monkeypatch.setattr(runner, "verify_checkpoint", lambda *args: None)
    monkeypatch.setattr(runner, "make_detector", lambda *args: (Detector(), None))
    r, _ = sample()
    r = {k: v for k, v in r.items() if k not in ["boxes", "scores", "classes"]}
    original = copy.deepcopy(r)
    updates = []
    result = runner.run(
        {"image.png": Image.new("RGB", (100, 100))},
        [r],
        {},
        {"format": "unlabeled", "names": analysis.NAMES},
        bootstrap=0,
        progress=lambda v, t: updates.append((v, t)),
    )
    assert r == original and result["records"][0]["scores"] == [0.9]
    assert updates[0][0] == 0 and updates[-1][0] == 1


def test_guided_upload_validates_before_model_and_invalidates_edits(
    monkeypatch, tmp_path
):
    import io

    import streamlit as st

    from dashboard.new_data import views

    class Upload(io.BytesIO):
        def __init__(self, name, data):
            super().__init__(data)
            self.name = name
            self.size = len(data)

    buf = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(buf, format="PNG")
    uploads = [
        Upload("image.png", buf.getvalue()),
        Upload("image.txt", b"0 .2 .2 .2 .2"),
    ]
    monkeypatch.setattr(
        st,
        "file_uploader",
        lambda label, *a, **k: uploads if label == "Images and annotations" else None,
    )
    monkeypatch.setattr(views, "ROOT", tmp_path)
    checkpoint = tmp_path / views.CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic presence only")
    calls = []

    def forbidden(*a, **k):
        calls.append(1)
        raise ValueError("Synthetic inference failure")

    monkeypatch.setattr(views.runner, "run", forbidden)
    app = AppTest.from_file(APP).run()
    app.sidebar.radio[0].set_value("Live Inference Mode").run()
    app.sidebar.radio[1].set_value("New Data Evaluation").run()
    next(x for x in app.selectbox if x.label == "Annotation format").set_value(
        "yolo"
    ).run()
    for label in ["Dataset name", "Dataset source", "Licence or permission"]:
        next(x for x in app.text_input if x.label == label).set_value("Synthetic test")
    next(
        x for x in app.selectbox if x.label == "Annotations created independently?"
    ).set_value("Yes")
    next(
        x for x in app.selectbox if x.label == "Used in training or tuning?"
    ).set_value("No")
    app.checkbox[0].check().run()
    next(b for b in app.button if b.label == "Validate dataset").click().run()
    assert not app.exception and not app.error and not calls
    assert app.session_state["prepared_data"]["records"][0]["gt_classes"] == [0]
    # Failure must preserve the last successful result, not silently replace it.
    next(b for b in app.button if b.label == "Run evaluation").click().run()
    assert calls == [1] and not app.exception
    assert any("Synthetic inference failure" in e.value for e in app.error)
    next(x for x in app.text_input if x.label == "Dataset name").set_value(
        "Changed input"
    ).run()
    assert "prepared_data" not in app.session_state
    assert not any(b.label == "Run evaluation" for b in app.button)
