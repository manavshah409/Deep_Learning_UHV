"""Dashboard contracts: no weights or datasets needed for evidence pages."""

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from dashboard import services as s


def test_artifact_missing_malformed_and_forbidden(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/bad.json").write_text("{")
    assert s.artifact("reports/missing.json") is None
    assert s.artifact("reports/bad.json") is None
    assert s.artifact("../secret.json") is None
    assert s.artifact("data/anything.json") is None
    (tmp_path / "reports/good.json").write_text('{"ok":true}')
    assert s.artifact("reports/good.json") == {"ok": True}


def test_checkpoint_verification(tmp_path):
    p = tmp_path / "test.pt"
    with pytest.raises(ValueError):
        s.verify_checkpoint(p)
    p.write_bytes(b"test")
    with pytest.raises(ValueError):
        s.verify_checkpoint(p)
    assert s.verify_checkpoint(p, hashlib.sha256(b"test").hexdigest())


@pytest.mark.parametrize(
    "n,expected",
    [(0, "Low"), (5, "Low"), (6, "Moderate"), (15, "Moderate"), (16, "High")],
)
def test_density(n, expected):
    assert s.density(n) == expected


def test_density_invalid():
    with pytest.raises(ValueError):
        s.density(3, 10, 5)


def test_counts_exports():
    row = {
        "class_id": 1,
        "name": "Sedan",
        "confidence": 0.7,
        "x1": 1,
        "y1": 2,
        "x2": 3,
        "y2": 4,
    }
    assert s.counts([row, row]) == {"Sedan": 2}
    csv, data = s.exports([row])
    assert "Sedan" in csv and json.loads(data) == [row]
    assert s.counts([]) == {}


@pytest.mark.parametrize(
    "name,data",
    [
        ("bad.png", b"broken"),
        ("bad.exe", b"broken"),
        ("large.jpg", b"x" * (10 * 1024**2 + 1)),
    ],
)
def test_reject_image(name, data):
    with pytest.raises(ValueError):
        s.decode_image(data, name)


def test_image_decoding():
    f = io.BytesIO()
    Image.new("RGB", (12, 12)).save(f, format="PNG")
    assert s.decode_image(f.getvalue(), "../../name.png").size == (12, 12)


def test_video_and_device():
    with pytest.raises(ValueError):
        s.validate_video("bad.txt", 10)
    with pytest.raises(ValueError):
        s.validate_video("huge.mp4", 101 * 1024**2)
    s.validate_video("clip.MOV", 10)
    assert s.safe_device("mps", False) == "cpu"
    assert s.safe_device("mps", True) == "mps"
    assert s.safe_device("cpu", True) == "cpu"


@pytest.mark.parametrize(
    "page",
    [
        "Project Overview",
        "Image Detection",
        "Recorded Video Detection",
        "Traffic Analytics",
        "Model Comparison",
        "Training Analysis",
        "Dataset Insights",
        "Documentation and Reproducibility",
    ],
)
def test_evidence_pages(page, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved mode attempted model loading")

    monkeypatch.setattr(s, "make_detector", forbidden)
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard/app.py"),
        default_timeout=20,
    ).run()
    app.sidebar.radio[1].set_value(page).run()
    assert not app.exception
    assert not app.error
    assert app.sidebar.radio[0].value == "Saved Evidence Mode"


def test_startup_without_artifacts_or_weights(tmp_path, monkeypatch):
    from dashboard import pages

    monkeypatch.setattr(s, "ROOT", tmp_path)
    monkeypatch.setattr(pages, "ROOT", tmp_path)
    pages.load.clear()
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard/app.py")
    ).run()
    assert not app.exception and not app.error
    app.sidebar.radio[1].set_value("Image Detection").run()
    app.sidebar.radio[0].set_value("Live Inference Mode").run()
    assert any("absent" in w.value for w in app.warning)
    pages.load.clear()


def test_historical_metrics_match_requested_protocol():
    m = s.artifact(
        "reports/evaluations/yolov8s_uvh26_mv_e1_validation_seed42_v2/metrics.json"
    )
    assert m["map50_95"] == pytest.approx(0.524760, abs=1e-6)
    m = s.artifact("reports/comparisons/E1_E3_stageE_v2/E1_metrics.json")
    assert m["precision"] == pytest.approx(0.614771, abs=1e-6)
