import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from src.evaluation.evaluate_baseline import extract_metrics, publish_bundle


@pytest.fixture(autouse=True)
def isolate_ultralytics_pillow_import():
    # Ultralytics globally patches Image.open on import. Keep its optional HEIF
    # fallback out of unrelated corrupt-PNG tests; do not install dependencies.
    from PIL import Image

    original = Image.open
    yield
    Image.open = original


def metrics():
    b = SimpleNamespace(
        ap_class_index=np.array([0]),
        p=np.array([0.5]),
        r=np.array([0.4]),
        f1=np.array([4 / 9]),
        ap50=np.array([0.6]),
        ap=np.array([0.3]),
        mp=0.5,
        mr=0.4,
        map50=0.6,
        map=0.3,
        px=np.linspace(0, 1, 1000),
        f1_curve=np.ones((1, 1000)) * 0.4,
    )
    return SimpleNamespace(box=b)


def test_returned_metrics_need_no_model_or_validator():
    result, rows = extract_metrics(metrics(), {0: "vehicle"})
    assert result["f1"] == pytest.approx(4 / 9)
    assert rows[0]["ap50"] == 0.6


def test_missing_or_nonfinite_metrics_fail():
    with pytest.raises(ValueError, match="unavailable"):
        extract_metrics(SimpleNamespace(), {0: "vehicle"})
    m = metrics()
    m.box.mp = float("nan")
    with pytest.raises(ValueError, match="finite"):
        extract_metrics(m, {0: "vehicle"})


def test_source_never_accesses_model_validator():
    tree = ast.parse(Path("src/evaluation/evaluate_baseline.py").read_text())
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr == "validator"
    ]
    assert any(
        isinstance(n, ast.keyword) and n.arg == "validator" for n in ast.walk(tree)
    )


def test_failed_export_never_publishes_complete_bundle(tmp_path):
    target = tmp_path / "evaluation"
    with pytest.raises(ValueError):
        publish_bundle(target, {"value": float("nan")}, [], {}, [], [])
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def test_atomic_bundle_and_no_overwrite(tmp_path):
    target = tmp_path / "evaluation"
    publish_bundle(target, {"value": 0.5}, [{"class_id": 0}], {}, [], [])
    assert (target / "COMPLETE.json").exists()
    with pytest.raises(FileExistsError):
        publish_bundle(target, {}, [], {}, [], [])
