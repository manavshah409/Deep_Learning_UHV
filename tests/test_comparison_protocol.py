"""Synthetic comparison contracts; no reserved dataset reads."""

import json

import numpy as np
import pytest

from src.evaluation import comparison_protocol as p
from src.evaluation.common_metrics import measure
from src.evaluation.complementarity import (
    paired_analysis,
    select_threshold,
    threshold_curve,
)
from src.evaluation.e3_metrics import measure as stage_d_measure


def record(scores=None):
    return {
        "image_id": 1,
        "width": 100,
        "height": 100,
        "gt_boxes": [[10, 10, 30, 30]],
        "gt_classes": [0],
        "boxes": [[10, 10, 30, 30], [10, 10, 30, 30]],
        "classes": [0, 0],
        "scores": scores or [0.8, 0.2],
    }


def test_shared_class_mapping_and_background():
    for cls in range(14):
        assert p.vehicle_class(cls, "E1") == p.vehicle_class(cls + 1, "E3") == cls
    assert p.vehicle_class(0, "E3") is None
    assert p.detection([1, 1, 2, 2], 0.5, 0, "E3", 10, 10) is None
    for cls, model in [(14, "E1"), (-1, "E1"), (15, "E3"), (0.5, "E1")]:
        with pytest.raises(ValueError):
            p.vehicle_class(cls, model)


def test_coordinates_clip_convert_and_reject_degenerate():
    d = p.detection([-2, 3, 12, 9], 0.8, 1, "E3", 10, 10)
    assert d["class_id"] == 0 and d["class_name"] == "Hatchback"
    assert d["xyxy"] == [0, 3, 10, 9] and d["xywh"] == [0, 3, 10, 6] and d["clipped"]
    with pytest.raises(ValueError):
        p.detection([12, 2, 15, 5], 0.9, 0, "E1", 10, 10)


@pytest.mark.parametrize(
    "box,score",
    [
        ([0, 0, float("nan"), 2], 0.9),
        ([0, 0, 2, 2], float("inf")),
        ([2, 2, 1, 1], 0.5),
        ([0, 0, 1, 1], 1.1),
    ],
)
def test_invalid_predictions_fail(box, score):
    with pytest.raises(ValueError):
        p.detection(box, score, 0, "E1", 10, 10)


def test_guard_rejects_before_opening_reserved(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "CALIBRATION", tmp_path / "calibration.json")
    reserved = tmp_path / "reserved.json"
    monkeypatch.setattr(p, "RESERVED", reserved)

    def forbidden(*args, **kwargs):
        raise AssertionError("Must not hash or open denied manifest")

    monkeypatch.setattr(p, "sha", forbidden)
    with pytest.raises(PermissionError):
        p.load_manifest(reserved)
    with pytest.raises(PermissionError):
        p.load_manifest(tmp_path / "renamed.json")
    with pytest.raises(PermissionError):
        p.guard_path(
            tmp_path / "reserved.jpg",
            tmp_path,
            [{"image": "cal.jpg", "label": "cal.txt"}],
        )


def test_guard_authorization_is_explicit_and_hash_checked(monkeypatch, tmp_path):
    reserved = tmp_path / "reserved_synthetic.json"
    reserved.write_text("[]")
    monkeypatch.setattr(p, "RESERVED", reserved)
    monkeypatch.setattr(p, "RESERVED_SHA", p.sha(reserved))
    assert p.load_manifest(reserved, authorize_reserved_evaluation=True) == []
    reserved.write_text("[{}]")
    with pytest.raises(ValueError):
        p.load_manifest(reserved, authorize_reserved_evaluation=True)


def test_protocol_seal_rejects_tampering():
    value = p.seal({"threshold": 0.5})
    assert p.verify_seal(value) == {"threshold": 0.5}
    value["payload"]["threshold"] = 0.1
    with pytest.raises(ValueError):
        p.verify_seal(value)


def test_threshold_tie_is_deterministic_higher():
    curve = [
        {"threshold": 0.2, "macro_class_f1": 0.5},
        {"threshold": 0.4, "macro_class_f1": 0.5},
    ]
    assert select_threshold(curve) == select_threshold(curve[::-1]) == 0.4


def test_threshold_curve_matches_full_evaluator():
    rows = [record()]
    for point in threshold_curve(rows, [0.001, 0.2, 0.5, 0.9]):
        m, *_ = measure(rows, p.NAMES, point["threshold"])
        assert point["macro_class_f1"] == pytest.approx(m["macro_class_f1"])
        assert point["precision"] == pytest.approx(m["precision"])


def test_evaluator_parity_with_stage_d():
    rows = [record()]
    a, ac, am, *_ = measure(rows, p.NAMES, 0.25)
    b, bc, bm, *_ = stage_d_measure(rows, p.NAMES)
    for metric in [
        "map50",
        "map50_95",
        "precision",
        "recall",
        "harmonic_aggregate_f1",
        "macro_class_f1",
    ]:
        assert a[metric] == pytest.approx(b[metric])
    assert ac == bc and np.array_equal(am, bm)


def test_complementarity_exact_object_partition():
    a = record()
    b = record([0.1, 0.05])
    result, _ = paired_analysis([a], [b], {"E1": 0.5, "E3": 0.5})
    assert result["counts"] == {"both": 0, "E1_only": 1, "E3_only": 0, "neither": 0}


def test_prediction_bundle_integrity_and_schema(tmp_path):
    cfg = {"preprocessing": {"kind": "synthetic"}}
    row = {
        "schema_version": p.SCHEMA_VERSION,
        "image_id": 1,
        "image_key": "x.jpg",
        "model_id": "E1",
        "checkpoint_sha256": "abc",
        "preprocessing": cfg["preprocessing"],
        "inference_configuration": cfg,
        "width": 10,
        "height": 10,
        "detections": [p.detection([1, 1, 4, 4], 0.9, 0, "E1", 10, 10)],
    }
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(row) + "\n")
    receipt = {
        "model_id": "E1",
        "checkpoint_sha256": "abc",
        "inference_configuration": cfg,
        "prediction_sha256": p.sha(path),
    }
    assert p.validate_bundle(path, receipt, [1])
    with pytest.raises(ValueError):
        p.validate_bundle(path, receipt, [2])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        p.validate_bundle(path, receipt, [1])


def test_unclipped_mapping_matches_library_for_intersecting_boxes():
    import subprocess
    import sys

    import torch

    raw = torch.tensor(
        [[100.0, 20.0, 200.0, 200.0], [447.8783, 372.1635, 598.4469, 385.2216]]
    )
    mapped = p.unletterbox_boxes(raw, (384, 640), (1080, 1920))
    d = p.detection(mapped[0].tolist(), 0.8, 0, "E1", 1920, 1080)
    # Import Ultralytics in an isolated process: it globally patches PIL.Image.open.
    code = "import json,torch; from ultralytics.utils.ops import scale_boxes; print(json.dumps(scale_boxes((384,640),torch.tensor([[100.,20.,200.,200.]]),(1080,1920)).tolist()))"
    expected = json.loads(
        subprocess.check_output([sys.executable, "-c", code], text=True)
    )
    assert np.allclose(d["xyxy"], expected[0])
    assert mapped[1, 3] > mapped[1, 1] > 1080
    with pytest.raises(p.OutsideImage):
        p.detection(mapped[1].tolist(), 0.00136, 4, "E1", 1920, 1080)
    # Truly malformed raw boxes are never classified as harmless padding.
    with pytest.raises(ValueError) as error:
        p.detection([0, 2, 3, 2], 0.5, 0, "E1", 10, 10)
    assert not isinstance(error.value, p.OutsideImage)


def test_frozen_comparison_artifacts_without_reserved_or_private_reads(monkeypatch):
    from pathlib import Path

    from scripts.validate_stage_e import validate

    original = Path.open

    def guarded(path, *args, **kwargs):
        assert "final_evaluation_1500" not in str(path)
        assert path.name != "val_manifest.json"
        assert "data/processed" not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert validate()["passed"]
