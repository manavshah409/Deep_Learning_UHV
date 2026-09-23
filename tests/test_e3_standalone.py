import pytest

from src.evaluation.e3_metrics import harmonic, measure

NAMES = [str(i) for i in range(14)]


def record(boxes, classes, scores):
    return {
        "image_id": 1,
        "width": 100,
        "height": 100,
        "gt_boxes": [[10, 10, 30, 30]],
        "gt_classes": [0],
        "boxes": boxes,
        "classes": classes,
        "scores": scores,
    }


def test_perfect_metrics_and_macro_f1():
    s, c, matrix, _, errors = measure([record([[10, 10, 30, 30]], [0], [0.9])], NAMES)
    assert s["map50_95"] == pytest.approx(1.0)
    assert s["harmonic_aggregate_f1"] == 1.0 and s["macro_class_f1"] == 1.0
    assert c[0]["gt_count"] == c[0]["prediction_count_fixed"] == 1
    assert (
        c[1]["ap50"] is None and matrix[0, 0] == 1 and errors[0]["class_aware_fn"] == 0
    )


def test_ap_is_separate_from_fixed_threshold():
    s, _, matrix, _, _ = measure([record([[10, 10, 30, 30]], [0], [0.1])], NAMES)
    assert s["map50"] == pytest.approx(1.0) and s["recall"] == 0.0
    assert matrix[0, 14] == 1


def test_duplicate_and_class_confusion_are_explicit():
    _, c, _, _, errors = measure(
        [record([[10, 10, 30, 30]] * 2, [0, 0], [0.9, 0.8])], NAMES
    )
    assert c[0]["tp"] == 1 and c[0]["fp"] == 1
    assert errors[0]["duplicate_predictions"] == [1]
    _, c, matrix, _, errors = measure([record([[10, 10, 30, 30]], [1], [0.9])], NAMES)
    assert c[0]["fn"] == 1 and c[1]["fp"] == 1 and matrix[0, 1] == 1
    assert not errors[0]["confusion_matches"][0]["correct_class"]


def test_empty_predictions_and_small_misses():
    s, _, matrix, _, errors = measure([record([], [], [])], NAMES)
    assert s["map50"] == 0.0 and s["recall"] == 0.0
    assert matrix[0, 14] == 1 and errors[0]["missed_original_small"] == [0]
    assert harmonic(0, 0) == 0.0


def test_delivered_artifacts_validate_without_private_data(monkeypatch):
    from pathlib import Path

    from scripts.validate_e3_stage_d import validate

    original = Path.open

    def guarded(path, *args, **kwargs):
        assert "final_evaluation_1500" not in str(path)
        assert "data/processed" not in str(path)
        assert path.name != "val_manifest.json"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert validate()["passed"]


def test_validator_rejects_inconsistent_f1():
    import csv
    import json

    from scripts.validate_e3_stage_d import REPORT, validate_metrics

    metrics = json.loads((REPORT / "metrics.json").read_text())
    classes = list(csv.DictReader((REPORT / "per_class.csv").open()))
    metrics["harmonic_aggregate_f1"] = 0.999
    with pytest.raises(AssertionError):
        validate_metrics(metrics, classes)
