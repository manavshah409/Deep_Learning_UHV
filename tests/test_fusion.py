"""Synthetic bounded fusion, training-fold separation and improvement gates."""

import copy

import numpy as np
import pytest

from src.evaluation.fusion import (
    candidates,
    class_nms,
    fuse,
    improvement_gate,
    make_folds,
    split_fold,
)


def record(boxes=None, scores=None, classes=None, iid=1):
    boxes = boxes if boxes is not None else [[0.0, 0.0, 10.0, 10.0]]
    return {
        "image_id": iid,
        "width": 100,
        "height": 100,
        "gt_boxes": [[0.0, 0.0, 10.0, 10.0]],
        "gt_classes": [0],
        "boxes": boxes,
        "scores": scores if scores is not None else [0.8] * len(boxes),
        "classes": classes if classes is not None else [0] * len(boxes),
    }


def config(kind, **values):
    return next(
        c
        for c in candidates()
        if c["algorithm"] == kind and all(c.get(k) == v for k, v in values.items())
    )


def test_nms_is_class_aware_and_score_ties_stable():
    boxes = [[0, 0, 10, 10]] * 3
    assert class_nms(boxes, [0.9, 0.9, 0.8], [0, 0, 1], 0.45) == [0, 2]


def test_union_obeys_max300_and_class_nms():
    a = record()
    b = record(scores=[0.7])
    output = fuse(a, b, config("union"))
    assert output["scores"] == [0.8]
    output = fuse(a, record(classes=[1]), config("union"))
    assert output["classes"] == [0, 1]


def test_wbf_formula_and_e1_anchor():
    a = record()
    b = record([[1, 1, 11, 11]], [0.6])
    cfg = config("wbf", weights=[0.7, 0.3])
    output = fuse(a, b, cfg)
    delta = 0.3 * 0.6 / (0.7 * 0.8 + 0.3 * 0.6)
    assert output["scores"] == pytest.approx([0.74])
    assert np.allclose(output["boxes"][0], [delta, delta, 10 + delta, 10 + delta])
    assert fuse(a, record([[40, 40, 50, 50]]), cfg)["boxes"] == a["boxes"]


def test_wbf_never_uses_one_e3_box_twice():
    a = record([[0, 0, 10, 10], [1, 1, 11, 11]], [0.9, 0.8])
    b = record(scores=[0.5])
    output = fuse(a, b, config("wbf"))
    assert sorted(output["scores"]) == pytest.approx(
        sorted([0.7 * 0.9 + 0.3 * 0.5, 0.8])
    )


def test_recovery_preserves_e1_and_blocks_any_class_overlap():
    a = record()
    b = record([[0, 0, 10, 10], [30, 30, 40, 40]], [0.9, 0.8], [1, 0])
    output = fuse(
        a, b, config("recovery"), {"eligible_classes": [0, 1], "weak_classes": []}
    )
    assert output["boxes"] == [a["boxes"][0], [30, 30, 40, 40]]
    assert output["classes"] == [0, 0]


def test_recovery_rare_classes_e1_only_and_weak_threshold_stricter():
    a = record()
    b = record([[30, 30, 40, 40]], [0.5], [1])
    cfg = config("recovery")
    assert (
        len(fuse(a, b, cfg, {"eligible_classes": [0], "weak_classes": []})["boxes"])
        == 1
    )
    assert (
        len(fuse(a, b, cfg, {"eligible_classes": [1], "weak_classes": [1]})["boxes"])
        == 1
    )
    assert (
        len(fuse(a, b, cfg, {"eligible_classes": [1], "weak_classes": []})["boxes"])
        == 2
    )


def test_recovery_prevents_added_duplicates():
    a = record()
    b = record([[30, 30, 40, 40]] * 2, [0.9, 0.8], [0, 1])
    output = fuse(
        a, b, config("recovery"), {"eligible_classes": [0, 1], "weak_classes": []}
    )
    assert len(output["scores"]) == 2


def test_recovery_never_evicts_e1_at_cap():
    a = record([[0, 0, 10, 10]] * 300, [0.1] * 300)
    output = fuse(
        a,
        record([[30, 30, 40, 40]]),
        config("recovery"),
        {"eligible_classes": [0], "weak_classes": []},
    )
    assert len(output["scores"]) == 300 and output["boxes"] == a["boxes"]


def test_fusion_rejects_invalid_or_unpaired_input():
    a = record()
    b = record(scores=[float("nan")])
    with pytest.raises(ValueError):
        fuse(a, b, config("union"))
    with pytest.raises(ValueError):
        fuse(a, record(iid=2), config("union"))
    with pytest.raises(ValueError):
        fuse(a, a, {"algorithm": "union", "id": "invented", "iou": 0.9})


def test_fold_partition_deterministic_disjoint_complete():
    ids = list(range(500))
    folds = make_folds(ids)
    assert folds == make_folds(ids[::-1]) and all(len(f) == 100 for f in folds)
    rows = [record(iid=i) for i in ids]
    for index in range(5):
        train, test = split_fold(rows, folds, index)
        assert len(train) == 400 and len(test) == 100
        assert not {r["image_id"] for r in train} & {r["image_id"] for r in test}
    broken = copy.deepcopy(folds)
    broken[0][0] = broken[1][0]
    with pytest.raises(ValueError):
        split_fold(rows, broken, 0)


def test_parameter_selection_sees_only_training_labels(monkeypatch):
    import scripts.stage_f_fusion as runner

    rows = [record(iid=i) for i in range(5)]

    def curve(records, grid):
        assert {r["image_id"] for r in records} == {0, 1, 2, 3}
        return [{"threshold": 0.4, "macro_class_f1": 0.5}]

    monkeypatch.setattr(runner, "threshold_curve", curve)
    selected = runner.choose(
        "baseline", [0, 1, 2, 3], {"E1": rows, "E3": rows}, None, {}, [], 0
    )
    assert selected["threshold"] == 0.4


def test_gate_requires_distributed_improvement_and_duplicate_safety():
    baseline = {
        "map50_95": 0.5,
        "precision": 0.6,
        "recall": 0.6,
        "macro_class_f1": 0.5,
        "predictions_fixed": 1000,
    }
    other = {**baseline, "macro_class_f1": 0.53}
    bc = [{"gt_count": 100, "f1": 0.5, "ap50_95": 0.5} for _ in range(14)]
    cc = [{**c, "f1": 0.53} for c in bc]
    assert improvement_gate(
        baseline, other, bc, cc, {"duplicates": 10}, {"duplicates": 10}
    )["passed"]
    assert not improvement_gate(
        baseline, other, bc, cc, {"duplicates": 10}, {"duplicates": 100}
    )["passed"]
    cc = copy.deepcopy(bc)
    cc[0]["f1"] = 0.99
    assert not improvement_gate(
        baseline, other, bc, cc, {"duplicates": 10}, {"duplicates": 10}
    )["passed"]


def test_bounded_grid_and_no_input_mutation():
    assert len(candidates()) == 19
    a = record()
    original = copy.deepcopy(a)
    for cfg in candidates():
        assert fuse(a, a, cfg, {"eligible_classes": [0], "weak_classes": []}) == fuse(
            a, a, cfg, {"eligible_classes": [0], "weak_classes": []}
        )
    assert a == original


def test_oof_thresholds_do_not_filter_ap_input():
    from scripts.stage_f_fusion import mixed_evaluation

    rows = [record(scores=[0.3], iid=1), record(scores=[0.3], iid=2)]
    summary, classes, *_ = mixed_evaluation(rows, {1: 0.2, 2: 0.4})
    assert summary["map50_95"] == pytest.approx(1.0)
    assert summary["recall"] == 0.5 and summary["precision"] == 1.0
    assert (
        classes[0]["prediction_count_raw"] == 2
        and classes[0]["prediction_count_fixed"] == 1
    )


def test_stage_f_delivered_artifacts_and_boundary(monkeypatch):
    from pathlib import Path

    from scripts.validate_stage_f import validate

    original = Path.open

    def guarded(path, *args, **kwargs):
        assert "final_evaluation_1500" not in str(path)
        assert path.name != "val_manifest.json"
        assert "data/processed" not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert validate()["passed"]
