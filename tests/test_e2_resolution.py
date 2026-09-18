import json
from pathlib import Path
import numpy as np
import pytest
from src.evaluation.e2_size_ap import area_groups
from src.evaluation.evaluate_baseline import publish_bundle


def test_original_area_bins_and_coco_endpoints():
    assert area_groups(100) == ["small"]
    assert area_groups(2000) == ["medium"]
    assert area_groups(10000) == ["large"]
    assert area_groups(1024) == ["small", "medium"]
    assert area_groups(9216) == ["medium", "large"]


def test_resolution_ids_are_mandatory():
    for source in ["evaluate_e2.py", "benchmark_e2.py"]:
        text = Path("src/evaluation", source).read_text()
        assert "Resolution-specific output ID required" in text
        assert "choices=[640, 960]" in text


def test_bundle_hash_integrity_and_no_overwrite(tmp_path):
    from src.data.common import sha256

    p = tmp_path / "E2_640_test"
    publish_bundle(p, {"metric": 0.5}, [{"class_id": 0}], {}, [], [])
    marker = json.loads((p / "COMPLETE.json").read_text())
    assert all(sha256(p / f) == h for f, h in marker["files_sha256"].items())
    with pytest.raises(FileExistsError):
        publish_bundle(p, {}, [], {}, [], [])
    (p / "metrics.json").write_text("{}")
    assert sha256(p / "metrics.json") != marker["files_sha256"]["metrics.json"]


def test_perfect_original_area_ap(tmp_path):
    from PIL import Image
    from src.evaluation.e2_size_ap import evaluate_sizes

    Image.new("RGB", (200, 200)).save(tmp_path / "a.png")
    # 10x10 small, 40x40 medium, 100x100 large, one class each.
    (tmp_path / "labels.txt").write_text(
        "0 .1 .1 .05 .05\n1 .4 .4 .2 .2\n2 .65 .65 .5 .5\n"
    )
    (tmp_path / "val_manifest.json").write_text(
        json.dumps([{"image": "a.png", "image_id": 1, "label": "labels.txt"}])
    )
    pred = [
        dict(image_id=1, file_name="a.png", category_id=i, bbox=box, score=0.99)
        for i, box in enumerate(
            [[15, 15, 10, 10], [60, 60, 40, 40], [80, 80, 100, 100]]
        )
    ]
    r = evaluate_sizes(tmp_path, pred, {0: "a", 1: "b", 2: "c"})
    one_based = [{**p, "category_id": p["category_id"] + 1} for p in pred]
    mapped = evaluate_sizes(tmp_path, one_based, {0: "a", 1: "b", 2: "c"}, [1, 2, 3])
    assert mapped["groups"] == r["groups"]
    for size in ["small", "medium", "large"]:
        assert r["groups"][size]["objects"] == 1
        assert np.isclose(r["groups"][size]["ap50_95"], 1)


def test_protocol_equality_only_allows_resolution_change():
    from src.evaluation.compare_e2 import check_protocol

    keys = [
        "weights_sha256",
        "validation_manifest_sha256",
        "dataset_yaml_sha256",
        "class_mapping_sha256",
        "device",
        "batch",
        "workers",
        "confidence_floor",
        "nms_iou",
        "max_det",
        "source_sha256",
        "size_ap_source_sha256",
        "shared_metric_source_sha256",
        "python",
        "torch",
        "ultralytics",
        "pycocotools",
    ]
    a = {k: "same" for k in keys}
    a["imgsz"] = 640
    b = {**a, "imgsz": 960}
    check_protocol(a, b)
    b["confidence_floor"] = "changed"
    with pytest.raises(ValueError, match="confidence_floor"):
        check_protocol(a, b)
