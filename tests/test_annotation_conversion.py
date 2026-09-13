import json
import pytest
from src.data.common import (
    load_coco,
    unique_index,
    category_mapping,
    convert_box,
    validate_line,
)


def test_coco_normalization():
    assert convert_box([10, 20, 30, 40], 100, 200) == pytest.approx(
        (0.25, 0.2, 0.3, 0.2)
    )


def test_whole_image():
    assert convert_box([0, 0, 100, 200], 100, 200) == (0.5, 0.5, 1, 1)


def test_mapping_preserves_names_sorted():
    assert category_mapping(
        [{"id": 9, "name": "Others"}, {"id": 2, "name": "Sedan"}]
    ) == [
        {"original_id": 2, "original_name": "Sedan", "yolo_id": 0, "name": "Sedan"},
        {"original_id": 9, "original_name": "Others", "yolo_id": 1, "name": "Others"},
    ]


def test_empty_annotations(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"images": [], "annotations": [], "categories": []}))
    assert load_coco(path)["annotations"] == []


def test_missing_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    with pytest.raises(ValueError):
        load_coco(path)


def test_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{")
    with pytest.raises(json.JSONDecodeError):
        load_coco(path)


def test_duplicate_image_ids():
    with pytest.raises(ValueError):
        unique_index([{"id": 1}, {"id": 1}])


def test_image_index():
    assert unique_index([{"id": 4, "file_name": "a.png"}])[4]["file_name"] == "a.png"


def test_duplicate_categories():
    with pytest.raises(ValueError):
        category_mapping([{"id": 1, "name": "a"}, {"id": 1, "name": "b"}])


def test_unknown_yolo_class():
    with pytest.raises(ValueError):
        validate_line("14 .5 .5 .2 .2", 14)
