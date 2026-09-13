import json
import pytest
from PIL import Image
from src.data.validate_raw import audit
from src.data import convert_to_yolo as conversion
from src.data.common import validate_line


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    for directory in ("configs", "reports/audit", "reports/tables", "data/processed"):
        (tmp_path / directory).mkdir(parents=True)
    config = tmp_path / "configs/paths.local.yaml"
    config.write_text("raw: data/raw\nprocessed: data/processed\nreports: reports\n")
    for split, color in [("train", "red"), ("val", "blue")]:
        base = tmp_path / "data/raw" / f"UVH-26-{split.title()}"
        (base / "data/000").mkdir(parents=True)
        Image.new("RGB", (100, 80), color).save(base / f"data/000/{split}.png")
        data = {
            "images": [
                {"id": 0, "file_name": f"{split}.png", "width": 100, "height": 80}
            ],
            "annotations": [
                {"id": 1, "image_id": 0, "category_id": 8, "bbox": [10, 20, 30, 40]}
            ]
            if split == "train"
            else [],
            "categories": [{"id": 8, "name": "Two-wheeler"}],
        }
        (base / f"UVH-26-MV-{split.title()}.json").write_text(json.dumps(data))
    monkeypatch.setattr(conversion, "ROOT", tmp_path)
    return config, tmp_path


def test_audit_convert_background_and_idempotence(dataset):
    config, root = dataset
    result = audit(config)
    assert result["splits"]["val"]["images_without_annotations"] == 1
    assert not result["leakage"]["shared_sha256"]
    out = conversion.convert(config, "test_v1")
    assert (out / "images/train/train.png").is_file()
    assert (out / "images/train/train.png").is_symlink()
    assert (out / "labels/val/val.txt").read_text() == ""
    line = (out / "labels/train/train.txt").read_text().strip()
    assert validate_line(line, 1)[1] == pytest.approx([0.25, 0.5, 0.3, 0.5])
    assert conversion.convert(config, "test_v1") == out
    path = root / "data/raw/UVH-26-Train/UVH-26-MV-Train.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="differs"):
        conversion.convert(config, "test_v1")


def test_unknown_category_blocks_conversion(dataset):
    config, root = dataset
    path = root / "data/raw/UVH-26-Train/UVH-26-MV-Train.json"
    data = json.loads(path.read_text())
    data["annotations"][0]["category_id"] = 99
    path.write_text(json.dumps(data))
    result = audit(config)
    assert result["splits"]["train"]["invalid_annotations"] == 1
    with pytest.raises(ValueError, match="Unknown category"):
        conversion.convert(config, "test_v1")


def test_duplicate_content_blocks_conversion(dataset):
    config, root = dataset
    val = root / "data/raw/UVH-26-Val/data/000/val.png"
    val.write_bytes((root / "data/raw/UVH-26-Train/data/000/train.png").read_bytes())
    result = audit(config)
    assert len(result["leakage"]["shared_sha256"]) == 1
    with pytest.raises(ValueError, match="leakage"):
        conversion.convert(config, "test_v1")


def test_missing_field_saved_before_audit_fails(dataset):
    config, root = dataset
    path = root / "data/raw/UVH-26-Train/UVH-26-MV-Train.json"
    data = json.loads(path.read_text())
    del data["images"][0]["id"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Missing mandatory fields"):
        audit(config)
    report = json.loads((root / "reports/audit/schema_failure.json").read_text())
    assert report["splits"]["train"]["missing_fields"][0]["missing"] == "id"


def test_duplicate_category_ids_block_conversion(dataset):
    config, root = dataset
    path = root / "data/raw/UVH-26-Train/UVH-26-MV-Train.json"
    data = json.loads(path.read_text())
    data["categories"].append(data["categories"][0].copy())
    path.write_text(json.dumps(data))
    report = audit(config)
    assert report["splits"]["train"]["duplicate_category_ids"] == [8]
    with pytest.raises(ValueError, match="structural"):
        conversion.convert(config, "test_v1")


def test_independent_subset_preparation_handles_background(dataset, monkeypatch):
    from src.data import prepare_subset
    from src.data.common import REVISION, sha256

    config, root = dataset
    monkeypatch.setattr(prepare_subset, "ROOT", root)
    interim = root / "data/interim"
    interim.mkdir()
    files = [
        {"rfilename": f"UVH-26-{s.title()}/data/000/{s}.png"} for s in ["train", "val"]
    ]
    (interim / "hub_metadata.json").write_text(
        json.dumps({"sha": REVISION, "siblings": files})
    )
    prepare_subset.prepare(config, "subset_test_v1", 1, 1, False)
    out = root / "data/processed/subset_test_v1"
    assert (out / "labels/val/val.txt").read_text() == ""
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest) == 2
    assert all(r["label_sha256"] == sha256(out / r["label"]) for r in manifest)
    assert (out / "images/train/train.png").is_file()
    with pytest.raises(ValueError, match="already exists"):
        prepare_subset.prepare(config, "subset_test_v1", 1, 1, False)
