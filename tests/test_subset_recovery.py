from PIL import Image
from src.data.audit_baseline_subset import audit_one, leakage
from src.data.recover_baseline import replacement_rank
from collections import Counter


def test_dimension_mismatch_checks_boxes_against_both_frames(tmp_path):
    Image.new("RGB", (80, 100)).save(tmp_path / "a.png")
    row = dict(split="train", image_id=1, source="a.png")
    metadata = {1: dict(width=100, height=100)}
    annotations = {1: [dict(id=10, category_id=1, bbox=[75, 10, 20, 20])]}
    result = audit_one(row, tmp_path, metadata, annotations, {1})
    assert "dimension_mismatch" in result["errors"]
    assert result["invalid_metadata_boxes"] == []
    assert result["invalid_actual_boxes"] == [
        dict(annotation_id=10, reason="outside_image")
    ]
    assert result["sha256"]


def test_corrupt_image_does_not_prevent_annotation_check(tmp_path):
    (tmp_path / "a.png").write_bytes(b"not a png")
    result = audit_one(
        dict(split="train", image_id=1, source="a.png"),
        tmp_path,
        {1: dict(width=100, height=100)},
        {1: [dict(id=9, category_id=99, bbox=[0, 0, -1, 2])]},
        {1},
    )
    assert any(e.startswith("decode_or_file_failure") for e in result["errors"])
    assert result["unknown_categories"] == [9]
    assert result["invalid_metadata_boxes"]


def test_split_leakage_independent_of_filenames():
    rows = [
        dict(split="train", image_id=1, source="a.png", sha256="same"),
        dict(split="val", image_id=2, source="b.png", sha256="same"),
    ]
    assert leakage(rows)["shared_sha256"] == ["same"]
    assert not leakage(rows)["passed"]
    rows[1]["sha256"] = "other"
    assert leakage(rows)["passed"]


def test_replacement_preserves_exposure_and_has_stable_tiebreak():
    row = dict(split="train", image_id=2)
    annotations = {2: [dict(category_id=7), dict(category_id=8), dict(category_id=8)]}
    target = Counter({7: 1, 8: 2})
    rank = replacement_rank(row, target, annotations, Counter({7: 100, 8: 200}))
    assert rank[0] == 0
    assert rank == replacement_rank(row, target, annotations, Counter({7: 100, 8: 200}))
    assert (
        replacement_rank(
            row, Counter({7: 2, 8: 2}), annotations, Counter({7: 100, 8: 200})
        )[0]
        > 0
    )
