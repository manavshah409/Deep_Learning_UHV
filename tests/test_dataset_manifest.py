import pytest
from src.data.common import validate_manifest
from src.data.build_subset import select


def row(split, i, name):
    return {"split": split, "image_id": i, "source": name}


def test_scoped_ids():
    validate_manifest([row("train", 0, "a.png"), row("val", 0, "b.png")])


def test_filename_leakage():
    with pytest.raises(ValueError):
        validate_manifest([row("train", 0, "train/a.png"), row("val", 1, "val/a.png")])


def test_duplicate_within_split():
    with pytest.raises(ValueError):
        validate_manifest([row("train", 0, "a.png"), row("train", 0, "b.png")])


def test_unknown_split():
    with pytest.raises(ValueError):
        validate_manifest([row("test", 0, "a.png")])


def test_subset_determinism_and_rare_coverage():
    rows = [row("train", i, f"{i}.png") for i in range(50)]
    cs = {i: ({0, 1} if i == 49 else {0}) for i in range(50)}
    a = select(rows, cs, 10, 42)
    assert a == select(list(reversed(rows)), cs, 10, 42)
    assert len(a) == 10 and len({r["image_id"] for r in a}) == 10
    assert 49 in {r["image_id"] for r in a}
    assert all(r["split"] == "train" for r in a)


def test_subset_too_small_for_coverage():
    with pytest.raises(ValueError):
        select(
            [row("train", i, f"{i}.png") for i in range(3)],
            {0: {0}, 1: {1}, 2: {2}},
            2,
            42,
        )
