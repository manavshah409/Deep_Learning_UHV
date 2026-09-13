import pytest
from src.data.common import convert_box, validate_line


@pytest.mark.parametrize(
    "box",
    [
        [0, 0, 0, 2],
        [0, 0, 2, -1],
        [-1, 0, 1, 1],
        [99, 0, 2, 2],
        [0, 99, 2, 2],
        [0, 0, float("nan"), 1],
        [0, 0, float("inf"), 1],
        [1, 2, 3],
        ["a", 1, 2, 3],
    ],
)
def test_invalid_boxes(box):
    with pytest.raises(ValueError):
        convert_box(box, 100, 100)


@pytest.mark.parametrize("w,h", [(0, 100), (100, -1), (float("nan"), 10)])
def test_bad_image_dimensions(w, h):
    with pytest.raises(ValueError):
        convert_box([1, 1, 2, 2], w, h)


@pytest.mark.parametrize(
    "line",
    [
        "0 .5 .5 .1",
        "1.0 .5 .5 .1 .1",
        "-1 .5 .5 .1 .1",
        "0 nan .5 .1 .1",
        "0 .5 .5 0 .1",
        "0 .99 .5 .2 .1",
        "0 .5 .5 .2 .2 1",
    ],
)
def test_invalid_yolo_rows(line):
    with pytest.raises(ValueError):
        validate_line(line, 14)


def test_valid_yolo():
    assert validate_line("0 .5 .5 1 1", 14) == (0, [0.5, 0.5, 1, 1])
