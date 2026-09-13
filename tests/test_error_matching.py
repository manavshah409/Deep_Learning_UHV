from src.evaluation.error_analysis import match_predictions


def test_one_prediction_cannot_match_twice():
    matches, fn, fp = match_predictions(
        [[0, 0, 10, 10], [0, 0, 10, 10]], [0, 0], [[0, 0, 10, 10]], [0]
    )
    assert len(matches) == 1 and len(fn) == 1 and not fp


def test_confusion_separate_from_miss():
    matches, fn, fp = match_predictions([[0, 0, 10, 10]], [0], [[0, 0, 10, 10]], [1])
    assert not matches[0]["correct_class"] and not fn and not fp


def test_empty_predictions():
    matches, fn, fp = match_predictions([[0, 0, 10, 10]], [0], [], [])
    assert not matches and fn == [0] and not fp
