import pytest
import torch
from src.training.loss_checks import assert_finite_losses


def test_named_loss_dictionary_from_current_ultralytics():
    assert_finite_losses(
        {"box_loss": torch.tensor(1.0), "cls_loss": 1.5, "dfl_loss": torch.tensor(0.9)}
    )


def test_tensor_loss_vector_is_supported():
    assert_finite_losses(torch.tensor([1.0, 1.5, 0.9]))


def test_nan_in_named_loss_stops_training():
    with pytest.raises(RuntimeError, match="Non-finite"):
        assert_finite_losses({"box_loss": 1.0, "cls_loss": torch.tensor(float("nan"))})


def test_infinite_loss_stops_training():
    with pytest.raises(RuntimeError, match="Non-finite"):
        assert_finite_losses(torch.tensor([1.0, float("inf")]))
