"""Finite-loss checks across tensor and named-loss trainer APIs."""

from collections.abc import Mapping


def assert_finite_losses(losses):
    import torch

    values = losses.values() if isinstance(losses, Mapping) else (losses,)
    for value in values:
        if not bool(torch.isfinite(torch.as_tensor(value)).all()):
            raise RuntimeError("Non-finite training loss: stop immediately")
