from contextlib import nullcontext as does_not_raise

import torch
from pytest import mark, raises
from utils.contexts import ExceptionContext
from utils.tensors import ones_, tensor_

from torchjd.scalarization import Constant

from ._asserts import assert_grad_flow, assert_returns_scalar


def test_value() -> None:
    losses = tensor_([1.0, 2.0, 3.0, 4.0])
    weights = tensor_([0.1, 0.2, 0.3, 0.4])
    torch.testing.assert_close(Constant(weights)(losses), tensor_(3.0))


@mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_expected_structure(shape: tuple[int, ...]) -> None:
    losses = ones_(shape)
    weights = ones_(shape)
    assert_returns_scalar(Constant(weights), losses)


@mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_grad_flow(shape: tuple[int, ...]) -> None:
    losses = ones_(shape)
    weights = ones_(shape) / losses.numel()
    assert_grad_flow(Constant(weights), losses)


@mark.parametrize(
    ["weights_shape", "losses_shape", "expectation"],
    [
        ((5,), (5,), does_not_raise()),
        ((3, 4), (3, 4), does_not_raise()),
        ((), (), does_not_raise()),
        ((5,), (4,), raises(ValueError)),
        ((5,), (5, 1), raises(ValueError)),
        ((3, 4), (4, 3), raises(ValueError)),
    ],
)
def test_shape_check(
    weights_shape: tuple[int, ...],
    losses_shape: tuple[int, ...],
    expectation: ExceptionContext,
) -> None:
    weights = ones_(weights_shape)
    losses = ones_(losses_shape)
    with expectation:
        _ = Constant(weights)(losses)


def test_representations() -> None:
    s = Constant(weights=torch.tensor([1.0, 2.0], device="cpu"))
    assert repr(s) == "Constant(weights=tensor([1., 2.]))"
    assert str(s) == "Constant"
