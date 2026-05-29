import torch
from pytest import mark, raises
from torch import Tensor
from utils.tensors import rand_, tensor_

from torchjd.scalarization import GeometricMean

from ._asserts import (
    assert_grad_flow,
    assert_permutation_invariant,
    assert_returns_scalar,
)
from ._inputs import shapes

positive_inputs: list[Tensor] = [rand_(shape) + 1 for shape in shapes]


def test_value() -> None:
    losses = tensor_([1.0, 2.0, 4.0])
    torch.testing.assert_close(GeometricMean()(losses), tensor_(2.0))


@mark.parametrize("losses", positive_inputs)
def test_expected_structure(losses: Tensor) -> None:
    assert_returns_scalar(GeometricMean(), losses)


@mark.parametrize("losses", positive_inputs)
def test_grad_flow(losses: Tensor) -> None:
    assert_grad_flow(GeometricMean(), losses)


@mark.parametrize("losses", positive_inputs)
def test_permutation_invariant(losses: Tensor) -> None:
    assert_permutation_invariant(GeometricMean(), losses)


@mark.parametrize(
    "invalid",
    [
        tensor_([1.0, 0.0]),
        tensor_([1.0, -1.0]),
        tensor_([1.0, 1e-13]),
    ],
)
def test_raises_on_non_positive_input(invalid: Tensor) -> None:
    with raises(ValueError):
        GeometricMean()(invalid)


def test_representations() -> None:
    s = GeometricMean()
    assert repr(s) == "GeometricMean()"
    assert str(s) == "GeometricMean"
