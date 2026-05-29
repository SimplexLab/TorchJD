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
    "negative",
    [
        tensor_([1.0, -1.0]),
        tensor_([-1e-13, 2.0]),
        tensor_([-1.0]),
    ],
)
def test_raises_on_negative_input(negative: Tensor) -> None:
    with raises(ValueError):
        GeometricMean()(negative)


def test_returns_zero_when_a_value_is_zero() -> None:
    # log(0) = -inf, so the geometric mean collapses to 0. This matches the LibMTL behavior;
    # the gradient is nan, which is expected for this method.
    assert GeometricMean()(tensor_([1.0, 0.0])) == 0.0


def test_does_not_raise_on_tiny_positive_input() -> None:
    # Tiny but strictly positive values are valid and must not be rejected.
    assert GeometricMean()(tensor_([1.0, 1e-13])).isfinite()


def test_representations() -> None:
    s = GeometricMean()
    assert repr(s) == "GeometricMean()"
    assert str(s) == "GeometricMean"
