import torch
from pytest import mark
from torch import Tensor
from utils.tensors import rand_, tensor_

from torchjd.scalarization import GLS

from ._asserts import (
    assert_grad_flow,
    assert_permutation_invariant,
    assert_returns_scalar,
)

shapes: list[list[int]] = [[], [5], [3, 4], [2, 3, 4]]
positive_inputs: list[Tensor] = [rand_(shape) + 1 for shape in shapes]


def test_value() -> None:
    losses = tensor_([1.0, 2.0, 4.0])
    torch.testing.assert_close(GLS()(losses), tensor_(2.0))


@mark.parametrize("losses", positive_inputs)
def test_expected_structure(losses: Tensor) -> None:
    assert_returns_scalar(GLS(), losses)


@mark.parametrize("losses", positive_inputs)
def test_grad_flow(losses: Tensor) -> None:
    assert_grad_flow(GLS(), losses)


@mark.parametrize("losses", positive_inputs)
def test_permutation_invariant(losses: Tensor) -> None:
    assert_permutation_invariant(GLS(), losses)


def test_propagates_nan_on_invalid_input() -> None:
    # GLS is undefined for non-positive values; nan must propagate (no silent clamp).
    assert GLS()(tensor_([1.0, -1.0])).isnan()


def test_representations() -> None:
    s = GLS()
    assert repr(s) == "GLS()"
    assert str(s) == "GLS"
