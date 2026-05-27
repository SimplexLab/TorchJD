import torch
from pytest import mark
from torch import Tensor
from utils.tensors import tensor_

from torchjd.scalarization import Mean

from ._asserts import (
    assert_grad_flow,
    assert_permutation_invariant,
    assert_returns_scalar,
)
from ._inputs import all_inputs, non_scalar_inputs


def test_value() -> None:
    losses = tensor_([1.0, 2.0, 3.0])
    torch.testing.assert_close(Mean()(losses), tensor_(2.0))


@mark.parametrize("losses", all_inputs)
def test_expected_structure(losses: Tensor) -> None:
    assert_returns_scalar(Mean(), losses)


@mark.parametrize("losses", non_scalar_inputs)
def test_grad_flow(losses: Tensor) -> None:
    assert_grad_flow(Mean(), losses)


@mark.parametrize("losses", non_scalar_inputs)
def test_permutation_invariant(losses: Tensor) -> None:
    assert_permutation_invariant(Mean(), losses)


def test_representations() -> None:
    s = Mean()
    assert repr(s) == "Mean()"
    assert str(s) == "Mean"
