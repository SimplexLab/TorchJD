import torch
from pytest import mark, raises
from torch import Tensor
from utils.tensors import tensor_

from torchjd.scalarization import STCH

from ._asserts import (
    assert_grad_flow,
    assert_permutation_invariant,
    assert_returns_scalar,
)
from ._inputs import all_inputs


def test_value_default() -> None:
    # Uniform weights, no reference: mu * logsumexp([0, 0]) = log(2).
    out = STCH(mu=1.0)(tensor_([0.0, 0.0]))
    torch.testing.assert_close(out, torch.log(tensor_(2.0)))


def test_value_with_weights() -> None:
    # weights = [1, 1] on values [1, 1]: mu * logsumexp([1, 1]) = 1 + log(2).
    out = STCH(mu=1.0, weights=tensor_([1.0, 1.0]))(tensor_([1.0, 1.0]))
    torch.testing.assert_close(out, 1.0 + torch.log(tensor_(2.0)))


def test_value_with_reference() -> None:
    # reference shifts values to [0, 0], so the result collapses back to log(2).
    out = STCH(mu=1.0, weights=tensor_([1.0, 1.0]), reference=tensor_([1.0, 1.0]))(
        tensor_([1.0, 1.0])
    )
    torch.testing.assert_close(out, torch.log(tensor_(2.0)))


@mark.parametrize("losses", all_inputs)
def test_expected_structure(losses: Tensor) -> None:
    assert_returns_scalar(STCH(mu=1.0), losses)


@mark.parametrize("losses", all_inputs)
def test_grad_flow(losses: Tensor) -> None:
    assert_grad_flow(STCH(mu=1.0), losses)


@mark.parametrize("losses", all_inputs)
def test_permutation_invariant(losses: Tensor) -> None:
    # With uniform weights and no reference, STCH is symmetric in its inputs.
    assert_permutation_invariant(STCH(mu=1.0), losses)


def test_does_not_overflow_for_large_values_and_small_mu() -> None:
    # `weights * values / mu` would overflow to inf before logsumexp can stabilize it. The
    # value-preserving centering keeps the result finite and equal to the dominant (max) term.
    values = tensor_([1e30, 2e30, 3e30])
    out = STCH(mu=1e-10)(values)
    assert out.isfinite()
    torch.testing.assert_close(out, tensor_(1e30))  # 3e30 weighted by the uniform 1/3.


@mark.parametrize("mu", [0.0, -1.0])
def test_raises_on_non_positive_mu(mu: float) -> None:
    with raises(ValueError):
        STCH(mu=mu)


def test_raises_on_weights_shape_mismatch() -> None:
    scalarizer = STCH(mu=1.0, weights=tensor_([1.0, 1.0, 1.0]))
    with raises(ValueError):
        scalarizer(tensor_([1.0, 1.0]))


def test_raises_on_reference_shape_mismatch() -> None:
    scalarizer = STCH(mu=1.0, reference=tensor_([1.0, 1.0, 1.0]))
    with raises(ValueError):
        scalarizer(tensor_([1.0, 1.0]))


def test_representations() -> None:
    s = STCH(mu=0.5)
    assert repr(s) == "STCH(mu=0.5, weights=None, reference=None)"
    assert str(s) == "STCH"
