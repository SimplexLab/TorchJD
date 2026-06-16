import torch
from pytest import mark, raises
from torch import Tensor
from utils.tensors import tensor_

from torchjd.scalarization import COSMOS

from ._asserts import (
    assert_grad_flow,
    assert_permutation_invariant,
    assert_returns_scalar,
)
from ._inputs import all_inputs


def test_value_default() -> None:
    # Uniform weights on equal values are perfectly aligned, so cos(r, L) = 1. The result is the
    # weighted sum (1) minus lambda (1): 0.
    out = COSMOS(lambda_=1.0)(tensor_([1.0, 1.0]))
    torch.testing.assert_close(out, tensor_(0.0))


def test_value_lambda_zero_is_linear_scalarization() -> None:
    # With lambda = 0 there is no cosine penalty, so COSMOS is the (uniform) weighted sum.
    out = COSMOS(lambda_=0.0)(tensor_([1.0, 2.0, 4.0]))
    torch.testing.assert_close(out, tensor_(7.0 / 3.0))


def test_value_with_weights() -> None:
    # With lambda = 0, only the linear term remains: 2*3 + 1*4 = 10.
    out = COSMOS(lambda_=0.0, weights=tensor_([2.0, 1.0]))(tensor_([3.0, 4.0]))
    torch.testing.assert_close(out, tensor_(10.0))


def test_full_formula() -> None:
    values = tensor_([1.0, 2.0, 4.0])
    weights = tensor_([0.5, 0.3, 0.2])
    lambda_ = 2.0
    weighted_sum = (weights * values).sum()
    expected = weighted_sum - lambda_ * weighted_sum / (weights.norm() * values.norm())
    torch.testing.assert_close(COSMOS(lambda_, weights=weights)(values), expected)


@mark.parametrize("values", all_inputs)
def test_expected_structure(values: Tensor) -> None:
    assert_returns_scalar(COSMOS(lambda_=1.0), values)


@mark.parametrize("values", all_inputs)
def test_grad_flow(values: Tensor) -> None:
    assert_grad_flow(COSMOS(lambda_=1.0), values)


@mark.parametrize("values", all_inputs)
def test_permutation_invariant(values: Tensor) -> None:
    # With uniform weights, both the weighted sum and the cosine term are symmetric in the inputs.
    assert_permutation_invariant(COSMOS(lambda_=1.0), values)


def test_nan_for_all_zero_values() -> None:
    # The cosine term divides by ||L||, so an all-zero vector of values produces nan.
    out = COSMOS(lambda_=1.0)(tensor_([0.0, 0.0]))
    assert out.isnan()


@mark.parametrize("lambda_", [-1.0, -0.5])
def test_raises_on_negative_lambda(lambda_: float) -> None:
    with raises(ValueError):
        COSMOS(lambda_=lambda_)


def test_raises_on_weights_shape_mismatch() -> None:
    scalarizer = COSMOS(lambda_=1.0, weights=tensor_([1.0, 1.0, 1.0]))
    with raises(ValueError):
        scalarizer(tensor_([1.0, 1.0]))


def test_representations() -> None:
    s = COSMOS(lambda_=0.5)
    assert repr(s) == "COSMOS(lambda_=0.5, weights=None)"
    assert str(s) == "COSMOS"
