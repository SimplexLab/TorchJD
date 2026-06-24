import torch
from pytest import mark, raises
from torch import Tensor
from utils.tensors import tensor_

from torchjd.scalarization import PBI

from ._asserts import (
    assert_grad_flow,
    assert_permutation_invariant,
    assert_returns_scalar,
)
from ._inputs import all_inputs


def _uniform(values: Tensor) -> Tensor:
    """Uniform preference vector matching the shape of `values`."""
    return torch.full_like(values, 1.0 / values.numel())


def test_value() -> None:
    # direction = [1, 1] / sqrt(2). For [2, 0]: d1 = sqrt(2), perpendicular = [1, -1] so
    # d2 = sqrt(2), and d1 + theta * d2 = 2 * sqrt(2).
    out = PBI(theta=1.0, weights=tensor_([1.0, 1.0]))(tensor_([2.0, 0.0]))
    torch.testing.assert_close(out, tensor_(2.0) * tensor_(2.0).sqrt())


def test_theta_zero_is_projection() -> None:
    # With theta = 0 only the projection d1 remains. For [2, 0] onto [1, 1] / sqrt(2): d1 = sqrt(2).
    out = PBI(theta=0.0, weights=tensor_([1.0, 1.0]))(tensor_([2.0, 0.0]))
    torch.testing.assert_close(out, tensor_(2.0).sqrt())


def test_reference_shifts_values() -> None:
    # Subtracting the reference [1, 1] from [3, 1] gives [2, 0], matching the no-reference case.
    with_reference = PBI(theta=1.0, weights=tensor_([1.0, 1.0]), reference=tensor_([1.0, 1.0]))
    out = with_reference(tensor_([3.0, 1.0]))
    expected = PBI(theta=1.0, weights=tensor_([1.0, 1.0]))(tensor_([2.0, 0.0]))
    torch.testing.assert_close(out, expected)


def test_full_formula() -> None:
    values = tensor_([1.0, 2.0, 4.0])
    weights = tensor_([0.5, 0.3, 0.2])
    reference = tensor_([0.5, 0.5, 0.5])
    theta = 5.0
    shifted = values - reference
    direction = weights / weights.norm()
    d1 = (shifted * direction).sum()
    d2 = (shifted - d1 * direction).norm()
    expected = d1 + theta * d2
    torch.testing.assert_close(PBI(theta, weights=weights, reference=reference)(values), expected)


def test_finite_when_values_on_preference_ray() -> None:
    # When the values lie exactly on the preference direction, d2 = 0. The constant under the square
    # root keeps both the value and the gradient finite (no nan), which is the whole point of the
    # stabilization.
    weights = tensor_([1.0, 2.0])
    leaf = weights.detach().clone().requires_grad_()  # values == weights, so they are on the ray.
    out = PBI(theta=5.0, weights=weights)(leaf)
    out.backward()
    assert out.isfinite()
    assert leaf.grad is not None
    assert leaf.grad.isfinite().all()


@mark.parametrize("values", all_inputs)
def test_expected_structure(values: Tensor) -> None:
    assert_returns_scalar(PBI(theta=5.0, weights=_uniform(values)), values)


@mark.parametrize("values", all_inputs)
def test_grad_flow(values: Tensor) -> None:
    assert_grad_flow(PBI(theta=5.0, weights=_uniform(values)), values)


@mark.parametrize("values", all_inputs)
def test_permutation_invariant(values: Tensor) -> None:
    # With uniform weights and no reference, both d1 and d2 are symmetric in the inputs.
    assert_permutation_invariant(PBI(theta=5.0, weights=_uniform(values)), values)


@mark.parametrize("theta", [-1.0, -0.5])
def test_raises_on_negative_theta(theta: float) -> None:
    with raises(ValueError):
        PBI(theta=theta, weights=tensor_([0.5, 0.5]))


def test_raises_on_weights_shape_mismatch() -> None:
    scalarizer = PBI(theta=5.0, weights=tensor_([1.0, 1.0, 1.0]))
    with raises(ValueError):
        scalarizer(tensor_([1.0, 1.0]))


def test_raises_on_reference_shape_mismatch() -> None:
    scalarizer = PBI(theta=5.0, weights=tensor_([1.0, 1.0]), reference=tensor_([0.0, 0.0, 0.0]))
    with raises(ValueError):
        scalarizer(tensor_([1.0, 1.0]))


def test_representations() -> None:
    s = PBI(theta=5.0, weights=torch.tensor([0.5, 0.5]))
    assert repr(s) == "PBI(theta=5.0, weights=tensor([0.5000, 0.5000]), reference=None)"
    assert str(s) == "PBI"
