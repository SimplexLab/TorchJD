from contextlib import nullcontext as does_not_raise

import torch
from pytest import mark, raises
from settings import DEVICE, DTYPE
from torch import Tensor
from utils.contexts import ExceptionContext
from utils.tensors import ones_, tensor_, zeros_

from torchjd.scalarization import UW

from ._asserts import assert_grad_flow, assert_returns_scalar
from ._inputs import all_inputs


def _uw(shape: int | tuple[int, ...]) -> UW:
    """Builds a `UW` whose log-variances live on the test device and dtype."""
    return UW(shape).to(device=DEVICE, dtype=DTYPE)


def test_value() -> None:
    # With log-variances initialized to 0, the result is 0.5 * sum(values).
    values = tensor_([1.0, 2.0, 4.0])
    torch.testing.assert_close(_uw((3,))(values), tensor_(3.5))


def test_int_shape_matches_tuple_shape() -> None:
    values = tensor_([1.0, 2.0, 4.0])
    assert UW(3).log_var.shape == (3,)
    torch.testing.assert_close(_uw(3)(values), _uw((3,))(values))


@mark.parametrize("values", all_inputs)
def test_expected_structure(values: Tensor) -> None:
    assert_returns_scalar(_uw(tuple(values.shape)), values)


@mark.parametrize("values", all_inputs)
def test_grad_flow(values: Tensor) -> None:
    assert_grad_flow(_uw(tuple(values.shape)), values)


@mark.parametrize("values", all_inputs)
def test_grad_flows_to_log_var(values: Tensor) -> None:
    scalarizer = _uw(tuple(values.shape))
    scalarizer(values).backward()
    assert scalarizer.log_var.grad is not None
    assert scalarizer.log_var.grad.isfinite().all()


@mark.parametrize(
    ["param_shape", "values_shape", "expectation"],
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
    param_shape: tuple[int, ...],
    values_shape: tuple[int, ...],
    expectation: ExceptionContext,
) -> None:
    scalarizer = _uw(param_shape)
    values = ones_(values_shape)
    with expectation:
        _ = scalarizer(values)


def test_reset_restores_initial_log_var() -> None:
    scalarizer = _uw((3,))
    with torch.no_grad():
        scalarizer.log_var.add_(1.0)
    scalarizer.reset()
    torch.testing.assert_close(scalarizer.log_var.detach(), zeros_((3,)))


def test_does_not_raise_on_negative_input() -> None:
    # Unlike GeometricMean, UW has no positivity precondition.
    values = tensor_([-1.0, -2.0, 3.0])
    assert_returns_scalar(_uw((3,)), values)


def test_is_trainable() -> None:
    scalarizer = _uw((2,))
    optimizer = torch.optim.SGD(scalarizer.parameters(), lr=0.1)
    values = tensor_([2.0, 5.0])
    optimizer.zero_grad()
    scalarizer(values).backward()
    optimizer.step()
    assert not torch.equal(scalarizer.log_var.detach(), zeros_((2,)))


def test_representations() -> None:
    assert repr(UW(3)) == "UW(shape=(3,))"
    assert repr(UW((2, 3))) == "UW(shape=(2, 3))"
    assert str(UW(3)) == "UW"
