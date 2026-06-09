from contextlib import nullcontext as does_not_raise

import torch
from pytest import mark, raises
from settings import DEVICE, DTYPE
from torch import Tensor
from utils.contexts import ExceptionContext
from utils.tensors import ones_, tensor_, zeros_

from torchjd.scalarization import IMTLL, UW

from ._asserts import assert_grad_flow, assert_returns_scalar
from ._inputs import all_inputs


def _imtl_l(shape: int | tuple[int, ...]) -> IMTLL:
    """Builds an `IMTLL` whose scales live on the test device and dtype."""
    return IMTLL(shape).to(device=DEVICE, dtype=DTYPE)


def test_value() -> None:
    # With scales initialized to 0, exp(0)=1 and -0=0, so the result is sum(values).
    values = tensor_([1.0, 2.0, 4.0])
    torch.testing.assert_close(_imtl_l((3,))(values), tensor_(7.0))


def test_int_shape_matches_tuple_shape() -> None:
    values = tensor_([1.0, 2.0, 4.0])
    assert IMTLL(3).log_scale.shape == (3,)
    torch.testing.assert_close(_imtl_l(3)(values), _imtl_l((3,))(values))


@mark.parametrize("values", all_inputs)
def test_expected_structure(values: Tensor) -> None:
    assert_returns_scalar(_imtl_l(tuple(values.shape)), values)


@mark.parametrize("values", all_inputs)
def test_grad_flow(values: Tensor) -> None:
    assert_grad_flow(_imtl_l(tuple(values.shape)), values)


@mark.parametrize("values", all_inputs)
def test_grad_flows_to_log_scale(values: Tensor) -> None:
    scalarizer = _imtl_l(tuple(values.shape))
    scalarizer(values).backward()
    assert scalarizer.log_scale.grad is not None
    assert scalarizer.log_scale.grad.isfinite().all()


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
    scalarizer = _imtl_l(param_shape)
    values = ones_(values_shape)
    with expectation:
        _ = scalarizer(values)


def test_reset_restores_initial_log_scale() -> None:
    scalarizer = _imtl_l((3,))
    with torch.no_grad():
        scalarizer.log_scale.add_(1.0)
    scalarizer.reset()
    torch.testing.assert_close(scalarizer.log_scale.detach(), zeros_((3,)))


def test_does_not_raise_on_negative_input() -> None:
    # IMTL-L is designed for positive losses but does not enforce a positivity precondition.
    values = tensor_([-1.0, -2.0, 3.0])
    assert_returns_scalar(_imtl_l((3,)), values)


def test_is_trainable() -> None:
    scalarizer = _imtl_l((2,))
    optimizer = torch.optim.SGD(scalarizer.parameters(), lr=0.1)
    values = tensor_([2.0, 5.0])
    optimizer.zero_grad()
    scalarizer(values).backward()
    optimizer.step()
    assert not torch.equal(scalarizer.log_scale.detach(), zeros_((2,)))


def test_equivalent_to_uw_up_to_factor_and_sign() -> None:
    # Locks the documented relationship: IMTL-L(s) == 2 * UW(-s), i.e. the two scalarizations are
    # equal up to a constant factor of 2 and the sign of the learned parameter.
    values = tensor_([0.5, 2.0, 4.0])
    imtl_l = _imtl_l((3,))
    uw = UW((3,)).to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        s = tensor_([0.3, -0.7, 1.2])
        imtl_l.log_scale.copy_(s)
        uw.log_var.copy_(-s)
    torch.testing.assert_close(imtl_l(values), 2.0 * uw(values))


def test_representations() -> None:
    assert repr(IMTLL(3)) == "IMTLL(shape=(3,))"
    assert repr(IMTLL((2, 3))) == "IMTLL(shape=(2, 3))"
    assert str(IMTLL(3)) == "IMTLL"
