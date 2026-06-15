from contextlib import nullcontext as does_not_raise

import torch
from pytest import mark, raises
from settings import DEVICE, DTYPE
from torch import Tensor
from utils.contexts import ExceptionContext
from utils.tensors import ones_, rand_, tensor_, zeros_

from torchjd.scalarization import FAMO

from ._asserts import assert_grad_flow, assert_returns_scalar
from ._inputs import shapes

# FAMO takes the log of the values, so they must be strictly positive.
positive_inputs = [rand_(shape) + 1.0 for shape in shapes]


def _famo(shape: int | tuple[int, ...]) -> FAMO:
    """Builds a `FAMO` whose logits and lower bounds live on the test device and dtype."""
    return FAMO(shape).to(device=DEVICE, dtype=DTYPE)


def test_value() -> None:
    # With logits initialized to 0, the task weights are uniform. The result is the normalized,
    # weighted sum of the log-values.
    values = tensor_([1.0, 2.0])
    z = tensor_([0.5, 0.5])
    shifted = values + 1e-8
    c = 1.0 / (z / shifted).sum()
    expected = (c * z * torch.log(shifted)).sum()
    torch.testing.assert_close(_famo(2)(values), expected)


def test_value_gradient_matches_formula() -> None:
    # The gradient w.r.t. the values is FAMO's balanced update direction: c * z / (values - b).
    values = tensor_([1.0, 2.0]).requires_grad_()
    _famo(2)(values).backward()
    z = tensor_([0.5, 0.5])
    shifted = values.detach() + 1e-8
    expected_grad = (1.0 / (z / shifted).sum()) * z / shifted
    torch.testing.assert_close(values.grad, expected_grad)


def test_int_shape_matches_tuple_shape() -> None:
    values = tensor_([1.0, 2.0, 4.0])
    assert FAMO(3)._w.shape == (3,)
    torch.testing.assert_close(_famo(3)(values), _famo((3,))(values))


@mark.parametrize("values", positive_inputs)
def test_expected_structure(values: Tensor) -> None:
    assert_returns_scalar(_famo(tuple(values.shape)), values)


@mark.parametrize("values", positive_inputs)
def test_grad_flow(values: Tensor) -> None:
    assert_grad_flow(_famo(tuple(values.shape)), values)


def test_forward_does_not_write_logit_grad() -> None:
    # The weights are detached inside `forward`, so backward populates the values' gradient but
    # never the logits' gradient (those are updated by `update` instead).
    famo = _famo(2)
    values = tensor_([1.0, 2.0]).requires_grad_()
    famo(values).backward()
    assert values.grad is not None
    assert famo._w.grad is None


def test_update_steps_the_logits() -> None:
    famo = _famo(2)
    famo(tensor_([1.0, 2.0]))
    famo.update(tensor_([0.5, 1.5]))
    assert famo._w.detach().isfinite().all()
    assert not torch.equal(famo._w.detach(), zeros_((2,)))


def test_update_clears_logit_grad() -> None:
    # After stepping its own optimizer, FAMO clears the logit gradient so it cannot leak into a user
    # optimizer the logits were mistakenly added to.
    famo = _famo(2)
    famo(tensor_([1.0, 2.0]))
    famo.update(tensor_([0.5, 1.5]))
    assert famo._w.grad is None


def test_update_uses_last_forward_losses() -> None:
    # `update` compares the losses from the most recent `forward` against the ones it receives. When
    # they are equal, the change is zero, so the logits do not move.
    famo = _famo(2)
    famo(tensor_([5.0, 5.0]))
    famo(tensor_([1.0, 4.0]))
    famo.update(tensor_([1.0, 4.0]))
    torch.testing.assert_close(famo._w.detach(), zeros_((2,)))


def test_update_before_forward_raises() -> None:
    with raises(ValueError):
        _famo(2).update(tensor_([1.0, 2.0]))


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
def test_forward_shape_check(
    param_shape: tuple[int, ...],
    values_shape: tuple[int, ...],
    expectation: ExceptionContext,
) -> None:
    scalarizer = _famo(param_shape)
    values = ones_(values_shape)
    with expectation:
        _ = scalarizer(values)


def test_update_shape_check() -> None:
    famo = _famo(2)
    famo(tensor_([1.0, 2.0]))
    with raises(ValueError):
        famo.update(tensor_([1.0, 2.0, 3.0]))


def test_min_losses_defaults_to_zeros() -> None:
    torch.testing.assert_close(FAMO(2).min_losses, torch.zeros(2))


def test_min_losses_wrong_shape_raises() -> None:
    with raises(ValueError):
        FAMO(2, min_losses=tensor_([0.0, 0.0, 0.0]))


def test_min_losses_shifts_values() -> None:
    # With a lower bound, the log is taken on values - min_losses.
    values = tensor_([2.0, 3.0])
    bound = tensor_([1.0, 1.0])
    famo = FAMO(2, min_losses=bound).to(device=DEVICE, dtype=DTYPE)
    z = tensor_([0.5, 0.5])
    shifted = values - bound + 1e-8
    c = 1.0 / (z / shifted).sum()
    expected = (c * z * torch.log(shifted)).sum()
    torch.testing.assert_close(famo(values), expected)


@mark.parametrize("lr", [-1.0])
def test_negative_lr_raises(lr: float) -> None:
    with raises(ValueError):
        FAMO(2, lr=lr)


@mark.parametrize("weight_decay", [-1.0])
def test_negative_weight_decay_raises(weight_decay: float) -> None:
    with raises(ValueError):
        FAMO(2, weight_decay=weight_decay)


def test_reset() -> None:
    famo = _famo(2)
    famo(tensor_([1.0, 2.0]))
    famo.update(tensor_([0.5, 1.5]))
    famo.reset()
    torch.testing.assert_close(famo._w.detach(), zeros_((2,)))
    assert famo._prev_losses is None
    assert famo._optimizer is None


def test_nan_propagates_for_value_below_bound() -> None:
    # log(values - min_losses) is undefined when a value is not above its bound; the nan must
    # propagate rather than being silently clamped.
    out = _famo(2)(tensor_([-1.0, 2.0]))
    assert out.isnan()


def test_representations() -> None:
    assert repr(FAMO(3)) == "FAMO(shape=(3,), lr=0.025, weight_decay=0.001)"
    assert repr(FAMO((2, 3))) == "FAMO(shape=(2, 3), lr=0.025, weight_decay=0.001)"
    assert str(FAMO(3)) == "FAMO"
