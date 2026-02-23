from typing import Any

from pytest import mark, raises
from torch.testing import assert_close
from utils.asserts import assert_grad_close, assert_has_jac, assert_has_no_jac
from utils.tensors import tensor_

from torchjd.aggregation import (
    Aggregator,
    ConFIG,
    Mean,
    PCGrad,
    UPGrad,
)
from torchjd.aggregation._aggregator_bases import WeightedAggregator
from torchjd.autojac._jac_to_grad import jac_to_grad


@mark.parametrize("aggregator", [Mean(), UPGrad(), PCGrad(), ConFIG()])
def test_various_aggregators(aggregator: Aggregator) -> None:
    """
    Tests that jac_to_grad works for various aggregators. For those that are weighted, the weights
    should also be returned. For the others, None should be returned.
    """

    t1 = tensor_(1.0, requires_grad=True)
    t2 = tensor_([2.0, 3.0], requires_grad=True)
    jac = tensor_([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    t1.__setattr__("jac", jac[:, 0])
    t2.__setattr__("jac", jac[:, 1:])
    expected_grad = aggregator(jac)
    g1 = expected_grad[0]
    g2 = expected_grad[1:]

    optional_weights = jac_to_grad([t1, t2], aggregator)

    assert_grad_close(t1, g1)
    assert_grad_close(t2, g2)

    if isinstance(aggregator, WeightedAggregator):
        assert optional_weights is not None
        expected_weights = aggregator.weighting(jac)
        assert_close(optional_weights, expected_weights)
    else:
        assert optional_weights is None


def test_single_tensor() -> None:
    """Tests that jac_to_grad works when a single tensor is provided."""

    aggregator = UPGrad()
    t = tensor_([2.0, 3.0, 4.0], requires_grad=True)
    jac = tensor_([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    t.__setattr__("jac", jac)
    g = aggregator(jac)

    jac_to_grad([t], aggregator)

    assert_grad_close(t, g)


def test_no_jac_field() -> None:
    """Tests that jac_to_grad fails when a tensor does not have a jac field."""

    aggregator = UPGrad()
    t1 = tensor_(1.0, requires_grad=True)
    t2 = tensor_([2.0, 3.0], requires_grad=True)
    jac = tensor_([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    t2.__setattr__("jac", jac[:, 1:])

    with raises(ValueError):
        jac_to_grad([t1, t2], aggregator)


def test_no_requires_grad() -> None:
    """Tests that jac_to_grad fails when a tensor does not require grad."""

    aggregator = UPGrad()
    t1 = tensor_(1.0, requires_grad=True)
    t2 = tensor_([2.0, 3.0], requires_grad=False)
    jac = tensor_([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    t1.__setattr__("jac", jac[:, 0])
    t2.__setattr__("jac", jac[:, 1:])

    with raises(ValueError):
        jac_to_grad([t1, t2], aggregator)


def test_row_mismatch() -> None:
    """Tests that jac_to_grad fails when the number of rows of the .jac is not constant."""

    aggregator = UPGrad()
    t1 = tensor_(1.0, requires_grad=True)
    t2 = tensor_([2.0, 3.0], requires_grad=True)
    t1.__setattr__("jac", tensor_([5.0, 6.0, 7.0]))  # 3 rows
    t2.__setattr__("jac", tensor_([[1.0, 2.0], [3.0, 4.0]]))  # 2 rows

    with raises(ValueError):
        jac_to_grad([t1, t2], aggregator)


def test_no_tensors() -> None:
    """Tests that jac_to_grad correctly does nothing when an empty list of tensors is provided."""

    with raises(ValueError):
        jac_to_grad([], UPGrad())


@mark.parametrize("retain_jac", [True, False])
def test_jacs_are_freed(retain_jac: bool) -> None:
    """Tests that jac_to_grad frees the jac fields if an only if retain_jac is False."""

    aggregator = UPGrad()
    t1 = tensor_(1.0, requires_grad=True)
    t2 = tensor_([2.0, 3.0], requires_grad=True)
    jac = tensor_([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    t1.__setattr__("jac", jac[:, 0])
    t2.__setattr__("jac", jac[:, 1:])

    jac_to_grad([t1, t2], aggregator, retain_jac=retain_jac)

    check = assert_has_jac if retain_jac else assert_has_no_jac
    check(t1)
    check(t2)


def test_noncontiguous_jac() -> None:
    """Tests that jac_to_grad works when the .jac field is non-contiguous."""

    aggregator = UPGrad()
    t = tensor_([2.0, 3.0, 4.0], requires_grad=True)
    jac_T = tensor_([[-4.0, 1.0], [1.0, 6.0], [1.0, 1.0]])
    jac = jac_T.T
    t.__setattr__("jac", jac)
    g = aggregator(jac)

    jac_to_grad([t], aggregator)
    assert_grad_close(t, g)


@mark.parametrize("aggregator", [UPGrad(), ConFIG()])
def test_aggregator_hook_is_run(aggregator: Aggregator) -> None:
    """
    Tests that jac_to_grad runs forward hooks registered on the aggregator, for both
    WeightedAggregator (UPGrad) and plain Aggregator (ConFIG) paths.
    """

    call_count = [0]  # Pointer to int

    def hook(_module: Any, _input: Any, _output: Any) -> None:
        call_count[0] += 1

    aggregator.register_forward_hook(hook)

    t = tensor_([2.0, 3.0], requires_grad=True)
    jac = tensor_([[-4.0, 1.0], [6.0, 1.0]])
    t.__setattr__("jac", jac)

    jac_to_grad([t], aggregator)

    assert call_count[0] == 1


@mark.parametrize("aggregator", [UPGrad(), PCGrad(), Mean()])
def test_weighting_hook_is_run(aggregator: WeightedAggregator) -> None:
    """
    Tests that jac_to_grad runs forward hooks registered on the inner weighting of a
    WeightedAggregator.
    """

    call_count = [0]  # Pointer to int

    def hook(_module: Any, _input: Any, _output: Any) -> None:
        call_count[0] += 1

    aggregator.weighting.register_forward_hook(hook)

    t = tensor_([2.0, 3.0], requires_grad=True)
    jac = tensor_([[-4.0, 1.0], [6.0, 1.0]])
    t.__setattr__("jac", jac)

    jac_to_grad([t], aggregator)

    assert call_count[0] == 1
