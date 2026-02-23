from typing import Any

from pytest import mark, raises
from torch import Tensor
from torch.testing import assert_close
from utils.asserts import assert_grad_close, assert_has_jac, assert_has_no_jac
from utils.tensors import tensor_

from torchjd.aggregation import (
    IMTLG,
    MGDA,
    Aggregator,
    AlignedMTL,
    ConFIG,
    Constant,
    DualProj,
    GradDrop,
    Krum,
    Mean,
    PCGrad,
    Random,
    Sum,
    TrimmedMean,
    UPGrad,
)
from torchjd.aggregation._aggregator_bases import GramianWeightedAggregator, WeightedAggregator
from torchjd.autojac._jac_to_grad import (
    _can_skip_jacobian_combination,
    _has_forward_hook,
    jac_to_grad,
)


@mark.parametrize(
    ["aggregator", "opt"],
    [(Mean(), False), (UPGrad(), True), (UPGrad(), False), (PCGrad(), True), (ConFIG(), False)],
)
def test_various_aggregators(aggregator: Aggregator, optimize: bool) -> None:
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

    if optimize:
        assert isinstance(aggregator, GramianWeightedAggregator)
        optional_weights = jac_to_grad([t1, t2], aggregator, optimize_gramian_computation=True)
    else:
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
    """Tests that jac_to_grad correctly raises when an empty list of tensors is provided."""

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


def test_has_forward_hook() -> None:
    """Tests that _has_forward_hook correctly detects the presence of forward hooks."""

    module = UPGrad()

    def dummy_forward_hook(_module, _input, _output) -> Tensor:
        return _output

    def dummy_forward_pre_hook(_module, _input) -> Tensor:
        return _input

    def dummy_backward_hook(_module, _grad_input, _grad_output) -> Tensor:
        return _grad_input

    def dummy_backward_pre_hook(_module, _grad_output) -> Tensor:
        return _grad_output

    # Module with no hooks or backward hooks only should return False
    assert not _has_forward_hook(module)
    module.register_full_backward_hook(dummy_backward_hook)
    assert not _has_forward_hook(module)
    module.register_full_backward_pre_hook(dummy_backward_pre_hook)
    assert not _has_forward_hook(module)

    # Module with forward hook should return True
    handle1 = module.register_forward_hook(dummy_forward_hook)
    assert _has_forward_hook(module)
    handle2 = module.register_forward_hook(dummy_forward_hook)
    assert _has_forward_hook(module)
    handle1.remove()
    assert _has_forward_hook(module)
    handle2.remove()
    assert not _has_forward_hook(module)

    # Module with forward pre-hook should return True
    handle3 = module.register_forward_pre_hook(dummy_forward_pre_hook)
    assert _has_forward_hook(module)
    handle4 = module.register_forward_pre_hook(dummy_forward_pre_hook)
    assert _has_forward_hook(module)
    handle3.remove()
    assert _has_forward_hook(module)
    handle4.remove()
    assert not _has_forward_hook(module)


_PARAMETRIZATIONS = [
    (AlignedMTL(), True),
    (DualProj(), True),
    (IMTLG(), True),
    (Krum(n_byzantine=1), True),
    (MGDA(), True),
    (PCGrad(), True),
    (UPGrad(), True),
    (ConFIG(), False),
    (Constant(tensor_([0.5, 0.5])), False),
    (GradDrop(), False),
    (Mean(), False),
    (Random(), False),
    (Sum(), False),
    (TrimmedMean(trim_number=1), False),
]

try:
    from torchjd.aggregation import CAGrad

    _PARAMETRIZATIONS.append((CAGrad(c=0.5), True))
except ImportError:
    pass

try:
    from torchjd.aggregation import NashMTL

    _PARAMETRIZATIONS.append((NashMTL(n_tasks=2), False))
except ImportError:
    pass


@mark.parametrize("aggregator, expected", _PARAMETRIZATIONS)
def test_can_skip_jacobian_combination(aggregator: Aggregator, expected: bool) -> None:
    """
    Tests that _can_skip_jacobian_combination correctly identifies when optimization can be used.
    """

    assert _can_skip_jacobian_combination(aggregator) == expected
    handle = aggregator.register_forward_hook(lambda _module, _input, output: output)
    assert not _can_skip_jacobian_combination(aggregator)
    handle.remove()
    assert _can_skip_jacobian_combination(aggregator) == expected
    handle = aggregator.register_forward_pre_hook(lambda _module, input: input)
    assert not _can_skip_jacobian_combination(aggregator)
    handle.remove()
    assert _can_skip_jacobian_combination(aggregator) == expected

    if isinstance(aggregator, GramianWeightedAggregator):
        handle = aggregator.weighting.register_forward_hook(lambda _module, _input, output: output)
        assert not _can_skip_jacobian_combination(aggregator)
        handle.remove()
        assert _can_skip_jacobian_combination(aggregator) == expected
        handle = aggregator.weighting.register_forward_pre_hook(lambda _module, input: input)
        assert not _can_skip_jacobian_combination(aggregator)
        handle.remove()
        assert _can_skip_jacobian_combination(aggregator) == expected


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


def test_with_hooks() -> None:
    """Tests that jac_to_grad correctly returns the weights modified by all applicable hooks."""

    def hook_aggregator(_module: Any, _input: Any, aggregation: Tensor) -> Tensor:
        return aggregation * 2  # should not affect the weights

    def hook_outer(_module: Any, _input: Any, weights: Tensor) -> Tensor:
        return weights * 3  # should affect the weights returned by jac_to_grad

    def hook_inner(_module: Any, _input: Any, weights: Tensor) -> Tensor:
        return weights * 5  # should affect the weights returned by jac_to_grad

    aggregator = UPGrad()
    aggregator.register_forward_hook(hook_aggregator)
    aggregator.weighting.register_forward_hook(hook_outer)
    aggregator.gramian_weighting.register_forward_hook(hook_inner)

    t = tensor_([2.0, 3.0], requires_grad=True)
    jac = tensor_([[-4.0, 1.0], [6.0, 1.0]])
    t.__setattr__("jac", jac)

    weights = jac_to_grad([t], aggregator)
    assert_close(weights, aggregator.weighting(jac))


def test_optimize_gramian_computation_error() -> None:
    """
    Tests that using optimize_gramian_computation on an incompatible aggregator raises an error.
    """

    aggregator = ConFIG()

    t1 = tensor_(1.0, requires_grad=True)
    t2 = tensor_([2.0, 3.0], requires_grad=True)
    jac = tensor_([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    t1.__setattr__("jac", jac[:, 0])
    t2.__setattr__("jac", jac[:, 1:])

    with raises(ValueError):
        jac_to_grad([t1, t2], aggregator, optimize_gramian_computation=True)  # ty:ignore[invalid-argument-type]
