from pytest import mark
from torch import Tensor
from torch.testing import assert_close
from utils.tensors import ones_, randn_

try:
    from torchjd.aggregation import NashMTL
except ImportError:
    import pytest

    pytest.skip("NashMTL dependencies not installed", allow_module_level=True)

from ._asserts import assert_expected_structure, assert_non_differentiable
from ._inputs import nash_mtl_matrices


def _make_aggregator(matrix: Tensor) -> NashMTL:
    return NashMTL(n_tasks=matrix.shape[0])


standard_pairs = [(_make_aggregator(matrix), matrix) for matrix in nash_mtl_matrices]
requires_grad_pairs = [(NashMTL(n_tasks=3), ones_(3, 5, requires_grad=True))]


# Note that as opposed to most aggregators, the expected structure is only tested with non-scaled
# matrices, and with matrices of > 1 row. Otherwise, NashMTL fails.
@mark.filterwarnings(
    "ignore:Solution may be inaccurate.",
    "ignore:You are solving a parameterized problem that is not DPP.",
)
@mark.parametrize(["aggregator", "matrix"], standard_pairs)
def test_expected_structure(aggregator: NashMTL, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.filterwarnings("ignore:You are solving a parameterized problem that is not DPP.")
@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: NashMTL, matrix: Tensor) -> None:
    assert_non_differentiable(aggregator, matrix)


@mark.filterwarnings("ignore: You are solving a parameterized problem that is not DPP.")
def test_nash_mtl_reset() -> None:
    """
    Tests that the reset method of NashMTL correctly resets its internal state, by verifying that
    the result is the same after reset as it is right after instantiation.

    To ensure that the aggregations are not all the same, we create different matrices to aggregate.
    """

    matrices = [randn_(3, 5) for _ in range(4)]
    aggregator = NashMTL(n_tasks=3, update_weights_every=3)
    expecteds = [aggregator(matrix) for matrix in matrices]

    aggregator.reset()
    results = [aggregator(matrix) for matrix in matrices]

    for result, expected in zip(results, expecteds, strict=True):
        assert_close(result, expected)


def test_representations() -> None:
    A = NashMTL(n_tasks=2, max_norm=1.5, update_weights_every=2, optim_niter=5)
    assert repr(A) == "NashMTL(n_tasks=2, max_norm=1.5, update_weights_every=2, optim_niter=5)"
    assert str(A) == "NashMTL"
