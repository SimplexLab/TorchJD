import pytest

pytest.importorskip("scipy")

from pytest import mark
from torch import Tensor
from utils.tensors import ones_

from torchjd.aggregation import FairGrad, Mean
from torchjd.aggregation._fairgrad import FairGradWeighting

from ._asserts import assert_expected_structure, assert_non_conflicting, assert_non_differentiable
from ._inputs import scaled_matrices, typical_matrices

# max_iters=1 is enough to produce a finite output for structure tests.
scaled_pairs = [(FairGrad(alpha=1.0, max_iters=1), matrix) for matrix in scaled_matrices]
typical_pairs = [(FairGrad(alpha=1.0, max_iters=1), matrix) for matrix in typical_matrices]
requires_grad_pairs = [(FairGrad(alpha=1.0, max_iters=1), ones_(3, 5, requires_grad=True))]
# max_iters=100 is sufficient for convergence on the base matrices.
non_conflicting_pairs = [
    (FairGrad(alpha=0.1, max_iters=100), matrix) for matrix in typical_matrices
]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: FairGrad, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: FairGrad, matrix: Tensor) -> None:
    assert_non_differentiable(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], non_conflicting_pairs)
def test_non_conflicting(aggregator: FairGrad, matrix: Tensor) -> None:
    assert_non_conflicting(aggregator, matrix)


@mark.parametrize(
    "alpha",
    [-1.0, 0.0, 0.1, 1.0, 10.0],
)
def test_alpha_check(alpha: float) -> None:
    _ = FairGrad(alpha=alpha)


def test_representations() -> None:
    A = FairGrad(alpha=0.1, max_iters=None)
    assert repr(A) == "FairGrad(alpha=0.1, max_iters=None)"
    assert str(A) == "0.1-FairGrad"

    W = FairGradWeighting(alpha=0.1, max_iters=None)
    assert repr(W) == "FairGradWeighting(alpha=0.1, max_iters=None)"
    assert str(W) == "0.1-FairGradWeighting"


def test_alpha_setter_updates_value() -> None:
    A = FairGrad(alpha=1.0)
    A.alpha = 2.0
    assert A.alpha == 2.0
    assert A.gramian_weighting.alpha == 2.0


def test_max_iters_setter_updates_value() -> None:
    A = FairGrad(alpha=1.0)
    A.max_iters = 50
    assert A.max_iters == 50
    assert A.gramian_weighting.max_iters == 50


def test_alpha_zero_gives_uniform_weights() -> None:
    aggregator = FairGrad(alpha=0.0)
    mean = Mean()
    for matrix in typical_matrices:
        assert aggregator(matrix).allclose(mean(matrix))
