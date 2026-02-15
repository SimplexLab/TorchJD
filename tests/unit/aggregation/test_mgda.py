from pytest import mark
from torch import Tensor
from torch.testing import assert_close
from utils.tensors import ones_, randn_

from torchjd._linalg import compute_gramian
from torchjd.aggregation import MGDA
from torchjd.aggregation._mgda import MGDAWeighting

from ._asserts import (
    assert_expected_structure,
    assert_non_conflicting,
    assert_permutation_invariant,
)
from ._inputs import scaled_matrices, typical_matrices

scaled_pairs = [(MGDA(), matrix) for matrix in scaled_matrices]
typical_pairs = [(MGDA(), matrix) for matrix in typical_matrices]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: MGDA, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_non_conflicting(aggregator: MGDA, matrix: Tensor):
    assert_non_conflicting(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: MGDA, matrix: Tensor):
    assert_permutation_invariant(aggregator, matrix)


@mark.parametrize(
    "shape",
    [
        (5, 7),
        (9, 37),
        (2, 14),
        (32, 114),
        (50, 100),
    ],
)
def test_mgda_satisfies_kkt_conditions(shape: tuple[int, int]):
    matrix = randn_(shape)
    gramian = compute_gramian(matrix)

    weighting = MGDAWeighting(epsilon=1e-05, max_iters=1000)
    weights = weighting(gramian)

    output_direction = gramian @ weights  # Stationarity
    lamb = -weights @ output_direction  # Complementary slackness
    mu = output_direction + lamb

    # Primal feasibility
    positive_weights = weights[weights >= 0]
    assert_close(positive_weights.norm(), weights.norm())

    weights_sum = weights.sum()
    assert_close(weights_sum, ones_([]))

    # Dual feasibility
    positive_mu = mu[mu >= 0]
    assert_close(positive_mu.norm(), mu.norm(), atol=1e-02, rtol=0.0)


def test_representations():
    A = MGDA(epsilon=0.001, max_iters=100)
    assert repr(A) == "MGDA(epsilon=0.001, max_iters=100)"
    assert str(A) == "MGDA"
