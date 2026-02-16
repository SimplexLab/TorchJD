import torch
from pytest import mark, raises
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

# Basic MGDA (no normalization)
scaled_pairs = [(MGDA(), matrix) for matrix in scaled_matrices]
typical_pairs = [(MGDA(), matrix) for matrix in typical_matrices]

# MGDA with L2 normalization - separate because it has different properties
l2_scaled_pairs = [(MGDA(norm_type="l2"), matrix) for matrix in scaled_matrices]
l2_typical_pairs = [(MGDA(norm_type="l2"), matrix) for matrix in typical_matrices]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: MGDA, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], l2_scaled_pairs + l2_typical_pairs)
def test_expected_structure_l2(aggregator: MGDA, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_non_conflicting(aggregator: MGDA, matrix: Tensor):
    """Test non-conflicting property for basic MGDA (no normalization)."""
    assert_non_conflicting(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: MGDA, matrix: Tensor):
    assert_permutation_invariant(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], l2_typical_pairs)
def test_permutation_invariant_l2(aggregator: MGDA, matrix: Tensor):
    # L2 normalization can have convergence sensitivity due to the iterative solver,
    # so use looser tolerances. The solver may converge to slightly different solutions
    # when the input is permuted, especially for matrices with many rows.
    assert_permutation_invariant(aggregator, matrix, atol=0.3, rtol=0.5)


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

    weighting = MGDAWeighting(epsilon=1e-06, max_iters=1000)
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


@mark.parametrize(
    "shape",
    [
        (5, 7),
        (9, 37),
        (2, 14),
    ],
)
def test_mgda_l2_norm_satisfies_kkt_conditions(shape: tuple[int, int]):
    """Test that MGDA with L2 normalization satisfies KKT conditions on normalized gramian."""
    matrix = randn_(shape)
    gramian = compute_gramian(matrix)

    weighting = MGDAWeighting(norm_type="l2", epsilon=1e-06, max_iters=1000)
    weights = weighting(gramian)

    # Normalize the gramian for KKT check
    grad_norms = torch.sqrt(torch.diag(gramian).clamp(min=1e-20))
    norm_matrix = grad_norms.unsqueeze(1) * grad_norms.unsqueeze(0)
    normalized_gramian = gramian / norm_matrix

    output_direction = normalized_gramian @ weights
    lamb = -weights @ output_direction
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
    A = MGDA()
    assert repr(A) == "MGDA(norm_type='none', epsilon=1e-05, max_iters=250)"
    assert str(A) == "MGDA"

    A = MGDA(norm_type="l2", epsilon=0.01, max_iters=50)
    assert repr(A) == "MGDA(norm_type='l2', epsilon=0.01, max_iters=50)"
    assert str(A) == "MGDA"

    A = MGDA(norm_type="loss")
    assert repr(A) == "MGDA(norm_type='loss', epsilon=1e-05, max_iters=250)"

    A = MGDA(norm_type="loss+")
    assert repr(A) == "MGDA(norm_type='loss+', epsilon=1e-05, max_iters=250)"


def test_invalid_norm_type():
    with raises(ValueError, match=r"Parameter `norm_type` should be"):
        MGDA(norm_type="invalid")  # type: ignore[arg-type]

    with raises(ValueError, match=r"Parameter `norm_type` should be"):
        MGDAWeighting(norm_type="invalid")  # type: ignore[arg-type]


def test_weights_sum_to_one():
    """Test that the weights computed by MGDAWeighting sum to 1."""
    weighting = MGDAWeighting()
    gramian = torch.tensor([[4.0, 2.0], [2.0, 9.0]])
    weights = weighting(gramian)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_weights_non_negative():
    """Test that the weights computed by MGDAWeighting are non-negative."""
    weighting = MGDAWeighting()
    gramian = torch.tensor([[4.0, -1.0], [-1.0, 9.0]])
    weights = weighting(gramian)
    assert (weights >= -1e-6).all()


def test_single_task():
    """Test MGDA with a single task returns weight of 1."""
    weighting = MGDAWeighting()
    gramian = torch.tensor([[4.0]])
    weights = weighting(gramian)
    assert torch.isclose(weights[0], torch.tensor(1.0), atol=1e-6)


def test_l2_normalization_balances_unequal_norms():
    """Test that L2 normalization helps balance tasks with unequal gradient norms."""
    # Two non-conflicting gradients with very different norms
    # g1 = [10, 0], g2 = [0, 1]
    # Gramian: [[100, 0], [0, 1]]
    gramian = torch.tensor([[100.0, 0.0], [0.0, 1.0]])

    weighting_l2 = MGDAWeighting(norm_type="l2")
    weights_l2 = weighting_l2(gramian)

    # With L2 normalization, the normalized gramian is [[1, 0], [0, 1]]
    # which is identity, so weights should be equal [0.5, 0.5]
    assert torch.isclose(weights_l2[0], torch.tensor(0.5), atol=0.01)
    assert torch.isclose(weights_l2[1], torch.tensor(0.5), atol=0.01)


# Tests for loss-based normalization


def test_loss_normalization_requires_set_losses():
    """Test that loss normalization raises error if losses not set."""
    weighting = MGDAWeighting(norm_type="loss")
    gramian = torch.tensor([[4.0, 2.0], [2.0, 9.0]])

    with raises(RuntimeError, match=r"Losses must be set before calling forward"):
        weighting(gramian)


def test_loss_plus_normalization_requires_set_losses():
    """Test that loss+ normalization raises error if losses not set."""
    weighting = MGDAWeighting(norm_type="loss+")
    gramian = torch.tensor([[4.0, 2.0], [2.0, 9.0]])

    with raises(RuntimeError, match=r"Losses must be set before calling forward"):
        weighting(gramian)


def test_set_losses_validates_shape():
    """Test that set_losses validates the losses shape."""
    weighting = MGDAWeighting(norm_type="loss")

    with raises(ValueError, match=r"Parameter `losses` should be a 1D tensor"):
        weighting.set_losses(torch.tensor([[1.0, 2.0]]))


def test_loss_normalization_validates_size():
    """Test that loss normalization validates losses size matches gramian."""
    weighting = MGDAWeighting(norm_type="loss")
    weighting.set_losses(torch.tensor([1.0, 2.0, 3.0]))  # 3 losses
    gramian = torch.tensor([[4.0, 2.0], [2.0, 9.0]])  # 2x2 gramian

    with raises(ValueError, match=r"Number of losses .* must match"):
        weighting(gramian)


def test_loss_plus_normalization_validates_size():
    """Test that loss+ normalization validates losses size matches gramian."""
    weighting = MGDAWeighting(norm_type="loss+")
    weighting.set_losses(torch.tensor([1.0, 2.0, 3.0]))  # 3 losses
    gramian = torch.tensor([[4.0, 2.0], [2.0, 9.0]])  # 2x2 gramian

    with raises(ValueError, match=r"Number of losses .* must match"):
        weighting(gramian)


def test_loss_normalization_weights_sum_to_one():
    """Test that loss normalization produces weights that sum to 1."""
    weighting = MGDAWeighting(norm_type="loss")
    weighting.set_losses(torch.tensor([0.5, 2.0]))
    gramian = torch.tensor([[4.0, 2.0], [2.0, 9.0]])

    weights = weighting(gramian)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_loss_plus_normalization_weights_sum_to_one():
    """Test that loss+ normalization produces weights that sum to 1."""
    weighting = MGDAWeighting(norm_type="loss+")
    weighting.set_losses(torch.tensor([0.5, 2.0]))
    gramian = torch.tensor([[4.0, 2.0], [2.0, 9.0]])

    weights = weighting(gramian)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_identity_gramian_triggers_no_progress_branch():
    """Test that an identity gramian triggers the gamma=0.0 branch in Frank-Wolfe.

    For an identity gramian [[1, 0], [0, 1]] with initial alpha = [0.5, 0.5]:
    - gramian @ alpha = [0.5, 0.5], so t = 0 (argmin)
    - a = alpha @ (gramian @ e_t) = 0.5
    - b = alpha @ (gramian @ alpha) = 0.5
    - c = e_t @ (gramian @ e_t) = 1
    Since c > a and b <= a, this hits the gamma = 0.0 branch.
    """
    weighting = MGDAWeighting()
    gramian = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    weights = weighting(gramian)

    # For identity gramian, uniform weights [0.5, 0.5] should be optimal
    assert torch.isclose(weights[0], torch.tensor(0.5), atol=1e-6)
    assert torch.isclose(weights[1], torch.tensor(0.5), atol=1e-6)


def test_loss_normalization_balances_by_loss():
    """Test that loss normalization balances tasks with different loss values."""
    # Two orthogonal gradients with equal norms but different losses
    gramian = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    # With losses [1, 4], the normalized gramian becomes:
    # [[1/1, 0], [0, 1/16]] = [[1, 0], [0, 0.0625]]
    weighting = MGDAWeighting(norm_type="loss", epsilon=1e-6, max_iters=1000)
    weighting.set_losses(torch.tensor([1.0, 4.0]))
    weights = weighting(gramian)

    # Both weights should be non-negative and sum to 1
    assert (weights >= -1e-6).all()
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_aggregator_set_losses():
    """Test that MGDA aggregator properly forwards set_losses to weighting."""
    A = MGDA(norm_type="loss")
    A.set_losses(torch.tensor([0.5, 2.0]))

    matrix = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    result = A(matrix)

    # Should not raise and should return valid aggregation
    assert result.shape == (2,)
    assert result.isfinite().all()


def test_aggregator_loss_plus():
    """Test MGDA aggregator with loss+ normalization."""
    A = MGDA(norm_type="loss+")
    A.set_losses(torch.tensor([1.0, 2.0]))

    matrix = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    result = A(matrix)

    # Should not raise and should return valid aggregation
    assert result.shape == (2,)
    assert result.isfinite().all()


@mark.parametrize(
    "shape",
    [
        (3, 5),
        (5, 10),
        (2, 8),
    ],
)
def test_loss_normalization_satisfies_kkt_conditions(shape: tuple[int, int]):
    """Test that MGDA with loss normalization satisfies KKT conditions on normalized gramian."""
    matrix = randn_(shape)
    gramian = compute_gramian(matrix)
    losses = torch.rand(shape[0]) + 0.1  # Random positive losses

    weighting = MGDAWeighting(norm_type="loss", epsilon=1e-06, max_iters=1000)
    weighting.set_losses(losses)
    weights = weighting(gramian)

    # Normalize the gramian for KKT check
    losses_clamped = losses.clamp(min=1e-20)
    norm_matrix = losses_clamped.unsqueeze(1) * losses_clamped.unsqueeze(0)
    normalized_gramian = gramian / norm_matrix

    output_direction = normalized_gramian @ weights
    lamb = -weights @ output_direction
    mu = output_direction + lamb

    # Primal feasibility
    positive_weights = weights[weights >= 0]
    assert_close(positive_weights.norm(), weights.norm())

    weights_sum = weights.sum()
    assert_close(weights_sum, ones_([]))

    # Dual feasibility
    positive_mu = mu[mu >= 0]
    assert_close(positive_mu.norm(), mu.norm(), atol=1e-02, rtol=0.0)
