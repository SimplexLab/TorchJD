import torch
from pytest import mark, raises
from torch import Tensor
from torch.testing import assert_close
from utils.tensors import randn_, tensor_

from torchjd.aggregation._aggregator_bases import GramianWeightedAggregator
from torchjd.aggregation._modo import MoDoWeighting

from ._asserts import assert_expected_structure
from ._inputs import scaled_matrices, typical_matrices

gramian_pairs = [
    (GramianWeightedAggregator(MoDoWeighting()), m) for m in typical_matrices + scaled_matrices
]


def test_representations() -> None:
    W = MoDoWeighting(gamma=0.1, rho=0.05)
    assert repr(W) == "MoDoWeighting(gamma=0.1, rho=0.05)"


@mark.parametrize(["aggregator", "matrix"], gramian_pairs)
def test_expected_structure_gramian_weighting(
    aggregator: GramianWeightedAggregator, matrix: Tensor
) -> None:
    assert_expected_structure(aggregator, matrix)


def test_reset_restores_first_step_behavior() -> None:
    J = randn_((3, 8))
    G = J @ J.T
    W = MoDoWeighting(gamma=0.1)
    first = W(G)
    W(G)
    W.reset()
    assert_close(first, W(G))


def test_gamma_setter_accepts_valid() -> None:
    W = MoDoWeighting()
    W.gamma = 0.01
    assert W.gamma == 0.01
    W.gamma = 0.1
    assert W.gamma == 0.1
    W.gamma = 1.0
    assert W.gamma == 1.0


def test_gamma_setter_rejects_non_positive() -> None:
    W = MoDoWeighting()
    with raises(ValueError, match="gamma"):
        W.gamma = 0.0
    with raises(ValueError, match="gamma"):
        W.gamma = -0.1


def test_rho_setter_accepts_valid() -> None:
    W = MoDoWeighting()
    W.rho = 0.0
    assert W.rho == 0.0
    W.rho = 0.1
    assert W.rho == 0.1


def test_rho_setter_rejects_negative() -> None:
    W = MoDoWeighting()
    with raises(ValueError, match="rho"):
        W.rho = -0.1


def test_output_lies_on_simplex() -> None:
    """The softmax projection ensures the weights sum to 1 and are non-negative."""

    J = randn_((4, 10))
    G = J @ J.T
    W = MoDoWeighting(gamma=0.1, rho=0.05)
    weights = W(G)
    assert weights.shape == (4,)
    assert (weights >= 0).all()
    assert_close(weights.sum(), tensor_(1.0))


def test_small_gamma_stays_near_uniform() -> None:
    """With a tiny gamma, one step barely moves lambda from the uniform initialisation."""

    J = randn_((3, 8))
    G = J @ J.T
    m = J.shape[0]
    W = MoDoWeighting(gamma=1e-8)
    uniform = tensor_([1.0 / m] * m)
    assert_close(W(G), uniform, atol=1e-6, rtol=1e-6)


def test_update_recurrence() -> None:
    """Verify one step of the softmax-projected gradient update by hand."""

    gamma = 0.1
    rho = 0.05
    J = randn_((3, 8))
    G = J @ J.T
    m = J.shape[0]

    W = MoDoWeighting(gamma=gamma, rho=rho)
    lambda_0 = tensor_([1.0 / m] * m)
    grad = G @ lambda_0 + rho * lambda_0
    expected = torch.softmax(lambda_0 - gamma * grad, dim=-1)

    assert_close(W(G), expected)


def test_two_consecutive_steps() -> None:
    """Verify two consecutive steps of the softmax-projected gradient update."""

    gamma = 0.1
    rho = 0.0
    J1 = randn_((3, 8))
    J2 = randn_((3, 8))
    G1 = J1 @ J1.T
    G2 = J2 @ J2.T
    m = J1.shape[0]

    W = MoDoWeighting(gamma=gamma, rho=rho)

    lambda_0 = tensor_([1.0 / m] * m)
    grad_1 = G1 @ lambda_0 + rho * lambda_0
    lambda_1 = torch.softmax(lambda_0 - gamma * grad_1, dim=-1)

    grad_2 = G2 @ lambda_1 + rho * lambda_1
    lambda_2 = torch.softmax(lambda_1 - gamma * grad_2, dim=-1)

    assert_close(W(G1), lambda_1)
    assert_close(W(G2), lambda_2)


def test_changing_m_auto_resets() -> None:
    """When the number of objectives changes, the state is re-initialised to uniform."""

    W = MoDoWeighting(gamma=0.1)
    W(randn_((3, 8)) @ randn_((3, 8)).T)
    # After a state-resetting call with m=2, the first output should equal the uniform step's output.
    fresh = MoDoWeighting(gamma=0.1)
    J = randn_((2, 8))
    G = J @ J.T
    assert_close(W(G), fresh(G))


def test_zero_rows() -> None:
    """A (0, 0) Gramian yields an empty weight vector."""

    W = MoDoWeighting()
    weights = W(tensor_([]).reshape(0, 0))
    assert weights.shape == (0,)
