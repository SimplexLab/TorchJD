import torch
from pytest import raises
from torch import Tensor
from torch.testing import assert_close
from utils.tensors import randn_, tensor_

from torchjd.aggregation._modo import MoDoWeighting


def _project_to_simplex(y: Tensor) -> Tensor:
    """Reference Euclidean projection onto the probability simplex, used to derive expected
    values independently of the implementation."""

    m = len(y)
    sorted_y = torch.sort(y, descending=True)[0]
    tmpsum = y.new_zeros(())
    tmax_f = (torch.sum(y) - 1.0) / m
    for i in range(m - 1):
        tmpsum = tmpsum + sorted_y[i]
        tmax = (tmpsum - 1.0) / (i + 1.0)
        if tmax > sorted_y[i + 1]:
            tmax_f = tmax
            break
    return torch.max(y - tmax_f, y.new_zeros(m))


def test_representations() -> None:
    W = MoDoWeighting(gamma=0.1, rho=0.05)
    assert repr(W) == "MoDoWeighting(gamma=0.1, rho=0.05)"


def test_reset_restores_first_step_behavior() -> None:
    J1 = randn_((3, 8))
    J2 = randn_((3, 8))
    G = J1 @ J2.T
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

    J1 = randn_((4, 10))
    J2 = randn_((4, 10))
    G = J1 @ J2.T
    W = MoDoWeighting(gamma=0.1, rho=0.05)
    weights = W(G)
    assert weights.shape == (4,)
    assert (weights >= 0).all()
    assert_close(weights.sum(), tensor_(1.0))


def test_update_recurrence() -> None:
    """Verify one step of the softmax-projected gradient update by hand."""

    gamma = 0.1
    rho = 0.05
    J1 = randn_((3, 8))
    J2 = randn_((3, 8))
    G = J1 @ J2.T
    m = J1.shape[0]

    W = MoDoWeighting(gamma=gamma, rho=rho)
    lambda_0 = tensor_([1.0 / m] * m)
    grad = G @ lambda_0 + rho * lambda_0
    expected = _project_to_simplex(lambda_0 - gamma * grad)

    assert_close(W(G), expected)


def test_two_consecutive_steps() -> None:
    """Verify two consecutive steps of the softmax-projected gradient update."""

    gamma = 0.1
    rho = 0.0
    J1 = randn_((3, 8))
    J2 = randn_((3, 8))
    J3 = randn_((3, 8))
    J4 = randn_((3, 8))
    G1 = J1 @ J2.T
    G2 = J3 @ J4.T
    m = J1.shape[0]

    W = MoDoWeighting(gamma=gamma, rho=rho)

    lambda_0 = tensor_([1.0 / m] * m)
    grad_1 = G1 @ lambda_0 + rho * lambda_0
    lambda_1 = _project_to_simplex(lambda_0 - gamma * grad_1)

    grad_2 = G2 @ lambda_1 + rho * lambda_1
    lambda_2 = _project_to_simplex(lambda_1 - gamma * grad_2)

    assert_close(W(G1), lambda_1)
    assert_close(W(G2), lambda_2)


def test_changing_m_auto_resets() -> None:
    """When the number of objectives changes, the state is re-initialised to uniform."""

    W = MoDoWeighting(gamma=0.1)
    W(randn_((3, 8)) @ randn_((3, 8)).T)
    # After a state-resetting call with m=2, the first output should equal the uniform step's output.
    fresh = MoDoWeighting(gamma=0.1)
    J1 = randn_((2, 8))
    J2 = randn_((2, 8))
    G = J1 @ J2.T
    assert_close(W(G), fresh(G))


def test_non_differentiable() -> None:
    """The _NonDifferentiable mixin must prevent autograd graph construction."""

    G = randn_((3, 8)) @ randn_((3, 8)).T
    G.requires_grad_(True)
    W = MoDoWeighting()
    weights = W(G)
    assert not weights.requires_grad


def test_non_symmetric_input() -> None:
    """MoDoWeighting must accept and correctly process a non-symmetric cross-batch matrix."""

    gamma = 0.1
    rho = 0.05
    J1 = randn_((3, 8))
    J2 = randn_((3, 8))
    G = J1 @ J2.T  # not symmetric, not PSD in general
    m = J1.shape[0]

    W = MoDoWeighting(gamma=gamma, rho=rho)
    lambda_0 = tensor_([1.0 / m] * m)
    grad = G @ lambda_0 + rho * lambda_0
    expected = _project_to_simplex(lambda_0 - gamma * grad)

    assert_close(W(G), expected)
    assert W(G).shape == (m,)
    assert (W(G) >= 0).all()


def test_projection2simplex_known_values() -> None:
    """The simplex projection matches hand-computed Euclidean projections."""

    # Already-positive input: the deficit (1 - sum) is spread equally, no clamping.
    assert_close(
        MoDoWeighting._projection2simplex(tensor_([0.5, 0.1, 0.1])),
        tensor_([0.6, 0.2, 0.2]),
    )
    # Input with a negative entry: it gets clamped to zero.
    assert_close(
        MoDoWeighting._projection2simplex(tensor_([1.0, 0.0, -0.5])),
        tensor_([1.0, 0.0, 0.0]),
    )
