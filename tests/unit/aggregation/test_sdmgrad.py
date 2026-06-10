import torch
from pytest import raises
from torch import Tensor
from torch.testing import assert_close
from utils.tensors import randn_, tensor_

from torchjd.aggregation._sdmgrad import SDMGradWeighting


def _project_to_simplex(y: Tensor) -> Tensor:
    """Reference Euclidean projection onto the probability simplex."""

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


def _sdmgrad_reference(
    A: Tensor,
    w: Tensor,
    w_tilde: Tensor,
    lr: float,
    momentum: float,
    n_iter: int,
    lamda: float,
) -> tuple[Tensor, Tensor]:
    """Reference inner solve. Returns the updated state ``w`` and the returned (normalized) weights."""

    velocity: Tensor | None = None
    for _ in range(n_iter):
        grad = A @ (w + lamda * w_tilde)
        velocity = grad if velocity is None else momentum * velocity + grad
        w = _project_to_simplex(w - lr * velocity)
    return w, (w + lamda * w_tilde) / (1.0 + lamda)


def test_representations() -> None:
    W = SDMGradWeighting(lr=10.0, momentum=0.5, n_iter=20, lamda=0.3)
    assert (
        repr(W) == "SDMGradWeighting(lr=10.0, momentum=0.5, n_iter=20, lamda=0.3, pref_vector=None)"
    )

    W_pref = SDMGradWeighting(pref_vector=tensor_([0.25, 0.75]))
    assert "pref_vector=tensor(" in repr(W_pref)


def test_reset_restores_first_step_behavior() -> None:
    J1 = randn_((3, 8))
    J2 = randn_((3, 8))
    A = J1 @ J2.T
    W = SDMGradWeighting()
    first = W(A)
    W(A)
    W.reset()
    assert_close(first, W(A))


def test_lr_setter_accepts_valid() -> None:
    W = SDMGradWeighting()
    W.lr = 0.5
    assert W.lr == 0.5


def test_lr_setter_rejects_non_positive() -> None:
    W = SDMGradWeighting()
    with raises(ValueError, match="lr"):
        W.lr = 0.0
    with raises(ValueError, match="lr"):
        W.lr = -1.0


def test_momentum_setter_accepts_valid() -> None:
    W = SDMGradWeighting()
    W.momentum = 0.0
    assert W.momentum == 0.0
    W.momentum = 0.9
    assert W.momentum == 0.9


def test_momentum_setter_rejects_out_of_range() -> None:
    W = SDMGradWeighting()
    with raises(ValueError, match="momentum"):
        W.momentum = -0.1
    with raises(ValueError, match="momentum"):
        W.momentum = 1.0


def test_n_iter_setter_rejects_non_positive() -> None:
    W = SDMGradWeighting()
    with raises(ValueError, match="n_iter"):
        W.n_iter = 0


def test_lamda_setter_accepts_valid() -> None:
    W = SDMGradWeighting()
    W.lamda = 0.0
    assert W.lamda == 0.0
    W.lamda = 0.6
    assert W.lamda == 0.6


def test_lamda_setter_rejects_negative() -> None:
    W = SDMGradWeighting()
    with raises(ValueError, match="lamda"):
        W.lamda = -0.1


def test_pref_vector_setter_rejects_non_1d() -> None:
    W = SDMGradWeighting()
    with raises(ValueError, match="pref_vector"):
        W.pref_vector = tensor_([[0.5, 0.5], [0.5, 0.5]])


def test_output_lies_on_simplex() -> None:
    """The simplex projection and (1+lamda) normalization keep the weights on the simplex."""

    J1 = randn_((4, 10))
    J2 = randn_((4, 10))
    A = J1 @ J2.T
    W = SDMGradWeighting(lamda=0.3)
    weights = W(A)
    assert weights.shape == (4,)
    assert (weights >= 0).all()
    assert_close(weights.sum(), tensor_(1.0))


def test_update_recurrence() -> None:
    """Verify one full inner solve by hand."""

    lr, momentum, n_iter, lamda = 10.0, 0.5, 5, 0.3
    J1 = randn_((3, 8))
    J2 = randn_((3, 8))
    A = J1 @ J2.T
    m = J1.shape[0]

    W = SDMGradWeighting(lr=lr, momentum=momentum, n_iter=n_iter, lamda=lamda)
    w0 = tensor_([1.0 / m] * m)
    w_tilde = tensor_([1.0 / m] * m)
    _, expected = _sdmgrad_reference(A, w0, w_tilde, lr, momentum, n_iter, lamda)

    assert_close(W(A), expected)


def test_two_consecutive_steps() -> None:
    """Verify warm-started carry-over across two consecutive calls."""

    lr, momentum, n_iter, lamda = 10.0, 0.5, 5, 0.3
    J1 = randn_((3, 8))
    J2 = randn_((3, 8))
    J3 = randn_((3, 8))
    J4 = randn_((3, 8))
    A1 = J1 @ J2.T
    A2 = J3 @ J4.T
    m = J1.shape[0]

    W = SDMGradWeighting(lr=lr, momentum=momentum, n_iter=n_iter, lamda=lamda)
    w_tilde = tensor_([1.0 / m] * m)

    w0 = tensor_([1.0 / m] * m)
    w1, out1 = _sdmgrad_reference(A1, w0, w_tilde, lr, momentum, n_iter, lamda)
    _, out2 = _sdmgrad_reference(A2, w1, w_tilde, lr, momentum, n_iter, lamda)

    assert_close(W(A1), out1)
    assert_close(W(A2), out2)


def test_custom_pref_vector() -> None:
    """A custom preference vector is used as the target direction and changes the output."""

    lr, momentum, n_iter, lamda = 10.0, 0.5, 5, 0.3
    J1 = randn_((3, 8))
    J2 = randn_((3, 8))
    A = J1 @ J2.T
    m = J1.shape[0]
    pref = tensor_([0.1, 0.2, 0.7])

    W = SDMGradWeighting(lr=lr, momentum=momentum, n_iter=n_iter, lamda=lamda, pref_vector=pref)
    w0 = tensor_([1.0 / m] * m)
    _, expected = _sdmgrad_reference(A, w0, pref, lr, momentum, n_iter, lamda)
    assert_close(W(A), expected)

    # The custom preference vector should change the output compared to the uniform default.
    W_default = SDMGradWeighting(lr=lr, momentum=momentum, n_iter=n_iter, lamda=lamda)
    assert not torch.allclose(W(A), W_default(A))


def test_pref_vector_wrong_length_raises() -> None:
    W = SDMGradWeighting(pref_vector=tensor_([0.5, 0.5]))
    J1 = randn_((3, 8))
    J2 = randn_((3, 8))
    A = J1 @ J2.T
    with raises(ValueError, match="pref_vector"):
        W(A)


def test_changing_m_auto_resets() -> None:
    """When the number of objectives changes, the warm-started state is re-initialised to uniform."""

    W = SDMGradWeighting()
    W(randn_((3, 8)) @ randn_((3, 8)).T)
    fresh = SDMGradWeighting()
    J1 = randn_((2, 8))
    J2 = randn_((2, 8))
    A = J1 @ J2.T
    assert_close(W(A), fresh(A))


def test_non_differentiable() -> None:
    """The _NonDifferentiable mixin must prevent autograd graph construction."""

    A = randn_((3, 8)) @ randn_((3, 8)).T
    A.requires_grad_(True)
    W = SDMGradWeighting()
    weights = W(A)
    assert not weights.requires_grad


def test_non_symmetric_input() -> None:
    """SDMGradWeighting must accept and correctly process a non-symmetric cross-batch matrix."""

    lr, momentum, n_iter, lamda = 10.0, 0.5, 5, 0.3
    J1 = randn_((3, 8))
    J2 = randn_((3, 8))
    A = J1 @ J2.T  # not symmetric, not PSD in general
    m = J1.shape[0]

    W = SDMGradWeighting(lr=lr, momentum=momentum, n_iter=n_iter, lamda=lamda)
    w0 = tensor_([1.0 / m] * m)
    w_tilde = tensor_([1.0 / m] * m)
    _, expected = _sdmgrad_reference(A, w0, w_tilde, lr, momentum, n_iter, lamda)

    assert_close(W(A), expected)
    assert (W(A) >= 0).all()


def test_projection2simplex_known_values() -> None:
    """The simplex projection matches hand-computed Euclidean projections."""

    assert_close(
        SDMGradWeighting._projection2simplex(tensor_([0.5, 0.1, 0.1])),
        tensor_([0.6, 0.2, 0.2]),
    )
    assert_close(
        SDMGradWeighting._projection2simplex(tensor_([1.0, 0.0, -0.5])),
        tensor_([1.0, 0.0, 0.0]),
    )
