import torch
from pytest import raises
from torch.testing import assert_close
from utils.tensors import eye_, randn_, tensor_

from torchjd.aggregation._sdmgrad import SDMGradWeighting


def test_representations() -> None:
    W = SDMGradWeighting(lr=10.0, momentum=0.5, n_iter=20, lambda_=0.3)
    assert (
        repr(W)
        == "SDMGradWeighting(lr=10.0, momentum=0.5, n_iter=20, lambda_=0.3, pref_vector=None)"
    )


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


def test_lambda_setter_accepts_valid() -> None:
    W = SDMGradWeighting()
    W.lambda_ = 0.0
    assert W.lambda_ == 0.0
    W.lambda_ = 0.6
    assert W.lambda_ == 0.6


def test_lambda_setter_rejects_negative() -> None:
    W = SDMGradWeighting()
    with raises(ValueError, match="lambda_"):
        W.lambda_ = -0.1


def test_pref_vector_setter_rejects_non_1d() -> None:
    W = SDMGradWeighting()
    with raises(ValueError, match="pref_vector"):
        W.pref_vector = tensor_([[0.5, 0.5], [0.5, 0.5]])


def test_output_lies_on_simplex() -> None:
    """The simplex projection and (1 + lambda_) normalization keep the weights on the simplex."""

    J1 = randn_((4, 10))
    J2 = randn_((4, 10))
    A = J1 @ J2.T
    W = SDMGradWeighting(lambda_=0.3)
    weights = W(A)
    assert weights.shape == (4,)
    assert (weights >= 0).all()
    assert_close(weights.sum(), tensor_(1.0))


def test_update_recurrence() -> None:
    """One inner-solve step matches the manually-computed expected output.

    With A = diag(2, 1, 0), n_iter=1, lr=10, lambda_=0.3, starting from uniform w=[1/3,1/3,1/3]:
      grad = A @ 1.3*[1/3,1/3,1/3] = [13/15, 13/30, 0]
      w - lr*grad = [-25/3, -4, 1/3]  ->  projected to [0, 0, 1]
      return ([0,0,1] + 0.3*[1/3,...]) / 1.3 = [1/13, 1/13, 11/13]
    """
    A = torch.diag(tensor_([2.0, 1.0, 0.0]))
    W = SDMGradWeighting(lr=10.0, momentum=0.5, n_iter=1, lambda_=0.3)
    assert_close(W(A), tensor_([1 / 13, 1 / 13, 11 / 13]))


def test_two_consecutive_steps() -> None:
    """Warm-started carry-over across two consecutive calls matches manually-computed values.

    Step 1: A1=diag(2,1,0) -> state w=[0,0,1], return [1/13, 1/13, 11/13]  (see test_update_recurrence)
    Step 2: A2=eye(3), warm start w=[0,0,1]:
      grad = [0.1, 0.1, 1.1]; w - lr*grad = [-1,-1,-10] -> projected to [0.5, 0.5, 0]
      return ([0.5,0.5,0] + 0.3*[1/3,...]) / 1.3 = [6/13, 6/13, 1/13]
    """
    A1 = torch.diag(tensor_([2.0, 1.0, 0.0]))
    A2 = eye_(3)
    W = SDMGradWeighting(lr=10.0, momentum=0.5, n_iter=1, lambda_=0.3)
    assert_close(W(A1), tensor_([1 / 13, 1 / 13, 11 / 13]))
    assert_close(W(A2), tensor_([6 / 13, 6 / 13, 1 / 13]))


def test_custom_pref_vector() -> None:
    """A custom preference vector changes the output relative to the uniform default.

    With A=diag(2,1,0), pref=[0,0,1], n_iter=1, lr=10, lambda_=0.3:
      grad = A @ ([1/3,1/3,1/3] + 0.3*[0,0,1]) = [2/3, 1/3, 0]
      w - lr*grad = [-19/3, -3, 1/3] -> projected to [0, 0, 1]
      return ([0,0,1] + 0.3*[0,0,1]) / 1.3 = [0, 0, 1]
    This differs from the uniform-pref result [1/13, 1/13, 11/13].
    """
    A = torch.diag(tensor_([2.0, 1.0, 0.0]))
    pref = tensor_([0.0, 0.0, 1.0])
    W_pref = SDMGradWeighting(lr=10.0, momentum=0.5, n_iter=1, lambda_=0.3, pref_vector=pref)
    assert_close(W_pref(A), tensor_([0.0, 0.0, 1.0]))

    W_default = SDMGradWeighting(lr=10.0, momentum=0.5, n_iter=1, lambda_=0.3)
    assert not torch.allclose(W_default(A), tensor_([0.0, 0.0, 1.0]))


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
