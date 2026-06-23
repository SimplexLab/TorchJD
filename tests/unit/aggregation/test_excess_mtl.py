import torch
from pytest import raises
from torch.testing import assert_close
from utils.tensors import randn_, tensor_

from torchjd.aggregation._excess_mtl import ExcessMTL, ExcessMTLWeighting


def test_representations() -> None:
    W = ExcessMTLWeighting(robust_step_size=1.0, n_warmup_steps=0)
    assert repr(W) == "ExcessMTLWeighting(robust_step_size=1.0, n_warmup_steps=0)"


def test_reset_restores_first_step_behavior() -> None:
    J = randn_((3, 8))
    W = ExcessMTLWeighting()
    first = W(J)
    W(J)
    W.reset()
    assert_close(first, W(J))


def test_robust_step_size_setter_accepts_valid() -> None:
    W = ExcessMTLWeighting()
    W.robust_step_size = 0.1
    assert W.robust_step_size == 0.1
    W.robust_step_size = 10.0
    assert W.robust_step_size == 10.0


def test_robust_step_size_setter_rejects_non_positive() -> None:
    W = ExcessMTLWeighting()
    with raises(ValueError, match="robust_step_size"):
        W.robust_step_size = 0.0
    with raises(ValueError, match="robust_step_size"):
        W.robust_step_size = -1.0


def test_n_warmup_steps_setter_accepts_valid() -> None:
    W = ExcessMTLWeighting()
    W.n_warmup_steps = 0
    assert W.n_warmup_steps == 0
    W.n_warmup_steps = 100
    assert W.n_warmup_steps == 100


def test_n_warmup_steps_setter_rejects_negative() -> None:
    W = ExcessMTLWeighting()
    with raises(ValueError, match="n_warmup_steps"):
        W.n_warmup_steps = -1


def test_output_lies_on_simplex() -> None:
    """The exponentiated update followed by normalisation keeps the weights on the simplex."""

    J = randn_((4, 10))
    W = ExcessMTLWeighting()
    # Call twice so the second call exercises the normalised-w branch
    W(J)
    weights = W(J)
    assert weights.shape == (4,)
    assert (weights >= 0).all()
    assert_close(weights.sum(), tensor_(1.0))


def test_warmup_returns_uniform() -> None:
    """During warmup every call must return [1/m, ..., 1/m] regardless of the input."""

    m, n_warmup = 3, 5
    W = ExcessMTLWeighting(n_warmup_steps=n_warmup)
    expected = tensor_([1.0 / m] * m)
    for _ in range(n_warmup):
        assert_close(W(randn_((m, 8))), expected)


def test_weights_change_after_warmup() -> None:
    """After warmup ends the weights must diverge from uniform when tasks have different excess risks."""

    W = ExcessMTLWeighting(n_warmup_steps=2, robust_step_size=1.0)
    # Symmetric warmup: equal excess risk for both tasks → equal initial_w
    J_sym = tensor_([[1.0, 0.0], [1.0, 0.0]])
    W(J_sym)
    W(J_sym)

    # Asymmetric step: task 0 has larger gradient → higher excess → weight must exceed task 1
    J_unequal = tensor_([[2.0, 0.0], [1.0, 0.0]])
    weights = W(J_unequal)
    assert weights[0] > weights[1]


def test_two_consecutive_steps() -> None:
    """Verify warm-started carry-over across two calls.

    Call 1: J = [[2., 0.], [1., 0.]] → weights = [e^2, e] / (e^2 + e)  (from test above)
    Call 2: J = [[1., 0.], [2., 0.]]
      grad_sum = [[4+1., 0.], [1+4., 0.]] = [[5., 0.], [5., 0.]]
      h ≈ [[sqrt(5), sqrt(eps)], [sqrt(5), sqrt(eps)]]
      w = [1/sqrt(5), 4/sqrt(5)]
      initial_w = [2, 1]  (from call 1)
      w_norm = [1/(2*sqrt(5)),  4/sqrt(5)]
      weights_2 = weights_1 * [exp(w_norm_0), exp(w_norm_1)] / normalization
    """
    J1 = tensor_([[2.0, 0.0], [1.0, 0.0]])
    J2 = tensor_([[1.0, 0.0], [2.0, 0.0]])
    W = ExcessMTLWeighting(robust_step_size=1.0, n_warmup_steps=0)

    e2 = torch.exp(tensor_(2.0))
    e1 = torch.exp(tensor_(1.0))
    weights_1 = tensor_([e2 / (e2 + e1), e1 / (e2 + e1)])
    assert_close(W(J1), weights_1)

    sqrt5 = torch.sqrt(tensor_(5.0))
    w_norm_0 = tensor_(1.0) / (tensor_(2.0) * sqrt5)
    w_norm_1 = tensor_(4.0) / sqrt5
    unnorm_0 = weights_1[0] * torch.exp(w_norm_0)
    unnorm_1 = weights_1[1] * torch.exp(w_norm_1)
    weights_2 = tensor_([unnorm_0 / (unnorm_0 + unnorm_1), unnorm_1 / (unnorm_0 + unnorm_1)])
    assert_close(W(J2), weights_2)


def test_warmup_baseline_is_average() -> None:
    """initial_w after warmup must equal the average excess risk collected during warmup.

    With n_warmup_steps=2 and J1=[[2,0],[1,0]], J2=[[1,0],[2,0]]:

    Warmup call 1 — sq_grad_sum_1 = J1**2 = [[4,0],[1,0]]:
      h_1 ≈ [[2, sqrt(eps)], [1, sqrt(eps)]]
      w_1 = [4/2, 1/1] = [2, 1]

    Warmup call 2 — sq_grad_sum_2 = J1**2 + J2**2 = [[5,0],[5,0]]:
      h_2 ≈ [[sqrt(5), sqrt(eps)], [sqrt(5), sqrt(eps)]]
      w_2 = [1/sqrt(5), 4/sqrt(5)]

    initial_w = (w_1 + w_2) / 2  (Appendix C.1 average)

    Post-warmup call 3 with J3 = J1 — sq_grad_sum_3 = [[9,0],[6,0]]:
      h_3 ≈ [[3, sqrt(eps)], [sqrt(6), sqrt(eps)]]
      w_3 = [4/3, 1/sqrt(6)]
      w_norm = w_3 / (initial_w + 1e-7)
      weights = [0.5, 0.5] * exp(w_norm) / normalize
    """

    J1 = tensor_([[2.0, 0.0], [1.0, 0.0]])
    J2 = tensor_([[1.0, 0.0], [2.0, 0.0]])
    J3 = tensor_([[2.0, 0.0], [1.0, 0.0]])
    W = ExcessMTLWeighting(n_warmup_steps=2, robust_step_size=1.0)

    W(J1)  # warmup step 1 — sq_grad_sum becomes J1**2
    W(J2)  # warmup step 2 — sq_grad_sum becomes J1**2 + J2**2

    sq_grad_sum_1 = J1**2
    h_1 = torch.sqrt(sq_grad_sum_1 + 1e-7)
    w_1 = (J1**2 / h_1).sum(dim=1)

    sq_grad_sum_2 = sq_grad_sum_1 + J2**2
    h_2 = torch.sqrt(sq_grad_sum_2 + 1e-7)
    w_2 = (J2**2 / h_2).sum(dim=1)

    initial_w = (w_1 + w_2) / 2  # Appendix C.1 baseline

    sq_grad_sum_3 = sq_grad_sum_2 + J3**2
    h_3 = torch.sqrt(sq_grad_sum_3 + 1e-7)
    w_3 = (J3**2 / h_3).sum(dim=1)
    w_norm = w_3 / (initial_w + 1e-7)
    pre_norm = tensor_([0.5, 0.5]) * torch.exp(w_norm)
    expected = pre_norm / pre_norm.sum()

    assert_close(W(J3), expected)


def test_n_steps_resets_on_m_change() -> None:
    """When the number of objectives changes the warmup counter must restart."""

    W = ExcessMTLWeighting(n_warmup_steps=10)
    # Burn through 5 warmup steps
    for _ in range(5):
        W(randn_((3, 8)))

    # Switch to 2 objectives — state including step counter resets
    fresh = ExcessMTLWeighting(n_warmup_steps=10)
    J = randn_((2, 8))
    assert_close(W(J), fresh(J))


def test_non_differentiable() -> None:
    """The _NonDifferentiable mixin must prevent autograd graph construction."""

    J = randn_((3, 8))
    J.requires_grad_(True)
    W = ExcessMTLWeighting()
    weights = W(J)
    assert not weights.requires_grad


# ExcessMTL (aggregator wrapper) tests


def test_excess_mtl_representations() -> None:
    agg = ExcessMTL(robust_step_size=2.0, n_warmup_steps=3)
    assert repr(agg) == "ExcessMTL(robust_step_size=2.0, n_warmup_steps=3)"


def test_excess_mtl_properties_delegate() -> None:
    agg = ExcessMTL(robust_step_size=1.0, n_warmup_steps=0)
    assert agg.robust_step_size == 1.0
    assert agg.n_warmup_steps == 0

    agg.robust_step_size = 0.5
    assert agg.robust_step_size == 0.5
    assert agg.weighting.robust_step_size == 0.5

    agg.n_warmup_steps = 5
    assert agg.n_warmup_steps == 5
    assert agg.weighting.n_warmup_steps == 5


def test_excess_mtl_reset_delegates() -> None:
    J = randn_((3, 8))
    agg = ExcessMTL(n_warmup_steps=0)
    first = agg(J)
    agg(J)
    agg.reset()
    assert_close(first, agg(J))
