from pytest import mark, raises
from torch import Tensor
from torch.testing import assert_close
from utils.optional_deps import base_weighting
from utils.tensors import randn_, tensor_

from torchjd.aggregation import GradVacWeighting, MeanWeighting
from torchjd.aggregation._aggregator_bases import (
    GramianWeightedAggregator,
    WeightedAggregator,
)
from torchjd.aggregation._cr_mogm import CRMOGMWeighting

from ._asserts import assert_expected_structure
from ._inputs import scaled_matrices, typical_matrices

# UPGradWeighting uses a QP solver that can fail on the extreme scales (0.0, 1e15) found in
# scaled_matrices, so the gramian-path structural test only uses typical_matrices.
matrix_pairs = [(WeightedAggregator(CRMOGMWeighting(MeanWeighting())), m) for m in typical_matrices]
gramian_pairs = [
    (GramianWeightedAggregator(CRMOGMWeighting(base_weighting())), m)
    for m in typical_matrices + scaled_matrices
]


@mark.parametrize(["aggregator", "matrix"], matrix_pairs)
def test_expected_structure_matrix_weighting(
    aggregator: WeightedAggregator, matrix: Tensor
) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], gramian_pairs)
def test_expected_structure_gramian_weighting(
    aggregator: GramianWeightedAggregator, matrix: Tensor
) -> None:
    assert_expected_structure(aggregator, matrix)


def test_reset_restores_first_step_behavior() -> None:
    """
    Use ``base_weighting`` so the weights actually depend on the input — with
    ``MeanWeighting`` the EMA would be a fixed point at the uniform weights and the test would
    be trivial.
    """

    J = randn_((3, 8))
    G = J @ J.T
    W = CRMOGMWeighting(base_weighting(), alpha=0.5)
    first = W(G)
    W(G)
    W.reset()
    assert_close(first, W(G))


def test_reset_propagates_to_stateful_weighting() -> None:
    """
    Verify that ``reset()`` calls the wrapped weighting's ``reset()`` when it is
    :class:`~torchjd.aggregation.Stateful`. Checks that ``GradVacWeighting``'s internal
    state is cleared after ``reset()``.
    """

    inner = GradVacWeighting()
    W = CRMOGMWeighting(inner, alpha=0.5)
    J = randn_((3, 8))
    W(J @ J.T)
    assert inner._phi_t is not None
    W.reset()
    assert inner._phi_t is None


def test_changing_m_raises() -> None:
    """Verify that changing the number of objectives after the first call raises a ValueError."""

    W = CRMOGMWeighting(MeanWeighting())
    W(randn_((3, 8)) @ randn_((3, 8)).T)
    with raises(ValueError, match="number of objectives"):
        W(randn_((2, 8)) @ randn_((2, 8)).T)


def test_alpha_setter_accepts_valid() -> None:
    W = CRMOGMWeighting(MeanWeighting())
    W.alpha = 0.0
    assert W.alpha == 0.0
    W.alpha = 0.5
    assert W.alpha == 0.5
    W.alpha = 1.0
    assert W.alpha == 1.0


def test_alpha_setter_rejects_out_of_range() -> None:
    W = CRMOGMWeighting(MeanWeighting())
    with raises(ValueError, match="alpha"):
        W.alpha = -0.1
    with raises(ValueError, match="alpha"):
        W.alpha = 1.1


def test_alpha_zero_reduces_to_bare_weighting() -> None:
    """
    With ``alpha=0`` the previous state is always multiplied by zero, so the smoothed weights
    equal the bare weighting's output on every call — not just the first.
    """

    J = randn_((3, 8))
    G = J @ J.T
    bare = base_weighting()
    smoothed = CRMOGMWeighting(base_weighting(), alpha=0.0)

    expected = bare(G)
    assert_close(smoothed(G), expected)
    assert_close(smoothed(G), expected)


def test_alpha_one_freezes_weights() -> None:
    """
    With ``alpha=1`` the fresh weights are multiplied by zero, so the smoothed weights stay at
    their initial value forever. When ``initial_weights`` is ``None``, the initial value is
    :math:`\\hat{\\lambda}_1`, so the output is frozen at the first step's bare weights.
    """

    J = randn_((3, 8))
    G = J @ J.T
    W = CRMOGMWeighting(base_weighting(), alpha=1.0)
    first = W(G)

    assert_close(W(G), first)
    assert_close(W(G), first)


def test_ema_is_applied() -> None:
    """Run two steps with ``alpha=0.9`` and check the EMA recurrence by hand."""

    alpha = 0.9
    J1 = randn_((3, 8))
    J2 = randn_((3, 8))
    G1 = J1 @ J1.T
    G2 = J2 @ J2.T

    bare = base_weighting()
    smoothed = CRMOGMWeighting(base_weighting(), alpha=alpha)

    lambda_hat_1 = bare(G1)
    lambda_hat_2 = bare(G2)

    # lambda_0 = lambda_hat_1, so lambda_1 = lambda_hat_1 regardless of alpha
    expected_1 = lambda_hat_1
    expected_2 = alpha * lambda_hat_1 + (1.0 - alpha) * lambda_hat_2

    assert_close(smoothed(G1), expected_1)
    assert_close(smoothed(G2), expected_2)


def test_initial_weights_used_as_lambda_0() -> None:
    """Verify that when ``initial_weights`` is provided it acts as :math:`\\lambda_0`."""

    alpha = 0.5
    J = randn_((3, 8))
    G = J @ J.T
    initial = tensor_([0.5, 0.3, 0.2])

    bare = base_weighting()
    W = CRMOGMWeighting(base_weighting(), alpha=alpha, initial_weights=initial)

    lambda_hat_1 = bare(G)
    expected_1 = alpha * initial + (1.0 - alpha) * lambda_hat_1

    assert_close(W(G), expected_1)


def test_reset_restores_initial_weights() -> None:
    """Verify that ``reset()`` restores the user-provided ``initial_weights`` as :math:`\\lambda_0`."""

    alpha = 0.5
    J = randn_((3, 8))
    G = J @ J.T
    initial = tensor_([0.5, 0.3, 0.2])

    W = CRMOGMWeighting(base_weighting(), alpha=alpha, initial_weights=initial)
    first = W(G)
    W(G)
    W.reset()
    assert_close(W(G), first)


def test_initial_weights_shape_mismatch_raises() -> None:
    """Verify that mismatched ``initial_weights`` shape raises a ``ValueError``."""

    W = CRMOGMWeighting(MeanWeighting(), initial_weights=tensor_([0.5, 0.5]))
    with raises(ValueError, match="initial_weights"):
        W(randn_((3, 8)) @ randn_((3, 8)).T)


def test_zero_columns() -> None:
    """
    A ``(2, 0)`` matrix has no columns to combine, so the aggregation must be empty. Zero-row
    inputs are intentionally not tested: ``MeanWeighting`` does ``1/m`` in Python and would
    raise ``ZeroDivisionError`` at ``m=0``, which is the wrapped weighting's responsibility.
    """

    aggregator = WeightedAggregator(CRMOGMWeighting(MeanWeighting()))
    out = aggregator(tensor_([]).reshape(2, 0))
    assert out.shape == (0,)
