from pytest import mark, raises
from torch import Tensor
from torch.testing import assert_close
from utils.tensors import randn_, tensor_

from torchjd.aggregation import MeanWeighting, UPGradWeighting
from torchjd.aggregation._aggregator_bases import (
    GramianWeightedAggregator,
    WeightedAggregator,
)
from torchjd.aggregation._cr_mogm import CRMOGMWeighting

from ._asserts import assert_expected_structure
from ._inputs import scaled_matrices, typical_matrices

# UPGradWeighting uses a QP solver that can fail on the extreme scales (0.0, 1e15) found in
# scaled_matrices, so the gramian-path structural test only uses typical_matrices.
matrix_pairs = [
    (WeightedAggregator(CRMOGMWeighting(MeanWeighting())), m)
    for m in typical_matrices + scaled_matrices
]
gramian_pairs = [
    (GramianWeightedAggregator(CRMOGMWeighting(UPGradWeighting())), m) for m in typical_matrices
]


def test_representations() -> None:
    W = CRMOGMWeighting(MeanWeighting(), alpha=0.9)
    expected = "CRMOGMWeighting(weighting=MeanWeighting(), alpha=0.9)"
    # Weighting does not define __str__, so it falls back to __repr__.
    assert repr(W) == expected
    assert str(W) == expected


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
    Use ``UPGradWeighting`` so the weights actually depend on the input — with
    ``MeanWeighting`` the EMA would be a fixed point at the uniform weights and the test would
    be trivial.
    """

    J = randn_((3, 8))
    G = J @ J.T
    W = CRMOGMWeighting(UPGradWeighting(), alpha=0.5)
    first = W(G)
    W(G)
    W.reset()
    assert_close(first, W(G))


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
    bare = UPGradWeighting()
    smoothed = CRMOGMWeighting(UPGradWeighting(), alpha=0.0)

    expected = bare(G)
    assert_close(smoothed(G), expected)
    assert_close(smoothed(G), expected)


def test_alpha_one_freezes_weights() -> None:
    """
    With ``alpha=1`` the fresh weights are multiplied by zero, so the smoothed weights stay at
    their initial uniform value forever. Note: the equality with uniform weights is a
    consequence of the uniform initialisation, not a general property of CR-MOGM.
    """

    J = randn_((3, 8))
    m = J.shape[0]
    W = CRMOGMWeighting(UPGradWeighting(), alpha=1.0)
    uniform = tensor_([1.0 / m] * m)

    assert_close(W(J @ J.T), uniform)
    assert_close(W(J @ J.T), uniform)


def test_ema_is_applied() -> None:
    """Run two steps with ``alpha=0.9`` and check the EMA recurrence by hand."""

    alpha = 0.9
    J1 = randn_((3, 8))
    J2 = randn_((3, 8))
    G1 = J1 @ J1.T
    G2 = J2 @ J2.T
    m = J1.shape[0]

    bare = UPGradWeighting()
    smoothed = CRMOGMWeighting(UPGradWeighting(), alpha=alpha)

    lambda_hat_1 = bare(G1)
    lambda_hat_2 = bare(G2)
    uniform = tensor_([1.0 / m] * m)

    expected_1 = alpha * uniform + (1.0 - alpha) * lambda_hat_1
    expected_2 = alpha * expected_1 + (1.0 - alpha) * lambda_hat_2

    assert_close(smoothed(G1), expected_1)
    assert_close(smoothed(G2), expected_2)


def test_zero_columns() -> None:
    """
    A ``(2, 0)`` matrix has no columns to combine, so the aggregation must be empty. Zero-row
    inputs are intentionally not tested: ``MeanWeighting`` does ``1/m`` in Python and would
    raise ``ZeroDivisionError`` at ``m=0``, which is the wrapped weighting's responsibility.
    """

    aggregator = WeightedAggregator(CRMOGMWeighting(MeanWeighting()))
    out = aggregator(tensor_([]).reshape(2, 0))
    assert out.shape == (0,)
