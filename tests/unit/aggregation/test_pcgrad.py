from pytest import mark
from torch import Tensor
from torch.testing import assert_close
from utils.tensors import ones_, randn_

from torchjd._linalg import compute_gramian
from torchjd.aggregation import PCGrad
from torchjd.aggregation._pcgrad import PCGradWeighting
from torchjd.aggregation._upgrad import UPGradWeighting

from ._asserts import assert_expected_structure, assert_non_differentiable
from ._inputs import scaled_matrices, typical_matrices

scaled_pairs = [(PCGrad(), matrix) for matrix in scaled_matrices]
typical_pairs = [(PCGrad(), matrix) for matrix in typical_matrices]
requires_grad_pairs = [(PCGrad(), ones_(3, 5, requires_grad=True))]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: PCGrad, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: PCGrad, matrix: Tensor) -> None:
    assert_non_differentiable(aggregator, matrix)


@mark.parametrize(
    "shape",
    [
        (2, 5),
        (2, 7),
        (2, 9),
        (2, 15),
        (2, 27),
        (2, 68),
        (2, 102),
        (2, 57),
        (2, 1200),
        (2, 11100),
    ],
)
def test_equivalence_upgrad_sum_two_rows(shape: tuple[int, int]) -> None:
    """
    Tests that UPGradWeighting of a SumWeighting is equivalent to PCGradWeighting for matrices of 2
    rows.
    """

    matrix = randn_(shape)
    gramian = compute_gramian(matrix)

    pc_grad_weighting = PCGradWeighting()
    upgrad_sum_weighting = UPGradWeighting(
        ones_((2,)),
        norm_eps=0.0,
    )

    result = pc_grad_weighting(gramian)
    expected = upgrad_sum_weighting(gramian)

    assert_close(result, expected, atol=4e-04, rtol=0.0)


def test_representations() -> None:
    A = PCGrad()
    assert repr(A) == "PCGrad()"
    assert str(A) == "PCGrad"
