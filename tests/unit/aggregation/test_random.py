from pytest import mark
from torch import Tensor

from torchjd.aggregation import Random

from ._asserts import assert_expected_structure, assert_strongly_stationary
from ._inputs import non_strong_matrices, scaled_matrices, typical_matrices

scaled_pairs = [(Random(), matrix) for matrix in scaled_matrices]
typical_pairs = [(Random(), matrix) for matrix in typical_matrices]
non_strong_pairs = [(Random(), matrix) for matrix in non_strong_matrices]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: Random, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], non_strong_pairs)
def test_strongly_stationary(aggregator: Random, matrix: Tensor) -> None:
    assert_strongly_stationary(aggregator, matrix)


def test_representations() -> None:
    A = Random()
    assert repr(A) == "Random()"
    assert str(A) == "Random"
