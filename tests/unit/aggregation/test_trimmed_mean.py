from contextlib import nullcontext as does_not_raise

from pytest import mark, raises
from torch import Tensor
from utils.contexts import ExceptionContext
from utils.tensors import ones_

from torchjd.aggregation import TrimmedMean

from ._asserts import assert_expected_structure, assert_permutation_invariant
from ._inputs import scaled_matrices_2_plus_rows, typical_matrices_2_plus_rows

scaled_pairs = [(TrimmedMean(trim_number=1), matrix) for matrix in scaled_matrices_2_plus_rows]
typical_pairs = [(TrimmedMean(trim_number=1), matrix) for matrix in typical_matrices_2_plus_rows]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: TrimmedMean, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: TrimmedMean, matrix: Tensor) -> None:
    assert_permutation_invariant(aggregator, matrix)


@mark.parametrize(
    ["trim_number", "expectation"],
    [
        (-5, raises(ValueError)),
        (-1, raises(ValueError)),
        (0, does_not_raise()),
        (1, does_not_raise()),
        (5, does_not_raise()),
    ],
)
def test_trim_number_check(trim_number: int, expectation: ExceptionContext) -> None:
    with expectation:
        _ = TrimmedMean(trim_number=trim_number)


@mark.parametrize(
    ["n_rows", "trim_number", "expectation"],
    [
        (1, 0, does_not_raise()),
        (1, 1, raises(ValueError)),
        (10, 0, does_not_raise()),
        (10, 4, does_not_raise()),
        (10, 5, raises(ValueError)),
    ],
)
def test_matrix_shape_check(n_rows: int, trim_number: int, expectation: ExceptionContext) -> None:
    matrix = ones_([n_rows, 5])
    aggregator = TrimmedMean(trim_number=trim_number)

    with expectation:
        _ = aggregator(matrix)


def test_representations() -> None:
    aggregator = TrimmedMean(trim_number=2)
    assert repr(aggregator) == "TrimmedMean(trim_number=2)"
    assert str(aggregator) == "TM2"
