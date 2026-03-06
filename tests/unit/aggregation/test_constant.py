from contextlib import nullcontext as does_not_raise

import torch
from pytest import mark, raises
from torch import Tensor
from utils.contexts import ExceptionContext
from utils.tensors import ones_, tensor_

from torchjd.aggregation import Constant

from ._asserts import (
    assert_expected_structure,
    assert_linear_under_scaling,
    assert_strongly_stationary,
)
from ._inputs import non_strong_matrices, scaled_matrices, typical_matrices


def _make_aggregator(matrix: Tensor) -> Constant:
    n_rows = matrix.shape[0]
    weights = tensor_([1.0 / n_rows] * n_rows)
    return Constant(weights)


scaled_pairs = [(_make_aggregator(matrix), matrix) for matrix in scaled_matrices]
typical_pairs = [(_make_aggregator(matrix), matrix) for matrix in typical_matrices]
non_strong_pairs = [(_make_aggregator(matrix), matrix) for matrix in non_strong_matrices]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: Constant, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_linear_under_scaling(aggregator: Constant, matrix: Tensor) -> None:
    assert_linear_under_scaling(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], non_strong_pairs)
def test_strongly_stationary(aggregator: Constant, matrix: Tensor) -> None:
    assert_strongly_stationary(aggregator, matrix)


@mark.parametrize(
    ["weights_shape", "expectation"],
    [
        ([], raises(ValueError)),
        ([0], does_not_raise()),
        ([1], does_not_raise()),
        ([10], does_not_raise()),
        ([0, 0], raises(ValueError)),
        ([0, 1], raises(ValueError)),
        ([1, 1], raises(ValueError)),
        ([1, 1, 1], raises(ValueError)),
        ([1, 1, 1, 1], raises(ValueError)),
        ([1, 1, 1, 1, 1], raises(ValueError)),
    ],
)
def test_weights_shape_check(weights_shape: list[int], expectation: ExceptionContext) -> None:
    weights = ones_(weights_shape)
    with expectation:
        _ = Constant(weights=weights)


@mark.parametrize(
    ["weights_shape", "n_rows", "expectation"],
    [
        ([0], 0, does_not_raise()),
        ([1], 1, does_not_raise()),
        ([5], 5, does_not_raise()),
        ([0], 1, raises(ValueError)),
        ([1], 0, raises(ValueError)),
        ([4], 5, raises(ValueError)),
        ([5], 4, raises(ValueError)),
    ],
)
def test_matrix_shape_check(
    weights_shape: list[int], n_rows: int, expectation: ExceptionContext
) -> None:
    matrix = ones_([n_rows, 5])
    weights = ones_(weights_shape)
    aggregator = Constant(weights)

    with expectation:
        _ = aggregator(matrix)


def test_representations() -> None:
    A = Constant(weights=torch.tensor([1.0, 2.0], device="cpu"))
    assert repr(A) == "Constant(weights=tensor([1., 2.]))"
    assert str(A) == "Constant([1., 2.])"
