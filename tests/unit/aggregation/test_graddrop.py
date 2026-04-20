import re
from contextlib import nullcontext as does_not_raise

import torch
from pytest import mark, raises
from torch import Tensor
from utils.contexts import ExceptionContext
from utils.tensors import ones_

from torchjd.aggregation import GradDrop

from ._asserts import assert_expected_structure, assert_non_differentiable
from ._inputs import scaled_matrices, typical_matrices

scaled_pairs = [(GradDrop(), matrix) for matrix in scaled_matrices]
typical_pairs = [(GradDrop(), matrix) for matrix in typical_matrices]
requires_grad_pairs = [(GradDrop(), ones_(3, 5, requires_grad=True))]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: GradDrop, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: GradDrop, matrix: Tensor) -> None:
    assert_non_differentiable(aggregator, matrix)


@mark.parametrize(
    ["leak_shape", "expectation"],
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
def test_leak_shape_check(leak_shape: list[int], expectation: ExceptionContext) -> None:
    leak = ones_(leak_shape)
    with expectation:
        _ = GradDrop(leak=leak)


@mark.parametrize(
    ["leak_shape", "n_rows", "expectation"],
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
    leak_shape: list[int], n_rows: int, expectation: ExceptionContext
) -> None:
    matrix = ones_([n_rows, 5])
    leak = ones_(leak_shape)
    aggregator = GradDrop(leak=leak)

    with expectation:
        _ = aggregator(matrix)


def test_representations() -> None:
    A = GradDrop(leak=torch.tensor([0.0, 1.0], device="cpu"))
    assert re.match(
        r"GradDrop\(f=<function _identity at 0x[0-9a-fA-F]+>, leak=tensor\(\[0\., 1\.\]\)\)",
        repr(A),
    )

    assert str(A) == "GradDrop([0., 1.])"

    A = GradDrop()
    assert re.match(r"GradDrop\(f=<function _identity at 0x[0-9a-fA-F]+>, leak=None\)", repr(A))
    assert str(A) == "GradDrop"


def test_leak_setter_updates_value() -> None:
    A = GradDrop()
    new_leak = torch.tensor([0.0, 0.5, 1.0])
    A.leak = new_leak
    assert A.leak is new_leak


def test_leak_setter_accepts_none() -> None:
    A = GradDrop(leak=torch.tensor([0.0, 1.0]))
    A.leak = None
    assert A.leak is None


def test_leak_setter_rejects_non_1d() -> None:
    A = GradDrop()
    with raises(ValueError, match="leak"):
        A.leak = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
