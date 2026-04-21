import torch
from pytest import mark, raises
from torch import Tensor
from utils.tensors import ones_

from torchjd.aggregation import AlignedMTL, ConstantWeighting

from ._asserts import assert_expected_structure, assert_permutation_invariant
from ._inputs import scaled_matrices, typical_matrices

aggregators = [
    AlignedMTL(),
    AlignedMTL(scale_mode="median"),
    AlignedMTL(scale_mode="rmse"),
]
scaled_pairs = [(aggregator, matrix) for aggregator in aggregators for matrix in scaled_matrices]
# test_permutation_invariant seems to fail on gpu for scale_mode="median" or scale_mode="rmse".
typical_pairs = [(AlignedMTL(), matrix) for matrix in typical_matrices]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: AlignedMTL, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: AlignedMTL, matrix: Tensor) -> None:
    assert_permutation_invariant(aggregator, matrix)


def test_representations() -> None:
    A = AlignedMTL(pref_vector=None)
    assert repr(A) == "AlignedMTL(pref_vector=None, scale_mode='min')"
    assert str(A) == "AlignedMTL"

    A = AlignedMTL(pref_vector=torch.tensor([1.0, 2.0, 3.0], device="cpu"))
    assert repr(A) == "AlignedMTL(pref_vector=tensor([1., 2., 3.]), scale_mode='min')"
    assert str(A) == "AlignedMTL([1., 2., 3.])"


def test_invalid_scale_mode() -> None:
    aggregator = AlignedMTL(scale_mode="test")  # ty: ignore[invalid-argument-type]
    matrix = ones_(3, 4)
    with raises(ValueError, match=r"Invalid scale_mode=.*Expected"):
        aggregator(matrix)


def test_pref_vector_setter_updates_value() -> None:
    A = AlignedMTL()
    new_pref = torch.tensor([1.0, 2.0, 3.0])
    A.pref_vector = new_pref
    assert A.pref_vector is new_pref
    assert isinstance(A.gramian_weighting.weighting, ConstantWeighting)
    assert A.gramian_weighting.weighting.weights is new_pref


def test_scale_mode_setter_updates_value() -> None:
    A = AlignedMTL()
    A.scale_mode = "rmse"
    assert A.scale_mode == "rmse"
    assert A.gramian_weighting.scale_mode == "rmse"
