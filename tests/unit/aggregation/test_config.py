import torch
from pytest import mark
from torch import Tensor
from utils.tensors import ones_, tensor_

from torchjd.aggregation import ConFIG, ConstantWeighting

from ._asserts import (
    assert_expected_structure,
    assert_linear_under_scaling,
    assert_non_differentiable,
    assert_permutation_invariant,
)
from ._inputs import non_strong_matrices, scaled_matrices, typical_matrices

scaled_pairs = [(ConFIG(), matrix) for matrix in scaled_matrices]
typical_pairs = [(ConFIG(), matrix) for matrix in typical_matrices]
non_strong_pairs = [(ConFIG(), matrix) for matrix in non_strong_matrices]
requires_grad_pairs = [(ConFIG(), ones_(3, 5, requires_grad=True))]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: ConFIG, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: ConFIG, matrix: Tensor) -> None:
    assert_permutation_invariant(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_linear_under_scaling(aggregator: ConFIG, matrix: Tensor) -> None:
    assert_linear_under_scaling(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: ConFIG, matrix: Tensor) -> None:
    assert_non_differentiable(aggregator, matrix)


def test_representations() -> None:
    A = ConFIG()
    assert repr(A) == "ConFIG(pref_vector=None)"
    assert str(A) == "ConFIG"

    A = ConFIG(pref_vector=torch.tensor([1.0, 2.0, 3.0], device="cpu"))
    assert repr(A) == "ConFIG(pref_vector=tensor([1., 2., 3.]))"
    assert str(A) == "ConFIG([1., 2., 3.])"


def test_pref_vector_setter_updates_value() -> None:
    A = ConFIG()
    new_pref = torch.tensor([1.0, 2.0, 3.0])
    A.pref_vector = new_pref
    assert A.pref_vector is new_pref
    assert isinstance(A.weighting, ConstantWeighting)
    assert A.weighting.weights is new_pref


def test_to_moves_pref_vector() -> None:
    A = ConFIG(pref_vector=tensor_([1.0, 2.0], dtype=torch.float32)).to(torch.float64)
    weighting = A.weighting
    assert isinstance(weighting, ConstantWeighting)

    assert A.pref_vector is not None
    assert A.pref_vector.dtype == torch.float64
    assert weighting.weights.dtype == torch.float64
    buffers = list(A.buffers())
    assert any(buffer is A.pref_vector for buffer in buffers)
    assert any(buffer is weighting.weights for buffer in buffers)
    assert len(A.state_dict()) == 0
