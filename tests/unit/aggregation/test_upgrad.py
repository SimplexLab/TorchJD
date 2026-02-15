import torch
from pytest import mark
from torch import Tensor
from utils.tensors import ones_

from torchjd.aggregation import UPGrad

from ._asserts import (
    assert_expected_structure,
    assert_linear_under_scaling,
    assert_non_conflicting,
    assert_non_differentiable,
    assert_permutation_invariant,
    assert_strongly_stationary,
)
from ._inputs import non_strong_matrices, scaled_matrices, typical_matrices

scaled_pairs = [(UPGrad(), matrix) for matrix in scaled_matrices]
typical_pairs = [(UPGrad(), matrix) for matrix in typical_matrices]
non_strong_pairs = [(UPGrad(), matrix) for matrix in non_strong_matrices]
requires_grad_pairs = [(UPGrad(), ones_(3, 5, requires_grad=True))]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: UPGrad, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_non_conflicting(aggregator: UPGrad, matrix: Tensor):
    assert_non_conflicting(aggregator, matrix, atol=4e-04, rtol=4e-04)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: UPGrad, matrix: Tensor):
    assert_permutation_invariant(aggregator, matrix, n_runs=5, atol=5e-07, rtol=5e-07)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_linear_under_scaling(aggregator: UPGrad, matrix: Tensor):
    assert_linear_under_scaling(aggregator, matrix, n_runs=5, atol=6e-02, rtol=6e-02)


@mark.parametrize(["aggregator", "matrix"], non_strong_pairs)
def test_strongly_stationary(aggregator: UPGrad, matrix: Tensor):
    assert_strongly_stationary(aggregator, matrix, threshold=5e-03)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: UPGrad, matrix: Tensor):
    assert_non_differentiable(aggregator, matrix)


def test_representations():
    A = UPGrad(pref_vector=None, norm_eps=0.0001, reg_eps=0.0001, solver="quadprog")
    assert repr(A) == "UPGrad(pref_vector=None, norm_eps=0.0001, reg_eps=0.0001, solver='quadprog')"
    assert str(A) == "UPGrad"

    A = UPGrad(
        pref_vector=torch.tensor([1.0, 2.0, 3.0], device="cpu"),
        norm_eps=0.0001,
        reg_eps=0.0001,
        solver="quadprog",
    )
    assert (
        repr(A) == "UPGrad(pref_vector=tensor([1., 2., 3.]), norm_eps=0.0001, reg_eps=0.0001, "
        "solver='quadprog')"
    )
    assert str(A) == "UPGrad([1., 2., 3.])"
