from utils.optional_deps import skip_if_deps_not_installed

from torchjd.aggregation import ConstantWeighting, UPGrad
from torchjd.linalg import QuadprogProjector

skip_if_deps_not_installed(QuadprogProjector)

import torch
from pytest import mark
from torch import Tensor
from utils.tensors import ones_

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
def test_expected_structure(aggregator: UPGrad, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_non_conflicting(aggregator: UPGrad, matrix: Tensor) -> None:
    assert_non_conflicting(aggregator, matrix, atol=4e-04, rtol=4e-04)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: UPGrad, matrix: Tensor) -> None:
    assert_permutation_invariant(aggregator, matrix, n_runs=5, atol=5e-07, rtol=5e-07)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_linear_under_scaling(aggregator: UPGrad, matrix: Tensor) -> None:
    assert_linear_under_scaling(aggregator, matrix, n_runs=5, atol=6e-02, rtol=6e-02)


@mark.parametrize(["aggregator", "matrix"], non_strong_pairs)
def test_strongly_stationary(aggregator: UPGrad, matrix: Tensor) -> None:
    assert_strongly_stationary(aggregator, matrix, threshold=5e-03)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: UPGrad, matrix: Tensor) -> None:
    assert_non_differentiable(aggregator, matrix)


def test_representations() -> None:
    A = UPGrad(pref_vector=None, projector=QuadprogProjector(norm_eps=0.001, reg_eps=0.01))
    assert (
        repr(A) == "UPGrad(pref_vector=None, projector=QuadprogProjector(norm_eps=0.001, "
        "reg_eps=0.01))"
    )
    assert str(A) == "UPGrad"

    A = UPGrad(
        pref_vector=torch.tensor([1.0, 2.0, 3.0], device="cpu"),
        projector=QuadprogProjector(norm_eps=0.001, reg_eps=0.01),
    )
    assert (
        repr(A) == "UPGrad(pref_vector=tensor([1., 2., 3.]), projector=QuadprogProjector("
        "norm_eps=0.001, reg_eps=0.01))"
    )
    assert str(A) == "UPGrad([1., 2., 3.])"


def test_pref_vector_setter_updates_value() -> None:
    A = UPGrad()
    new_pref = torch.tensor([1.0, 2.0, 3.0])
    A.pref_vector = new_pref
    assert A.pref_vector is new_pref
    assert isinstance(A.gramian_weighting.weighting, ConstantWeighting)
    assert A.gramian_weighting.weighting.weights is new_pref


def test_projector_getter_returns_default() -> None:
    A = UPGrad()
    assert isinstance(A.projector, QuadprogProjector)


def test_projector_setter_updates_value() -> None:
    A = UPGrad()
    new_projector = QuadprogProjector(norm_eps=0.001, reg_eps=0.01)
    A.projector = new_projector
    assert A.projector is new_projector
    assert A.gramian_weighting.projector is new_projector
