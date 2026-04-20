import torch
from pytest import mark, raises
from torch import Tensor
from utils.tensors import ones_

from torchjd.aggregation import ConstantWeighting, DualProj
from torchjd.aggregation._dualproj import DualProjWeighting

from ._asserts import (
    assert_expected_structure,
    assert_non_conflicting,
    assert_non_differentiable,
    assert_permutation_invariant,
    assert_strongly_stationary,
)
from ._inputs import non_strong_matrices, scaled_matrices, typical_matrices

scaled_pairs = [(DualProj(), matrix) for matrix in scaled_matrices]
typical_pairs = [(DualProj(), matrix) for matrix in typical_matrices]
non_strong_pairs = [(DualProj(), matrix) for matrix in non_strong_matrices]
requires_grad_pairs = [(DualProj(), ones_(3, 5, requires_grad=True))]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: DualProj, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_non_conflicting(aggregator: DualProj, matrix: Tensor) -> None:
    assert_non_conflicting(aggregator, matrix, atol=1e-04, rtol=1e-04)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: DualProj, matrix: Tensor) -> None:
    assert_permutation_invariant(aggregator, matrix, n_runs=5, atol=2e-07, rtol=2e-07)


@mark.parametrize(["aggregator", "matrix"], non_strong_pairs)
def test_strongly_stationary(aggregator: DualProj, matrix: Tensor) -> None:
    assert_strongly_stationary(aggregator, matrix, threshold=3e-03)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: DualProj, matrix: Tensor) -> None:
    assert_non_differentiable(aggregator, matrix)


def test_representations() -> None:
    A = DualProj(pref_vector=None, norm_eps=0.0001, reg_eps=0.0001, solver="quadprog")
    assert (
        repr(A) == "DualProj(pref_vector=None, norm_eps=0.0001, reg_eps=0.0001, solver='quadprog')"
    )
    assert str(A) == "DualProj"

    A = DualProj(
        pref_vector=torch.tensor([1.0, 2.0, 3.0], device="cpu"),
        norm_eps=0.0001,
        reg_eps=0.0001,
        solver="quadprog",
    )
    assert (
        repr(A) == "DualProj(pref_vector=tensor([1., 2., 3.]), norm_eps=0.0001, reg_eps=0.0001, "
        "solver='quadprog')"
    )
    assert str(A) == "DualProj([1., 2., 3.])"


def test_pref_vector_setter_updates_value() -> None:
    A = DualProj()
    new_pref = torch.tensor([1.0, 2.0, 3.0])
    A.pref_vector = new_pref
    assert A.pref_vector is new_pref
    assert isinstance(A.gramian_weighting.weighting, ConstantWeighting)
    assert A.gramian_weighting.weighting.weights is new_pref


def test_norm_eps_setter_updates_value() -> None:
    A = DualProj()
    A.norm_eps = 0.25
    assert A.norm_eps == 0.25


def test_reg_eps_setter_updates_value() -> None:
    A = DualProj()
    A.reg_eps = 0.25
    assert A.reg_eps == 0.25


def test_norm_eps_setter_rejects_negative() -> None:
    A = DualProj()
    with raises(ValueError, match="norm_eps"):
        A.norm_eps = -1e-9


def test_reg_eps_setter_rejects_negative() -> None:
    A = DualProj()
    with raises(ValueError, match="reg_eps"):
        A.reg_eps = -1e-9


def test_weighting_norm_eps_setter_rejects_negative() -> None:
    W = DualProjWeighting()
    with raises(ValueError, match="norm_eps"):
        W.norm_eps = -1e-9


def test_weighting_reg_eps_setter_rejects_negative() -> None:
    W = DualProjWeighting()
    with raises(ValueError, match="reg_eps"):
        W.reg_eps = -1e-9
