from contextlib import nullcontext as does_not_raise

from pytest import mark, raises
from torch import Tensor
from utils.contexts import ExceptionContext
from utils.tensors import ones_

try:
    from torchjd.aggregation import CAGrad
    from torchjd.aggregation._cagrad import CAGradWeighting
except ImportError:
    import pytest

    pytest.skip("CAGrad dependencies not installed", allow_module_level=True)

from ._asserts import assert_expected_structure, assert_non_conflicting, assert_non_differentiable
from ._inputs import scaled_matrices, typical_matrices

scaled_pairs = [(CAGrad(c=0.5), matrix) for matrix in scaled_matrices]
typical_pairs = [(CAGrad(c=0.5), matrix) for matrix in typical_matrices]
requires_grad_pairs = [(CAGrad(c=0.5), ones_(3, 5, requires_grad=True))]
non_conflicting_pairs_1 = [(CAGrad(c=1.0), matrix) for matrix in typical_matrices]
non_conflicting_pairs_2 = [(CAGrad(c=2.0), matrix) for matrix in typical_matrices]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: CAGrad, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: CAGrad, matrix: Tensor) -> None:
    assert_non_differentiable(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], non_conflicting_pairs_1 + non_conflicting_pairs_2)
def test_non_conflicting(aggregator: CAGrad, matrix: Tensor) -> None:
    """Tests that CAGrad is non-conflicting when c >= 1 (it should not hold when c < 1)."""
    assert_non_conflicting(aggregator, matrix)


@mark.parametrize(
    ["c", "expectation"],
    [
        (-5.0, raises(ValueError)),
        (-1.0, raises(ValueError)),
        (0.0, does_not_raise()),
        (1.0, does_not_raise()),
        (50.0, does_not_raise()),
    ],
)
def test_c_check(c: float, expectation: ExceptionContext) -> None:
    with expectation:
        _ = CAGrad(c=c)


def test_representations() -> None:
    A = CAGrad(c=0.5, norm_eps=0.0001)
    assert repr(A) == "CAGrad(c=0.5, norm_eps=0.0001)"
    assert str(A) == "CAGrad0.5"


def test_c_setter_updates_value() -> None:
    A = CAGrad(c=0.5)
    A.c = 1.25
    assert A.c == 1.25
    assert A.gramian_weighting.c == 1.25


def test_norm_eps_setter_updates_value() -> None:
    A = CAGrad(c=0.5)
    A.norm_eps = 0.25
    assert A.norm_eps == 0.25
    assert A.gramian_weighting.norm_eps == 0.25


def test_c_setter_rejects_negative() -> None:
    A = CAGrad(c=0.5)
    with raises(ValueError, match="c"):
        A.c = -1e-9


def test_norm_eps_setter_rejects_negative() -> None:
    A = CAGrad(c=0.5)
    with raises(ValueError, match="norm_eps"):
        A.norm_eps = -1e-9


def test_weighting_c_setter_rejects_negative() -> None:
    W = CAGradWeighting(c=0.5)
    with raises(ValueError, match="c"):
        W.c = -1e-9


def test_weighting_norm_eps_setter_rejects_negative() -> None:
    W = CAGradWeighting(c=0.5)
    with raises(ValueError, match="norm_eps"):
        W.norm_eps = -1e-9
