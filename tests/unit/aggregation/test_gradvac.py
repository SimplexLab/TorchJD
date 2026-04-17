import torch
from pytest import mark, raises
from torch import Tensor
from torch.testing import assert_close
from utils.tensors import ones_, randn_, tensor_

from torchjd.aggregation import GradVac, GradVacWeighting

from ._asserts import assert_expected_structure, assert_non_differentiable, assert_stateful
from ._inputs import scaled_matrices, typical_matrices, typical_matrices_2_plus_rows

scaled_pairs = [(GradVac(), m) for m in scaled_matrices]
typical_pairs = [(GradVac(), m) for m in typical_matrices]
requires_grad_pairs = [(GradVac(), ones_(3, 5, requires_grad=True))]


def test_representations() -> None:
    A = GradVac()
    assert repr(A) == "GradVac(beta=0.5, eps=1e-08)"
    assert str(A) == "GradVac"


def test_beta_out_of_range() -> None:
    with raises(ValueError, match="beta"):
        GradVac(beta=-0.1)
    with raises(ValueError, match="beta"):
        GradVac(beta=1.1)


def test_beta_setter_out_of_range() -> None:
    A = GradVac()
    with raises(ValueError, match="beta"):
        A.beta = -0.1
    with raises(ValueError, match="beta"):
        A.beta = 1.1


def test_beta_setter_updates_value() -> None:
    A = GradVac()
    A.beta = 0.25
    assert A.beta == 0.25


def test_eps_rejects_negative() -> None:
    with raises(ValueError, match="eps"):
        GradVac(eps=-1e-9)


def test_eps_setter_rejects_negative() -> None:
    A = GradVac()
    with raises(ValueError, match="eps"):
        A.eps = -1e-9


def test_eps_can_be_changed_between_steps() -> None:
    J = tensor_([[1.0, 0.0], [0.0, 1.0]])
    A = GradVac()
    A.eps = 1e-6
    assert A(J).isfinite().all()
    A.reset()
    A.eps = 1e-10
    assert A(J).isfinite().all()


def test_zero_rows_returns_zero_vector() -> None:
    out = GradVac()(tensor_([]).reshape(0, 3))
    assert_close(out, tensor_([0.0, 0.0, 0.0]))


def test_zero_columns_returns_zero_vector() -> None:
    out = GradVac()(tensor_([]).reshape(2, 0))
    assert out.shape == (0,)


def test_reproducible_with_manual_seed() -> None:
    J = randn_((3, 8))
    torch.manual_seed(12345)
    A1 = GradVac(beta=0.3)
    out1 = A1(J)
    torch.manual_seed(12345)
    A2 = GradVac(beta=0.3)
    out2 = A2(J)
    assert_close(out1, out2)


@mark.parametrize("matrix", typical_matrices_2_plus_rows)
def test_reset_restores_first_step_behavior(matrix: Tensor) -> None:
    torch.manual_seed(7)
    A = GradVac(beta=0.5)
    first = A(matrix)
    A(matrix)
    A.reset()
    torch.manual_seed(7)
    assert_close(first, A(matrix))


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: GradVac, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: GradVac, matrix: Tensor) -> None:
    assert_non_differentiable(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_stateful(aggregator: GradVac, matrix: Tensor) -> None:
    assert_stateful(aggregator, matrix)


def test_weighting_beta_out_of_range() -> None:
    with raises(ValueError, match="beta"):
        GradVacWeighting(beta=-0.1)
    with raises(ValueError, match="beta"):
        GradVacWeighting(beta=1.1)


def test_weighting_eps_rejects_negative() -> None:
    with raises(ValueError, match="eps"):
        GradVacWeighting(eps=-1e-9)


def test_weighting_reset_restores_first_step_behavior() -> None:
    J = randn_((3, 8))
    G = J @ J.T
    torch.manual_seed(7)
    w = GradVacWeighting(beta=0.5)
    first = w(G)
    w(G)
    w.reset()
    torch.manual_seed(7)
    assert_close(first, w(G))


def test_aggregator_and_weighting_agree() -> None:
    """GradVac()(J) == GradVacWeighting()(J @ J.T) @ J for any matrix J."""

    J = randn_((3, 8))
    G = J @ J.T

    torch.manual_seed(42)
    A = GradVac(beta=0.3)
    expected = A(J)

    torch.manual_seed(42)
    W = GradVacWeighting(beta=0.3)
    weights = W(G)
    result = weights @ J

    assert_close(result, expected, rtol=1e-4, atol=1e-4)
