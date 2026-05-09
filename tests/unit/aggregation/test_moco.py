from pytest import mark, raises
from torch import Tensor
from torch.testing import assert_close
from utils.tensors import ones_, randn_, tensor_

from torchjd.aggregation import MoCo, MoCoWeighting

from ._asserts import assert_expected_structure, assert_non_differentiable
from ._inputs import scaled_matrices, typical_matrices

scaled_pairs = [(MoCo(), matrix) for matrix in scaled_matrices]
typical_pairs = [(MoCo(), matrix) for matrix in typical_matrices]
requires_grad_pairs = [(MoCo(), ones_(3, 5, requires_grad=True))]
PARAMETER_VALUES = [
    ("beta", 0.25),
    ("beta_sigma", 0.75),
    ("gamma", 0.2),
    ("gamma_sigma", 0.6),
    ("rho", 0.1),
]


def test_representations() -> None:
    A = MoCo(beta=0.25, beta_sigma=0.75, gamma=0.2, gamma_sigma=0.6, rho=0.1)
    assert repr(A) == "MoCo(beta=0.25, beta_sigma=0.75, gamma=0.2, gamma_sigma=0.6, rho=0.1)"
    assert str(A) == "MoCo"


def test_zero_rows_returns_zero_vector() -> None:
    out = MoCo()(tensor_([]).reshape(0, 3))
    assert_close(out, tensor_([0.0, 0.0, 0.0]))


def test_zero_columns_returns_zero_vector() -> None:
    out = MoCo()(tensor_([]).reshape(2, 0))
    assert out.shape == (0,)


@mark.parametrize("matrix", typical_matrices)
def test_reset_restores_first_step_behavior(matrix: Tensor) -> None:
    A = MoCo()
    first = A(matrix)
    A(matrix)
    A.reset()
    assert_close(first, A(matrix))


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: MoCo, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: MoCo, matrix: Tensor) -> None:
    assert_non_differentiable(aggregator, matrix)


def test_weighting_reset_restores_first_step_behavior() -> None:
    J = randn_((3, 8))
    W = MoCoWeighting()
    first = W(J)
    W(J)
    W.reset()
    assert_close(first, W(J))


def test_weighting_reset_clears_state() -> None:
    J = randn_((3, 8))
    W = MoCoWeighting()
    W(J)

    W.reset()

    assert W.step == 0
    with raises(RuntimeError, match="moving gradient estimate"):
        _ = W.y


def test_aggregator_reset_clears_weighting_state() -> None:
    J = randn_((3, 8))
    A = MoCo()
    A(J)

    A.reset()

    assert A.weighting.step == 0
    with raises(RuntimeError, match="moving gradient estimate"):
        _ = A.weighting.y


def test_y_getter_returns_current_moving_gradient_estimate() -> None:
    J = randn_((3, 8))
    W = MoCoWeighting()

    W(J)

    assert_close(W.y, 0.5 * J)


def test_weighting_matches_aggregator_state_update() -> None:
    J = randn_((3, 8))

    A = MoCo(beta=0.3, beta_sigma=0.4, gamma=0.2, gamma_sigma=0.6, rho=0.1)
    expected = A(J)

    W = MoCoWeighting(beta=0.3, beta_sigma=0.4, gamma=0.2, gamma_sigma=0.6, rho=0.1)
    weights = W(J)
    result = weights @ W.y

    assert_close(result, expected)


@mark.parametrize(["attribute", "value"], PARAMETER_VALUES)
def test_getters_return_constructor_values(attribute: str, value: float) -> None:
    A = MoCo(**{attribute: value})

    assert getattr(A, attribute) == value
    assert getattr(A.weighting, attribute) == value


@mark.parametrize(["attribute", "value"], PARAMETER_VALUES)
def test_weighting_getters_return_constructor_values(attribute: str, value: float) -> None:
    W = MoCoWeighting(**{attribute: value})

    assert getattr(W, attribute) == value


def test_aggregator_setters_update_values() -> None:
    A = MoCo()
    A.beta = 0.25
    A.beta_sigma = 0.75
    A.gamma = 0.2
    A.gamma_sigma = 0.6
    A.rho = 0.1
    assert A.beta == 0.25
    assert A.beta_sigma == 0.75
    assert A.gamma == 0.2
    assert A.gamma_sigma == 0.6
    assert A.rho == 0.1
    assert A.weighting.beta == 0.25
    assert A.weighting.beta_sigma == 0.75
    assert A.weighting.gamma == 0.2
    assert A.weighting.gamma_sigma == 0.6
    assert A.weighting.rho == 0.1


@mark.parametrize(["attribute", "value"], PARAMETER_VALUES)
def test_aggregator_setter_updates_matching_weighting_value(attribute: str, value: float) -> None:
    A = MoCo()

    setattr(A, attribute, value)

    assert getattr(A, attribute) == value
    assert getattr(A.weighting, attribute) == value


@mark.parametrize(["attribute", "value"], PARAMETER_VALUES)
def test_weighting_setter_updates_value(attribute: str, value: float) -> None:
    W = MoCoWeighting()

    setattr(W, attribute, value)

    assert getattr(W, attribute) == value


@mark.parametrize("attribute", ["beta", "beta_sigma", "gamma", "gamma_sigma", "rho"])
def test_aggregator_setters_reject_negative(attribute: str) -> None:
    A = MoCo()
    with raises(ValueError, match=attribute):
        setattr(A, attribute, -1e-9)


@mark.parametrize("attribute", ["beta", "beta_sigma", "gamma", "gamma_sigma", "rho"])
def test_weighting_setters_reject_negative(attribute: str) -> None:
    W = MoCoWeighting()
    with raises(ValueError, match=attribute):
        setattr(W, attribute, -1e-9)
