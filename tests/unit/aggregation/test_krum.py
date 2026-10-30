from contextlib import nullcontext as does_not_raise

from pytest import mark, raises
from torch import Tensor
from torch.optim import SGD
from torch.testing import assert_close
from utils.contexts import ExceptionContext
from utils.tensors import ones_, tensor_, zeros_

from torchjd.aggregation import Krum
from torchjd.aggregation._krum import KrumWeighting
from torchjd.autojac import backward, jac_to_grad

from ._asserts import assert_expected_structure
from ._inputs import scaled_matrices_2_plus_rows, typical_matrices_2_plus_rows

scaled_pairs = [(Krum(n_byzantine=1), matrix) for matrix in scaled_matrices_2_plus_rows]
typical_pairs = [(Krum(n_byzantine=1), matrix) for matrix in typical_matrices_2_plus_rows]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: Krum, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(
    ["n_byzantine", "expectation"],
    [
        (-5, raises(ValueError)),
        (-1, raises(ValueError)),
        (0, does_not_raise()),
        (1, does_not_raise()),
        (5, does_not_raise()),
    ],
)
def test_n_byzantine_check(n_byzantine: int, expectation: ExceptionContext) -> None:
    with expectation:
        _ = Krum(n_byzantine=n_byzantine, n_selected=1)


@mark.parametrize(
    ["n_selected", "expectation"],
    [
        (-5, raises(ValueError)),
        (-1, raises(ValueError)),
        (0, raises(ValueError)),
        (1, does_not_raise()),
        (5, does_not_raise()),
    ],
)
def test_n_selected_check(n_selected: int, expectation: ExceptionContext) -> None:
    with expectation:
        _ = Krum(n_byzantine=1, n_selected=n_selected)


@mark.parametrize(
    ["n_byzantine", "n_selected", "n_rows", "expectation"],
    [
        (1, 1, 3, raises(ValueError)),
        (1, 1, 4, does_not_raise()),
        (1, 4, 4, does_not_raise()),
        (12, 4, 14, raises(ValueError)),
        (12, 4, 15, does_not_raise()),
        (12, 15, 15, does_not_raise()),
        (12, 16, 15, raises(ValueError)),
    ],
)
def test_matrix_shape_check(
    n_byzantine: int,
    n_selected: int,
    n_rows: int,
    expectation: ExceptionContext,
) -> None:
    aggregator = Krum(n_byzantine=n_byzantine, n_selected=n_selected)
    matrix = ones_([n_rows, 5])

    with expectation:
        _ = aggregator(matrix)


def test_representations() -> None:
    A = Krum(n_byzantine=1, n_selected=2)
    assert repr(A) == "Krum(n_byzantine=1, n_selected=2)"
    assert str(A) == "Krum1-2"


def test_n_byzantine_setter_updates_value() -> None:
    A = Krum(n_byzantine=1)
    A.n_byzantine = 3
    assert A.n_byzantine == 3
    assert A.gramian_weighting.n_byzantine == 3


def test_n_selected_setter_updates_value() -> None:
    A = Krum(n_byzantine=1)
    A.n_selected = 3
    assert A.n_selected == 3
    assert A.gramian_weighting.n_selected == 3


def test_n_byzantine_setter_rejects_negative() -> None:
    A = Krum(n_byzantine=1)
    with raises(ValueError, match="n_byzantine"):
        A.n_byzantine = -1


def test_n_selected_setter_rejects_non_positive() -> None:
    A = Krum(n_byzantine=1)
    with raises(ValueError, match="n_selected"):
        A.n_selected = 0
    with raises(ValueError, match="n_selected"):
        A.n_selected = -1


def test_weighting_n_byzantine_setter_rejects_negative() -> None:
    W = KrumWeighting(n_byzantine=1)
    with raises(ValueError, match="n_byzantine"):
        W.n_byzantine = -1


def test_weighting_n_selected_setter_rejects_non_positive() -> None:
    W = KrumWeighting(n_byzantine=1)
    with raises(ValueError, match="n_selected"):
        W.n_selected = 0


def test_negative_squared_distances_are_clamped() -> None:
    x = tensor_([1.0, 1.0, 2.0, 2.1, 11.0])
    gramian = x.unsqueeze(1) * x.unsqueeze(0)
    gramian[0, 1] = gramian[0, 1] + 1e-6
    gramian[1, 0] = gramian[1, 0] + 1e-6

    weights = KrumWeighting(n_byzantine=1)(gramian)

    assert weights[:2].sum().item() == 1.0
    assert weights[2:].abs().sum().item() == 0.0


def test_squared_distance_scores_select_correct_row() -> None:
    # Summing the two nearest squared distances gives scores [30, 44, 28, 38, 56].
    # Summing their square roots instead incorrectly selects row 0.
    matrix = tensor_([[3, 0], [0, 7], [2, 5], [5, 0], [6, 7]])
    weights = KrumWeighting(n_byzantine=1)(matrix @ matrix.T)

    assert_close(weights, tensor_([0, 0, 1, 0, 0]))
    assert_close(Krum(n_byzantine=1)(matrix), tensor_([2, 5]))


@mark.parametrize(
    ["n_selected", "expected_weights"],
    [
        (1, [1, 0, 0, 0, 0]),
        (2, [0.5, 0.5, 0, 0, 0]),
        (5, [0.2, 0.2, 0.2, 0.2, 0.2]),
    ],
)
def test_multikrum_squared_distance_scores(n_selected: int, expected_weights: list[float]) -> None:
    # Scores [21, 22, 26, 33, 43] rank rows 0 and 1 first. Unsquared distances
    # rank rows 0 and 2 first, changing the Multi-Krum average.
    matrix = tensor_([[3, 0], [3, 7], [2, 0], [5, 4], [0, 7]])
    weights = KrumWeighting(n_byzantine=1, n_selected=n_selected)(matrix @ matrix.T)

    assert_close(weights, tensor_(expected_weights))
    assert_close(
        Krum(n_byzantine=1, n_selected=n_selected)(matrix),
        tensor_(expected_weights) @ matrix,
    )


@mark.parametrize("optimize_gramian_computation", [False, True])
def test_squared_distance_scores_in_optimizer_step(optimize_gramian_computation: bool) -> None:
    param = zeros_(2, requires_grad=True)
    optimizer = SGD([param], lr=0.1)
    targets = -tensor_([[3, 0], [0, 7], [2, 5], [5, 0], [6, 7]])
    losses = 0.5 * (param - targets).square().sum(dim=1)

    backward(losses, inputs=[param])
    weights = jac_to_grad(
        [param],
        Krum(n_byzantine=1),
        optimize_gramian_computation=optimize_gramian_computation,
    )
    assert_close(weights, tensor_([0, 0, 1, 0, 0]))
    assert_close(param.grad, tensor_([2, 5]))
    optimizer.step()
    assert_close(param, tensor_([-0.2, -0.5]))
