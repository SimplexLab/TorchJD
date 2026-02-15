import torch
from pytest import raises
from torch import Tensor
from torch.testing import assert_close
from utils.tensors import rand_, randperm_

from torchjd.aggregation import Aggregator
from torchjd.aggregation._utils.non_differentiable import NonDifferentiableError


def assert_expected_structure(aggregator: Aggregator, matrix: Tensor) -> None:
    """
    Tests that the vector returned by the `__call__` method of an `Aggregator` has the expected
    structure: it should return a vector whose dimension should be the number of columns of the
    input matrix, and that should only contain finite values (no `nan`, `inf` or `-inf`). Note that
    this property implies that the `__call__` method does not raise any exception.
    """

    vector = aggregator(matrix)  # Will fail if the call raises an exception
    assert vector.shape == matrix.shape[1:]
    assert vector.isfinite().all()


def assert_non_conflicting(
    aggregator: Aggregator,
    matrix: Tensor,
    atol: float = 4e-04,
    rtol: float = 4e-04,
) -> None:
    """Tests empirically that a given `Aggregator` satisfies the non-conflicting property."""

    vector = aggregator(matrix)
    output_direction = matrix @ vector
    positive_directions = output_direction[output_direction >= 0]
    assert_close(positive_directions.norm(), output_direction.norm(), atol=atol, rtol=rtol)


def assert_permutation_invariant(
    aggregator: Aggregator,
    matrix: Tensor,
    n_runs: int = 5,
    atol: float = 1e-04,
    rtol: float = 1e-04,
) -> None:
    """
    Tests empirically that for a given `Aggregator`, randomly permuting rows of the input matrix
    doesn't change the aggregation.
    """

    def permute_randomly(matrix_: Tensor) -> Tensor:
        row_permutation = randperm_(matrix_.size(dim=0))
        return matrix_[row_permutation]

    vector = aggregator(matrix)

    for _ in range(n_runs):
        permuted_matrix = permute_randomly(matrix)
        permuted_vector = aggregator(permuted_matrix)

        assert_close(vector, permuted_vector, atol=atol, rtol=rtol)


def assert_linear_under_scaling(
    aggregator: Aggregator,
    matrix: Tensor,
    n_runs: int = 5,
    atol: float = 1e-04,
    rtol: float = 1e-04,
) -> None:
    """Tests empirically that a given `Aggregator` satisfies the linear under scaling property."""

    for _ in range(n_runs):
        c1 = rand_(matrix.shape[0])
        c2 = rand_(matrix.shape[0])
        alpha = rand_([])
        beta = rand_([])

        x1 = aggregator(torch.diag(c1) @ matrix)
        x2 = aggregator(torch.diag(c2) @ matrix)
        x = aggregator(torch.diag(alpha * c1 + beta * c2) @ matrix)
        expected = alpha * x1 + beta * x2

        assert_close(x, expected, atol=atol, rtol=rtol)


def assert_strongly_stationary(
    aggregator: Aggregator,
    matrix: Tensor,
    threshold: float = 5e-03,
) -> None:
    """
    Tests empirically that a given `Aggregator` is strongly stationary.

    An aggregator `A` is strongly stationary if for any matrix `J` with `A(J)=0`, `J` is strongly
    stationary, i.e., there exists `0<w` such that `J^T w=0`. In this class, we test the
    contraposition: whenever `J` is not strongly stationary, we must have `A(J) != 0`.
    """

    vector = aggregator(matrix)
    norm = vector.norm().item()
    assert norm > threshold


def assert_non_differentiable(aggregator: Aggregator, matrix: Tensor):
    """
    Tests empirically that a given non-differentiable `Aggregator` correctly raises a
    NonDifferentiableError whenever we try to backward through it.
    """

    vector = aggregator(matrix)
    with raises(NonDifferentiableError):
        vector.backward(torch.ones_like(vector))
