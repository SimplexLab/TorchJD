import numpy as np
import torch
from pytest import mark, raises
from torch.testing import assert_close
from utils.tensors import rand_, randn_

from torchjd.aggregation._utils.dual_cone import _project_weight_vector, project_weights


@mark.parametrize("shape", [(5, 7), (9, 37), (2, 14), (32, 114), (50, 100)])
def test_solution_weights(shape: tuple[int, int]) -> None:
    r"""
    Tests that `_project_weights` returns valid weights corresponding to the projection onto the
    dual cone of a matrix with the specified shape.

    Validation is performed by verifying that the solution satisfies the `KKT conditions
    <https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions>`_ for the
    quadratic program that projects vectors onto the dual cone of a matrix. Specifically, the
    solution should satisfy the equivalent set of conditions described in Lemma 4 of [1].

    Let `u` be a vector of weights and `G` a positive semi-definite matrix. Consider the quadratic
    problem of minimizing `v^T G v` subject to `u \preceq v`.

    Then `w` is a solution if and only if it satisfies the following three conditions:
    1. **Dual feasibility:** `u \preceq w`
    2. **Primal feasibility:** `0 \preceq G w`
    3. **Complementary slackness:** `u^T G w = w^T G w`

    Reference:
    [1] `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.
    """

    J = randn_(shape)
    G = J @ J.T
    u = rand_(shape[0])

    w = project_weights(u, G, "quadprog")
    dual_gap = w - u

    # Dual feasibility
    dual_gap_positive_part = dual_gap[dual_gap >= 0.0]
    assert_close(dual_gap_positive_part.norm(), dual_gap.norm(), atol=1e-05, rtol=0)

    primal_gap = G @ w

    # Primal feasibility
    primal_gap_positive_part = primal_gap[primal_gap >= 0]
    assert_close(primal_gap_positive_part.norm(), primal_gap.norm(), atol=1e-04, rtol=0)

    # Complementary slackness
    slackness = dual_gap @ primal_gap
    assert_close(slackness, torch.zeros_like(slackness), atol=3e-03, rtol=0)


@mark.parametrize("shape", [(5, 7), (9, 37), (32, 114)])
@mark.parametrize("scaling", [2 ** (-4), 2 ** (-2), 2**2, 2**4])
def test_scale_invariant(shape: tuple[int, int], scaling: float) -> None:
    """
    Tests that `_project_weights` is invariant under scaling.
    """

    J = randn_(shape)
    G = J @ J.T
    u = rand_(shape[0])

    w = project_weights(u, G, "quadprog")
    w_scaled = project_weights(u, scaling * G, "quadprog")

    assert_close(w_scaled, w)


@mark.parametrize("shape", [(5, 2, 3), (1, 3, 6, 9), (2, 1, 1, 5, 8), (3, 1)])
def test_tensorization_shape(shape: tuple[int, ...]) -> None:
    """
    Tests that applying `_project_weights` on a tensor is equivalent to applying it on the tensor
    reshaped as matrix and to reshape the result back to the original tensor's shape.
    """

    matrix = randn_([shape[-1], shape[-1]])
    U_tensor = randn_(shape)
    U_matrix = U_tensor.reshape([-1, shape[-1]])

    G = matrix @ matrix.T

    W_tensor = project_weights(U_tensor, G, "quadprog")
    W_matrix = project_weights(U_matrix, G, "quadprog")

    assert_close(W_matrix.reshape(shape), W_tensor)


def test_project_weight_vector_failure() -> None:
    """Tests that `_project_weight_vector` raises an error when the input G has too large values."""

    large_J = np.random.randn(10, 100) * 1e5
    large_G = large_J @ large_J.T
    with raises(ValueError):
        _project_weight_vector(np.ones(10), large_G, "quadprog")
