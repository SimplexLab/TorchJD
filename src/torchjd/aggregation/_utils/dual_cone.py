from typing import Literal, TypeAlias

import numpy as np
import torch
from qpsolvers import solve_qp
from torch import Tensor

SUPPORTED_SOLVER: TypeAlias = Literal["quadprog"]


def project_weights(U: Tensor, G: Tensor, solver: SUPPORTED_SOLVER) -> Tensor:
    """
    Computes the tensor of weights corresponding to the projection of the vectors in `U` onto the
    rows of a matrix whose Gramian is provided.

    :param U: The tensor of weights corresponding to the vectors to project, of shape `[..., m]`.
    :param G: The Gramian matrix of shape `[m, m]`. It must be symmetric and positive definite.
    :param solver: The quadratic programming solver to use.
    :return: A tensor of projection weights with the same shape as `U`.
    """

    G_ = _to_array(G)
    U_ = _to_array(U)

    W = np.apply_along_axis(lambda u: _project_weight_vector(u, G_, solver), axis=-1, arr=U_)

    return torch.as_tensor(W, device=G.device, dtype=G.dtype)


def _project_weight_vector(u: np.ndarray, G: np.ndarray, solver: SUPPORTED_SOLVER) -> np.ndarray:
    r"""
    Computes the weights `w` of the projection of `J^T u` onto the dual cone of the rows of `J`,
    given `G = J J^T` and `u`. In other words, this computes the `w` that satisfies
    `\pi_J(J^T u) = J^T w`, with `\pi_J` defined in Equation 3 of [1].

    By Proposition 1 of [1], this is equivalent to solving for `v` the following quadratic program:
    minimize        v^T G v
    subject to      u \preceq v

    Reference:
    [1] `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

    :param u: The vector of weights `u` of shape `[m]` corresponding to the vector `J^T u` to
        project.
    :param G: The Gramian matrix of `J`, equal to `J J^T`, and of shape `[m, m]`. It must be
        symmetric and positive definite.
    :param solver: The quadratic programming solver to use.
    """

    m = G.shape[0]
    w = solve_qp(G, np.zeros(m), -np.eye(m), -u, solver=solver)

    if w is None:  # This may happen when G has large values.
        raise ValueError("Failed to solve the quadratic programming problem.")

    return w


def _to_array(tensor: Tensor) -> np.ndarray:
    """Transforms a tensor into a numpy array with float64 dtype."""

    return tensor.cpu().detach().numpy().astype(np.float64)
