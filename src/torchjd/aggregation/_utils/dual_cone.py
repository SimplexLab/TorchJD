from typing import Literal, TypeAlias

import torch
from proxsuite.torch.qplayer import QPFunction
from torch import Tensor

SUPPORTED_SOLVER: TypeAlias = Literal["proxsuite"]


def project_weights(U: Tensor, G: Tensor, solver: SUPPORTED_SOLVER) -> Tensor:
    """
    Computes the tensor of weights corresponding to the projection of the vectors in `U` onto the
    rows of a matrix whose Gramian is provided.

    :param U: The tensor of weights corresponding to the vectors to project, of shape `[..., m]`.
    :param G: The Gramian matrix of shape `[m, m]`. It must be symmetric and positive definite.
    :param solver: The quadratic programming solver to use.
    :return: A tensor of projection weights with the same shape as `U`.
    """

    original_shape = U.shape
    m = G.shape[0]
    U_flat = U.reshape(-1, m)  # [nBatch, m]

    W = _project_weight_vector_batch(U_flat, G, solver)

    return W.reshape(original_shape)


# TODO: should merge docstrings appropriately

# def _project_weight_vector(u: np.ndarray, G: np.ndarray, solver: SUPPORTED_SOLVER) -> np.ndarray:
#     r"""
#     Computes the weights `w` of the projection of `J^T u` onto the dual cone of the rows of `J`,
#     given `G = J J^T` and `u`. In other words, this computes the `w` that satisfies
#     `\pi_J(J^T u) = J^T w`, with `\pi_J` defined in Equation 3 of [1].

#     By Proposition 1 of [1], this is equivalent to solving for `v` the following quadratic program:
#     minimize        v^T G v
#     subject to      u \preceq v

#     Reference:
#     [1] `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

#     :param u: The vector of weights `u` of shape `[m]` corresponding to the vector `J^T u` to
#         project.
#     :param G: The Gramian matrix of `J`, equal to `J J^T`, and of shape `[m, m]`. It must be
#         symmetric and positive definite.
#     :param solver: The quadratic programming solver to use.
#     """
#     ...


@torch.no_grad()
def _project_weight_vector_batch(U: Tensor, G: Tensor, _solver: SUPPORTED_SOLVER) -> Tensor:
    r"""
    Solves the batch of quadratic programs minimizing `v^T G v` subject to `u_i \preceq v_i` for
    each row `u_i` of `U`.

    :param U: The tensor of vectors `u_i` of shape `[n, m]`.
    :param G: The Gramian matrix of shape `[m, m]`. It must be symmetric and positive definite.
    :param solver: The quadratic programming solver to use.
    :return: A tensor of projection weights of shape `[n, m]`.
    """

    _, m = U.shape
    device = U.device
    dtype = U.dtype

    Q = G.cpu().to(dtype=torch.float64)
    p = torch.zeros(m, dtype=torch.float64)
    C = -torch.eye(m, dtype=torch.float64)
    lb = torch.full((m,), -1e20, dtype=torch.float64)
    ub = -U.cpu().to(dtype=torch.float64)

    solver_fn = QPFunction(structural_feasibility=True)
    zhats, _, _ = solver_fn(Q, p, torch.Tensor(), torch.Tensor(), C, lb, ub)

    return zhats.to(device=device, dtype=dtype)
