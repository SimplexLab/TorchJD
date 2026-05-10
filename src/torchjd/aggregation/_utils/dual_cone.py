import os
from typing import Literal, TypeAlias

import numpy as np
import torch
from proxsuite import proxqp
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

    n, m = U.shape
    device = U.device
    dtype = U.dtype

    Q_np = G.cpu().to(dtype=torch.float64).numpy()
    p_np = np.zeros(m, dtype=np.float64)
    C_np = -np.eye(m, dtype=np.float64)
    lb_np = np.full(m, -1e20, dtype=np.float64)
    ub_np = (-U.cpu().to(dtype=torch.float64)).numpy()

    batch_qps = proxqp.dense.BatchQP()
    default_rho = 5.0e-5

    for i in range(n):
        qp = batch_qps.init_qp_in_place(m, 0, m)
        qp.settings.primal_infeasibility_solving = False
        qp.settings.max_iter = 1000
        qp.settings.max_iter_in = 100
        qp.settings.default_rho = default_rho
        qp.settings.refactor_rho_threshold = default_rho
        qp.settings.eps_abs = 1e-9
        qp.init(H=Q_np, g=p_np, A=None, b=None, C=C_np, l=lb_np, u=ub_np[i], rho=default_rho)

    num_threads = max(1, (os.cpu_count() or 2) // 2)
    proxqp.dense.solve_in_parallel(num_threads=num_threads, qps=batch_qps)

    zhats_np = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        zhats_np[i] = batch_qps.get(i).results.x

    return torch.from_numpy(zhats_np).to(device=device, dtype=dtype)
