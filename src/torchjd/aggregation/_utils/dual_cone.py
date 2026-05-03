from typing import Literal, TypeAlias

import numpy as np
import torch
from qpsolvers import solve_qp
from torch import Tensor

SUPPORTED_SOLVER: TypeAlias = Literal["quadprog", "qpth"]


def project_weights(U: Tensor, G: Tensor, solver: SUPPORTED_SOLVER) -> Tensor:
    """
    Computes the tensor of weights corresponding to the projection of the vectors in `U` onto the
    rows of a matrix whose Gramian is provided.

    :param U: The tensor of weights corresponding to the vectors to project, of shape `[..., m]`.
    :param G: The Gramian matrix of shape `[m, m]`. It must be symmetric and positive definite.
    :param solver: The quadratic programming solver to use. ``"quadprog"`` converts tensors to
        CPU numpy arrays and uses qpsolvers. ``"qpth"`` solves natively on the same device as
        the input tensors (e.g. CUDA) using the ``qpth`` package (optional dependency).
    :return: A tensor of projection weights with the same shape as `U`.
    """

    if solver == "qpth":
        return _project_weights_qpth(U, G)

    G_ = _to_array(G)
    U_ = _to_array(U)

    W = np.apply_along_axis(lambda u: _project_weight_vector(u, G_, solver), axis=-1, arr=U_)

    return torch.as_tensor(W, device=G.device, dtype=G.dtype)


def _project_weights_qpth(U: Tensor, G: Tensor) -> Tensor:
    r"""
    Computes the tensor of projection weights using qpth, keeping computation on the device of
    the input tensors and running without gradient tracking.

    :param U: The tensor of weights to project, of shape `[..., m]`.
    :param G: The Gramian matrix of shape `[m, m]`. It must be symmetric and positive definite.
    """
    from qpth.qp import QPFunction  # lazy import: qpth is an optional dependency

    shape = U.shape
    m = shape[-1]
    batch_size = U.numel() // m
    device = G.device
    original_dtype = G.dtype

    # Use float64 for numerical precision, matching the quadprog solver's behavior.
    U_flat = U.reshape(batch_size, m).double()
    G_double = G.double()

    # QP formulation: minimize (1/2) v^T (2G) v + 0^T v subject to -I v <= -u (i.e., u <= v)
    Q = (2.0 * G_double).unsqueeze(0).expand(batch_size, m, m).contiguous()
    p = torch.zeros(batch_size, m, device=device, dtype=torch.float64)
    G_ineq = (
        (-torch.eye(m, device=device, dtype=torch.float64))
        .unsqueeze(0)
        .expand(batch_size, m, m)
        .contiguous()
    )
    h_ineq = -U_flat
    A = torch.zeros(batch_size, 0, m, device=device, dtype=torch.float64)
    b = torch.zeros(batch_size, 0, device=device, dtype=torch.float64)

    with torch.no_grad():
        W_flat = QPFunction(verbose=False, maxIter=10, check_Q_spd=False, notImprovedLim=1)(
            Q, p, G_ineq, h_ineq, A, b
        )

    if torch.any(torch.isnan(W_flat)):
        raise ValueError("Failed to solve the quadratic programming problem.")

    return W_flat.to(original_dtype).reshape(shape)


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
