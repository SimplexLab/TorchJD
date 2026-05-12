from abc import ABC, abstractmethod

import numpy as np
import torch
from qpsolvers import solve_qp
from torch import Tensor

from ._gramian import normalize, regularize
from ._matrix import PSDMatrix


class DualConeProjector(ABC):
    @abstractmethod
    def __call__(self, U: Tensor, G: PSDMatrix) -> Tensor:
        r"""
        Computes the weights `w` of the projection of `J^T u` onto the dual cone of
        the rows of `J`, provided `G = J J^T` and `u`. In other words, this computes the `w` that
        satisfies `\pi_J(J^T u) = J^T w`, with `\pi_J` defined in Equation 3 of [1].

        By Proposition 1 of [1], this is equivalent to solving for `v` the following quadratic
        program:
        minimize        v^T G v
        subject to      u \preceq v

        Reference:
        [1] `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

        :param U: The tensor of weights corresponding to the vectors to project, of shape `[..., m]`.
        :param G: The Gramian matrix of shape `[m, m]`. It must be symmetric and positive definite.
        :return: A tensor of projection weights with the same shape as `U`.
        """


def projector_or_default(projector: DualConeProjector | None) -> DualConeProjector:
    if projector is None:
        return QuadprogProjector()
    return projector


class QuadprogProjector(DualConeProjector):
    r"""
    Solves the quadratic program defined in :meth:`DualConeProjector.__call__` using the
    ``quadprog`` QP solver.

    :param norm_eps: A small value to avoid division by zero when normalizing.
    :param reg_eps: A small value to add to the diagonal of the gramian of the matrix. Due to
        numerical errors when computing the gramian, it might not exactly be positive definite.
        This issue can make the optimization fail. Adding ``reg_eps`` to the diagonal of the gramian
        ensures that it is positive definite.
    """

    def __init__(
        self,
        *,
        norm_eps: float = 0.0001,
        reg_eps: float = 0.0001,
    ) -> None:
        self.norm_eps = norm_eps
        self.reg_eps = reg_eps

    def __repr__(self) -> str:
        return f"QuadprogProjector(norm_eps={self.norm_eps}, reg_eps={self.reg_eps})"

    def __call__(self, U: Tensor, G: PSDMatrix) -> Tensor:

        G = regularize(normalize(G, self.norm_eps), self.reg_eps)

        G_ = _to_array(G)
        U_ = _to_array(U)

        W = np.apply_along_axis(lambda u: self._project_weight_vector(u, G_), axis=-1, arr=U_)

        return torch.as_tensor(W, device=G.device, dtype=G.dtype)

    def _project_weight_vector(self, u: np.ndarray, G: np.ndarray) -> np.ndarray:

        m = G.shape[0]
        w = solve_qp(G, np.zeros(m), -np.eye(m), -u, solver="quadprog")

        if w is None:  # This may happen when G has large values.
            raise ValueError("Failed to solve the quadratic programming problem.")

        return w


def _to_array(tensor: Tensor) -> np.ndarray:
    """Transforms a tensor into a numpy array with float64 dtype."""

    return tensor.cpu().detach().numpy().astype(np.float64)
