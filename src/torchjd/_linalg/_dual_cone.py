from abc import ABC, abstractmethod
from typing import Literal, TypeAlias

import numpy as np
import torch
from qpsolvers import solve_qp
from torch import Tensor

from ._matrix import PSDMatrix


class DualConeProjector(ABC):
    @abstractmethod
    def project_weights(self, U: Tensor, G: PSDMatrix) -> Tensor:
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
        return QPSolverBased("quadprog")
    return projector


class QPSolverBased(DualConeProjector):
    SUPPORTED_SOLVER: TypeAlias = Literal["quadprog"]

    def __init__(self, solver: SUPPORTED_SOLVER) -> None:
        self.solver = solver

    def project_weights(self, U: Tensor, G: Tensor) -> Tensor:

        G_ = _to_array(G)
        U_ = _to_array(U)

        W = np.apply_along_axis(lambda u: self._project_weight_vector(u, G_), axis=-1, arr=U_)

        return torch.as_tensor(W, device=G.device, dtype=G.dtype)

    def _project_weight_vector(self, u: np.ndarray, G: np.ndarray) -> np.ndarray:

        m = G.shape[0]
        w = solve_qp(G, np.zeros(m), -np.eye(m), -u, solver=self.solver)

        if w is None:  # This may happen when G has large values.
            raise ValueError("Failed to solve the quadratic programming problem.")

        return w


def _to_array(tensor: Tensor) -> np.ndarray:
    """Transforms a tensor into a numpy array with float64 dtype."""

    return tensor.cpu().detach().numpy().astype(np.float64)
