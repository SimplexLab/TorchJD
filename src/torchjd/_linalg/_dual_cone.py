import contextlib
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from torchjd._mixins import _WithOptionalDeps

with contextlib.suppress(ImportError):
    import numpy as np
    from qpsolvers import solve_qp

from ._gramian import normalize, regularize
from ._matrix import PSDMatrix


class DualConeProjector(ABC):
    """
    Abstract class whose instances are responsible for projecting vectors onto the dual cone of the
    rows of a matrix, or rather the dual form of this problem.
    """

    @abstractmethod
    def __call__(self, U: Tensor, G: PSDMatrix) -> Tensor:
        r"""
        Computes for each vector :math:`u` in the provided tensor ``U``
        the weights :math:`w` of the projection of :math:`J^\top u` onto the dual cone of
        the rows of :math:`J`, provided :math:`G = J J^\top` and :math:`u`. In other words, this
        computes the :math:`w` that satisfies :math:`\pi_J(J^\top u) = J^\top w`, with
        :math:`\pi_J` defined in Equation 3 of [1].

        By Proposition 1 of [1], this is equivalent to solving for :math:`v` the following
        quadratic program:

        .. math::

            \min_{v} \quad & v^\top G v \\
            \text{subject to} \quad & u \preceq v

        Reference:
        [1] `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

        :param U: The tensor of weights corresponding to the vectors to project, of shape
            ``[..., m]``.
        :param G: The Gramian matrix of shape ``[m, m]``. It must be symmetric and positive
            semi-definite.
        :return: A tensor of projection weights with the same shape as ``U``.
        """


def projector_or_default(projector: DualConeProjector | None) -> DualConeProjector:
    if projector is None:
        return QuadprogProjector()
    return projector


class QuadprogProjector(_WithOptionalDeps, DualConeProjector):
    r"""
    Solves the quadratic program defined in :meth:`DualConeProjector.__call__` using the
    `quadprog <https://github.com/quadprog/quadprog>`_ QP solver.

    :param norm_eps: A small value to avoid division by zero when normalizing.
    :param reg_eps: A small value to add to the diagonal of the gramian of the matrix. Due to
        numerical errors when computing the gramian, it might not exactly be positive definite.
        This issue can make the optimization fail. Adding ``reg_eps`` to the diagonal of the gramian
        ensures that it is positive definite.
    """

    _REQUIRED_DEPS = ["numpy", "qpsolvers", "quadprog"]
    _INSTALL_HINT = 'Install them with: pip install "torchjd[quadprog_projector]"'

    def __init__(
        self,
        *,
        norm_eps: float = 0.0001,
        reg_eps: float = 0.0001,
    ) -> None:
        super().__init__()
        self._norm_eps = norm_eps
        self._reg_eps = reg_eps

    @property
    def norm_eps(self) -> float:
        return self._norm_eps

    @norm_eps.setter
    def norm_eps(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"norm_eps must be non-negative, but got {value}.")
        self._norm_eps = value

    @property
    def reg_eps(self) -> float:
        return self._reg_eps

    @reg_eps.setter
    def reg_eps(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"reg_eps must be non-negative, but got {value}.")
        self._reg_eps = value

    def __repr__(self) -> str:
        return f"QuadprogProjector(norm_eps={self._norm_eps}, reg_eps={self._reg_eps})"

    def __call__(self, U: Tensor, G: PSDMatrix) -> Tensor:

        G = regularize(normalize(G, self._norm_eps), self._reg_eps)

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
