import contextlib
from typing import cast

import torch
from torch import Tensor

from torchjd._linalg import normalize
from torchjd._mixins import _WithOptionalDeps
from torchjd.linalg import PSDMatrix

from ._aggregator_bases import GramianWeightedAggregator
from ._mixins import _NonDifferentiable
from ._weighting_bases import _GramianWeighting

with contextlib.suppress(ImportError):
    import cvxpy as cp
    import numpy as np


# Non-differentiable: the cvxpy solver operates on numpy arrays, breaking the autograd graph.
class CAGradWeighting(_WithOptionalDeps, _NonDifferentiable, _GramianWeighting):
    _REQUIRED_DEPS = ["numpy", "cvxpy", "clarabel"]
    _INSTALL_HINT = 'Install them with: pip install "torchjd[cagrad]"'
    """
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.PSDMatrix`]
    giving the weights of :class:`~torchjd.aggregation.CAGrad`.

    :param c: The scale of the radius of the ball constraint.
    :param norm_eps: A small value to avoid division by zero when normalizing.

    .. note::
        This implementation differs from the `official implementations
        <https://github.com/Cranial-XIX/CAGrad/>`_ in the way the underlying optimization problem is
        solved. This uses the `CLARABEL <https://oxfordcontrol.github.io/ClarabelDocs/stable/>`_
        solver of `cvxpy <https://www.cvxpy.org/index.html>`_ rather than the `scipy.minimize
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
        function.
    """

    def __init__(self, c: float, norm_eps: float = 0.0001) -> None:
        super().__init__()
        self.c = c
        self.norm_eps = norm_eps

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        U, S, _ = torch.svd(normalize(gramian, self.norm_eps))

        reduced_matrix = U @ S.sqrt().diag()
        reduced_array = reduced_matrix.cpu().detach().numpy().astype(np.float64)

        dimension = gramian.shape[0]
        reduced_g_0 = reduced_array.T @ np.ones(dimension) / dimension
        sqrt_phi = self.c * np.linalg.norm(reduced_g_0, 2).item()

        w = cp.Variable(shape=dimension)
        cost = (reduced_array @ reduced_g_0).T @ w + sqrt_phi * cp.norm(reduced_array.T @ w, 2)
        problem = cp.Problem(objective=cp.Minimize(cost), constraints=[w >= 0, cp.sum(w) == 1])

        problem.solve(cp.CLARABEL)
        w_opt = cast(np.ndarray, w.value)

        g_w_norm = np.linalg.norm(reduced_array.T @ w_opt, 2).item()
        if g_w_norm >= self.norm_eps:
            weight_array = np.ones(dimension) / dimension
            weight_array += (sqrt_phi / g_w_norm) * w_opt
        else:
            # We are approximately on the pareto front
            weight_array = np.zeros(dimension)

        weights = torch.from_numpy(weight_array).to(device=gramian.device, dtype=gramian.dtype)

        return weights

    @property
    def c(self) -> float:
        return self._c

    @c.setter
    def c(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"c must be non-negative, but got {value}.")

        self._c = value

    @property
    def norm_eps(self) -> float:
        return self._norm_eps

    @norm_eps.setter
    def norm_eps(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"norm_eps must be non-negative, but got {value}.")

        self._norm_eps = value


class CAGrad(_NonDifferentiable, GramianWeightedAggregator):
    """
    :class:`~torchjd.aggregation.GramianWeightedAggregator` as defined in Algorithm 1 of
    `Conflict-Averse Gradient Descent for Multi-task Learning
    <https://arxiv.org/pdf/2110.14048.pdf>`_.

    :param c: The scale of the radius of the ball constraint.
    :param norm_eps: A small value to avoid division by zero when normalizing.

    .. note::
        This aggregator requires optional dependencies. When they are not installed, instantiating
        it raises an :class:`ImportError` with installation instructions.
        To install them, use ``pip install "torchjd[cagrad]"``.
    """

    gramian_weighting: CAGradWeighting

    def __init__(self, c: float, norm_eps: float = 0.0001) -> None:
        super().__init__(CAGradWeighting(c=c, norm_eps=norm_eps))

    @property
    def c(self) -> float:
        return self.gramian_weighting.c

    @c.setter
    def c(self, value: float) -> None:
        self.gramian_weighting.c = value

    @property
    def norm_eps(self) -> float:
        return self.gramian_weighting.norm_eps

    @norm_eps.setter
    def norm_eps(self, value: float) -> None:
        self.gramian_weighting.norm_eps = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(c={self.c}, norm_eps={self.norm_eps})"

    def __str__(self) -> str:
        c_str = str(self.c).rstrip("0")
        return f"CAGrad{c_str}"
