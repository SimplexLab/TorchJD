from typing import cast

from torchjd._linalg import PSDMatrix

from ._utils.check_dependencies import check_dependencies_are_installed
from ._weighting_bases import Weighting

check_dependencies_are_installed(["cvxpy", "clarabel"])

import cvxpy as cp
import numpy as np
import torch
from torch import Tensor

from torchjd._linalg import normalize

from ._aggregator_bases import GramianWeightedAggregator
from ._utils.non_differentiable import raise_non_differentiable_error


class CAGrad(GramianWeightedAggregator):
    """
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` as defined in Algorithm 1 of
    `Conflict-Averse Gradient Descent for Multi-task Learning
    <https://arxiv.org/pdf/2110.14048.pdf>`_.

    :param c: The scale of the radius of the ball constraint.
    :param norm_eps: A small value to avoid division by zero when normalizing.

    .. note::
        This aggregator is not installed by default. When not installed, trying to import it should
        result in the following error:
        ``ImportError: cannot import name 'CAGrad' from 'torchjd.aggregation'``.
        To install it, use ``pip install "torchjd[cagrad]"``.
    """

    def __init__(self, c: float, norm_eps: float = 0.0001) -> None:
        super().__init__(CAGradWeighting(c=c, norm_eps=norm_eps))
        self._c = c
        self._norm_eps = norm_eps

        # This prevents considering the computed weights as constant w.r.t. the matrix.
        self.register_full_backward_pre_hook(raise_non_differentiable_error)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(c={self._c}, norm_eps={self._norm_eps})"

    def __str__(self) -> str:
        c_str = str(self._c).rstrip("0")
        return f"CAGrad{c_str}"


class CAGradWeighting(Weighting[PSDMatrix]):
    """
    :class:`~torchjd.aggregation._weighting_bases.Weighting` giving the weights of
    :class:`~torchjd.aggregation.CAGrad`.

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

        if c < 0.0:
            raise ValueError(f"Parameter `c` should be a non-negative float. Found `c = {c}`.")

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
