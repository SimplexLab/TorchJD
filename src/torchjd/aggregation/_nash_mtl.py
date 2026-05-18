# Partly adapted from https://github.com/AvivNavon/nash-mtl — MIT License, Copyright (c) 2022 Aviv Navon.
# See NOTICES for the full license text.

from __future__ import annotations

import contextlib

import torch
from torch import Tensor

from torchjd._mixins import _WithOptionalDeps
from torchjd.aggregation._mixins import Stateful, _NonDifferentiable

from ._aggregator_bases import WeightedAggregator
from ._weighting_bases import _MatrixWeighting

with contextlib.suppress(ImportError):
    import cvxpy as cp
    import numpy as np
    from cvxpy import Expression, SolverError


# Non-differentiable: the cvxpy solver operates on numpy arrays, breaking the autograd graph.
class _NashMTLWeighting(_WithOptionalDeps, _NonDifferentiable, Stateful, _MatrixWeighting):
    _REQUIRED_DEPS = ["numpy", "cvxpy", "ecos"]
    _INSTALL_HINT = 'Install them with: pip install "torchjd[nash_mtl]"'
    """
    :class:`~torchjd.aggregation._mixins.Stateful`
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.Matrix`] that
    extracts weights using the step decision of Algorithm 1 of `Multi-Task Learning as a Bargaining
    Game <https://arxiv.org/pdf/2202.01017.pdf>`_.

    :param n_tasks: The number of tasks, corresponding to the number of rows in the provided
        matrices.
    :param max_norm: Maximum value of the norm of :math:`J^T w`. A value of ``0`` disables the
        norm clipping.
    :param update_weights_every: A parameter determining how often the actual weighting should be
        performed. A larger value means that the same weights will be re-used for more calls to the
        weighting.
    :param optim_niter: The number of iterations of the underlying optimization process.

    .. note::
        Changing any of these parameters after instantiation does not automatically reset the
        internal state. Call :meth:`reset` if needed (especially after changing ``n_tasks``, which
        affects the shape of the cached state).
    """

    def __init__(
        self,
        n_tasks: int,
        max_norm: float,
        update_weights_every: int,
        optim_niter: int,
    ) -> None:
        super().__init__()

        self.n_tasks = n_tasks
        self.optim_niter = optim_niter
        self.update_weights_every = update_weights_every
        self.max_norm = max_norm

        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = np.eye(self.n_tasks)
        self.step = 0.0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)

    @property
    def n_tasks(self) -> int:
        return self._n_tasks

    @n_tasks.setter
    def n_tasks(self, value: int) -> None:
        if value <= 0:
            raise ValueError(f"n_tasks must be a positive integer, but got {value}.")

        self._n_tasks = value

    @property
    def max_norm(self) -> float:
        return self._max_norm

    @max_norm.setter
    def max_norm(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"max_norm must be non-negative, but got {value}.")

        self._max_norm = value

    @property
    def update_weights_every(self) -> int:
        return self._update_weights_every

    @update_weights_every.setter
    def update_weights_every(self, value: int) -> None:
        if value <= 0:
            raise ValueError(
                f"update_weights_every must be a positive integer, but got {value}.",
            )

        self._update_weights_every = value

    @property
    def optim_niter(self) -> int:
        return self._optim_niter

    @optim_niter.setter
    def optim_niter(self, value: int) -> None:
        if value <= 0:
            raise ValueError(f"optim_niter must be a positive integer, but got {value}.")

        self._optim_niter = value

    def _stop_criteria(self, gtg: np.ndarray, alpha_t: np.ndarray) -> bool:
        return bool(
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value) < 1e-6),
        )

    def _solve_optimization(self, gtg: np.ndarray) -> np.ndarray:
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except (SolverError, ValueError):
                # On macOS, SolverError can happen with: Solver 'ECOS' failed.
                # No idea why. The corresponding matrix is of shape [9, 11] with rank 5.
                # ValueError happens with for example matrix [[0., 0.], [0., 1.]].
                # Maybe other exceptions can happen in other cases.
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self) -> Expression:
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self) -> None:
        self.alpha_param = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(shape=(self.n_tasks,), value=self.prvs_alpha)
        self.G_param = cp.Parameter(shape=(self.n_tasks, self.n_tasks), value=self.init_gtg)
        self.normalization_factor_param = cp.Parameter(shape=(1,), value=np.array([1.0]))

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = [
            -cp.log(a * self.normalization_factor_param) - cp.log(G_a) <= 0
            for a, G_a in zip(self.alpha_param, G_alpha, strict=True)
        ]
        obj = cp.Minimize(cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param)
        self.prob = cp.Problem(obj, constraint)

    def forward(self, matrix: Tensor, /) -> Tensor:
        if self.step == 0:
            self._init_optim_problem()

        if (self.step % self.update_weights_every) == 0:
            self.step += 1

            G = matrix
            GTG = torch.mm(G, G.t())

            self.normalization_factor = torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            GTG = GTG / self.normalization_factor.item()
            alpha = self._solve_optimization(GTG.cpu().detach().numpy())
        else:
            self.step += 1
            alpha = self.prvs_alpha

        alpha = torch.from_numpy(alpha).to(device=matrix.device, dtype=matrix.dtype)

        if self.max_norm > 0:
            norm = torch.linalg.norm(alpha @ matrix)
            if norm > self.max_norm:
                alpha = (alpha / norm) * self.max_norm

        return alpha

    def reset(self) -> None:
        """Resets the internal state of the algorithm."""

        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = np.eye(self.n_tasks)
        self.step = 0.0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)


class NashMTL(_NonDifferentiable, Stateful, WeightedAggregator):
    """
    :class:`~torchjd.aggregation._mixins.Stateful`
    :class:`~torchjd.aggregation.WeightedAggregator` as proposed in Algorithm 1 of
    `Multi-Task Learning as a Bargaining Game <https://arxiv.org/pdf/2202.01017.pdf>`_.

    :param n_tasks: The number of tasks, corresponding to the number of rows in the provided
        matrices.
    :param max_norm: Maximum value of the norm of :math:`J^T w`. A value of ``0`` disables the
        norm clipping.
    :param update_weights_every: A parameter determining how often the actual weighting should be
        performed. A larger value means that the same weights will be re-used for more calls to the
        aggregator.
    :param optim_niter: The number of iterations of the underlying optimization process.

    .. note::
        This aggregator requires optional dependencies. When they are not installed, instantiating
        it raises an :class:`ImportError` with installation instructions.
        To install them, use ``pip install "torchjd[nash_mtl]"``.

    .. warning::
        This implementation was adapted from the `official implementation
        <https://github.com/AvivNavon/nash-mtl/tree/main>`_, which has some flaws. Use with caution.

    .. warning::
        This aggregator is stateful. Its output will thus depend not only on the input matrix, but
        also on its state. It thus depends on previously seen matrices. It should be reset between
        experiments.

    .. note::
        Changing any of these parameters after instantiation does not automatically reset the
        internal state. Call :meth:`reset` if needed (especially after changing ``n_tasks``, which
        affects the shape of the cached state).
    """

    weighting: _NashMTLWeighting

    def __init__(
        self,
        n_tasks: int,
        max_norm: float = 1.0,
        update_weights_every: int = 1,
        optim_niter: int = 20,
    ) -> None:
        super().__init__(
            weighting=_NashMTLWeighting(
                n_tasks=n_tasks,
                max_norm=max_norm,
                update_weights_every=update_weights_every,
                optim_niter=optim_niter,
            ),
        )

    @property
    def n_tasks(self) -> int:
        return self.weighting.n_tasks

    @n_tasks.setter
    def n_tasks(self, value: int) -> None:
        self.weighting.n_tasks = value

    @property
    def max_norm(self) -> float:
        return self.weighting.max_norm

    @max_norm.setter
    def max_norm(self, value: float) -> None:
        self.weighting.max_norm = value

    @property
    def update_weights_every(self) -> int:
        return self.weighting.update_weights_every

    @update_weights_every.setter
    def update_weights_every(self, value: int) -> None:
        self.weighting.update_weights_every = value

    @property
    def optim_niter(self) -> int:
        return self.weighting.optim_niter

    @optim_niter.setter
    def optim_niter(self, value: int) -> None:
        self.weighting.optim_niter = value

    def reset(self) -> None:
        """Resets the internal state of the algorithm."""
        self.weighting.reset()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_tasks={self.n_tasks}, max_norm={self.max_norm}, "
            f"update_weights_every={self.update_weights_every}, optim_niter={self.optim_niter})"
        )
