# Partly adapted from https://github.com/AvivNavon/nash-mtl — MIT License, Copyright (c) 2022 Aviv Navon.
# See NOTICES for the full license text.

from __future__ import annotations

import contextlib

import torch
from torch import Tensor

from torchjd._mixins import Stateful, _WithOptionalDeps
from torchjd.aggregation._mixins import _NonDifferentiable

from ._aggregator_bases import WeightedAggregator
from ._weighting_bases import _MatrixWeighting

with contextlib.suppress(ImportError):
    import cvxpy as cp
    import numpy as np
    from cvxpy import Expression, SolverError


# Non-differentiable: the cvxpy solver operates on numpy arrays, breaking the autograd graph.
class _NashMTLWeighting(_WithOptionalDeps, _MatrixWeighting, Stateful, _NonDifferentiable):
    _REQUIRED_DEPS = ["numpy", "cvxpy", "ecos"]
    _INSTALL_HINT = 'Install them with: pip install "torchjd[nash_mtl]"'

    """
    :class:`~torchjd.Stateful`
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
            raise ValueError(f"update_weights_every must be a positive integer, but got {value}.")
        self._update_weights_every = value

    @property
    def optim_niter(self) -> int:
        return self._optim_niter

    @optim_niter.setter
    def optim_niter(self, value: int) -> None:
        if value <= 0:
            raise ValueError(f"optim_niter must be a positive integer, but got {value}.")
        self._optim_niter = value

    def reset(self) -> None:
        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = np.eye(self.n_tasks)
        self.step = 0.0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)

    def forward(self, matrices: Tensor, /) -> Tensor:
        return matrices[0]
