# Partly adapted from https://github.com/OptMN-Lab/fairgrad/blob/main/methods/weight_methods.py#L811-L825 — MIT License, Copyright (c) 2024 OptMN-Lab.
# See NOTICES for the full license text.

from __future__ import annotations

import contextlib

import numpy as np
import torch
from torch import Tensor

from torchjd.linalg import PSDMatrix

from ._aggregator_bases import GramianWeightedAggregator
from ._mixins import _NonDifferentiable, _WithOptionalDeps
from ._weighting_bases import _GramianWeighting

with contextlib.suppress(ImportError):
    from scipy.optimize import least_squares


# Non-differentiable: the scipy solver operates on numpy arrays, breaking the autograd graph.
class FairGradWeighting(_WithOptionalDeps, _NonDifferentiable, _GramianWeighting):
    r"""
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.PSDMatrix`] giving the
    weights of :class:`~torchjd.aggregation.FairGrad`, as defined in Equation 4 of `Fair Resource
    Allocation in Multi-Task Learning <https://arxiv.org/pdf/2402.15638>`_.

    :param alpha: The parameter controlling the type of fairness in the alpha-fairness
        formulation.
    :param max_iters: The maximum number of iterations of the optimization loop. If set to None,
        the default value of ``scipy.optimize.least_squares`` (``100 * m``) will be used.

    .. note::
        This implementation was adapted from the `official implementation
        <https://github.com/OptMN-Lab/fairgrad/blob/main/methods/weight_methods.py#L811-L825>`_.

    .. note::
        This aggregator requires optional dependencies. When they are not installed, instantiating
        it raises an :class:`ImportError` with installation instructions.
        To install them, use ``pip install "torchjd[fairgrad]"``.
    """

    _REQUIRED_DEPS = ["scipy"]
    _INSTALL_HINT = 'Install it with: pip install "torchjd[fairgrad]"'

    def __init__(self, alpha: float, max_iters: int | None = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.max_iters = max_iters

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        m = gramian.shape[0]
        uniform = np.ones(m) / m

        if self.alpha == 0:
            # When alpha=0, the alpha-fairness formulation reduces to linear scalarization with
            # uniform weights (see Section 3 of https://arxiv.org/pdf/2402.15638).
            weight_array = uniform
        else:
            gramian_array = gramian.detach().cpu().numpy()

            def objective(x: np.ndarray) -> np.ndarray:
                return np.dot(gramian_array, x) - np.power(x, -1.0 / self.alpha)

            res = least_squares(objective, uniform, bounds=(0, np.inf), max_nfev=self.max_iters)
            weight_array = res.x

        return torch.tensor(weight_array).to(device=gramian.device, dtype=gramian.dtype)

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha}, max_iters={self.max_iters})"

    def __str__(self) -> str:
        return f"{self.alpha}-FairGradWeighting"


class FairGrad(_NonDifferentiable, GramianWeightedAggregator):
    r"""
    :class:`~torchjd.aggregation.GramianWeightedAggregator` using the step decision of Algorithm 1
    of `Fair Resource Allocation in Multi-Task Learning
    <https://arxiv.org/pdf/2402.15638.pdf>`_.

    :param alpha: The parameter controlling the type of fairness in the alpha-fairness
        formulation.
    :param max_iters: The maximum number of iterations of the optimization loop. If set to None,
        the default value of ``scipy.optimize.least_squares`` (``100 * m``) will be used.

    .. note::
        This aggregator requires optional dependencies. When they are not installed, instantiating
        it raises an :class:`ImportError` with installation instructions.
        To install them, use ``pip install "torchjd[fairgrad]"``.
    """

    gramian_weighting: FairGradWeighting

    def __init__(self, alpha: float, max_iters: int | None = None) -> None:
        super().__init__(FairGradWeighting(alpha=alpha, max_iters=max_iters))

    @property
    def alpha(self) -> float:
        return self.gramian_weighting.alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self.gramian_weighting.alpha = value

    @property
    def max_iters(self) -> int | None:
        return self.gramian_weighting.max_iters

    @max_iters.setter
    def max_iters(self, value: int | None) -> None:
        self.gramian_weighting.max_iters = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha}, max_iters={self.max_iters})"

    def __str__(self) -> str:
        return f"{self.alpha}-FairGrad"
