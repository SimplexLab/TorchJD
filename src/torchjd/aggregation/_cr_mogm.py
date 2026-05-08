from __future__ import annotations

from typing import TypeVar

import torch
from torch import Tensor

from torchjd.aggregation._mixins import Stateful

from ._weighting_bases import Weighting

_T = TypeVar("_T", contravariant=True, bound=Tensor)


class CRMOGMWeighting(Weighting[_T], Stateful):
    r"""
    :class:`~torchjd.aggregation._mixins.Stateful`
    :class:`~torchjd.aggregation._weighting_bases.Weighting` that wraps another
    :class:`~torchjd.aggregation._weighting_bases.Weighting` and stabilises the weights it
    produces with an exponential moving average (EMA) across calls. This is the weight-smoothing
    modifier from `On the Convergence of Stochastic Multi-Objective Gradient Manipulation and
    Beyond <https://proceedings.neurips.cc/paper_files/paper/2022/file/f91bd64a3620aad8e70a27ad9cb3ca57-Paper-Conference.pdf>`_
    (NeurIPS 2022).

    Let :math:`\hat{\lambda}_k` be the weights returned by the wrapped weighting at step
    :math:`k`. The smoothed weights returned by ``CRMOGMWeighting`` are:

    .. math::

        \lambda_k = \alpha \, \lambda_{k-1} + (1 - \alpha) \, \hat{\lambda}_k

    with :math:`\lambda_0 = \begin{bmatrix} \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^\top
    \in \mathbb{R}^m`. The state :math:`\lambda_{k-1}` is initialised lazily on the first
    forward call once :math:`m` is known and is reset automatically when ``m`` changes.

    Because ``CRMOGMWeighting`` is generic in the input type ``_T``, it can wrap either a
    ``MatrixWeighting`` or a ``GramianWeighting``. Creating a corresponding :class:`~torchjd.aggregation.Aggregator` can be done by composing it with the appropriate
    aggregator base:

    .. code-block:: python

        from torchjd.aggregation import MeanWeighting, UPGradWeighting
        from torchjd.aggregation._aggregator_bases import (
            GramianWeightedAggregator, WeightedAggregator,
        )
        from torchjd.aggregation._cr_mogm import CRMOGMWeighting

        matrix_aggregator = WeightedAggregator(CRMOGMWeighting(MeanWeighting()))
        gramian_aggregator = GramianWeightedAggregator(CRMOGMWeighting(UPGradWeighting()))

    This weighting is stateful: it keeps :math:`\lambda_{k-1}` across calls. Use :meth:`reset`
    when restarting the smoothing from uniform weights. Note that calling :meth:`reset` will also
    reset the wrapped weighting if it is :class:`~torchjd.aggregation.Stateful`.

    :param weighting: The wrapped weighting whose output is smoothed.
    :param alpha: EMA coefficient on the previous weights. ``alpha=0`` disables smoothing
        (``CRMOGMWeighting`` returns ``weighting``'s output verbatim) and ``alpha=1`` freezes
        the weights at their initial uniform value. The default of ``0.9`` follows the usual
        EMA convention (analogous to Adam's :math:`\beta_1`).

    .. note::
        ``alpha`` is a fixed ``float`` for simplicity. Corollary 1 of the paper recommends a
        schedule where :math:`\alpha_k` starts near 0 and increases toward 1 as the learning
        rate decays. Update ``alpha`` between forward calls via the public attribute on the
        wrapping aggregator:

        .. code-block:: python

            # With WeightedAggregator
            aggregator.weighting.alpha = 1 - current_lr / initial_lr

            # With GramianWeightedAggregator
            aggregator.gramian_weighting.alpha = 1 - current_lr / initial_lr
    """

    def __init__(self, weighting: Weighting[_T], alpha: float = 0.1) -> None:
        super().__init__()
        self.weighting = weighting
        self.alpha = alpha
        self._lambda: Tensor | None = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Attribute `alpha` must be in [0, 1]. Found alpha={value!r}.")
        self._alpha = value

    def reset(self) -> None:
        """Clears the EMA state so the next forward starts from uniform weights."""

        if isinstance(self.weighting, Stateful):
            self.weighting.reset()
        self._lambda = None

    def forward(self, stat: _T, /) -> Tensor:
        lambda_hat = self.weighting(stat)

        lambda_prev = self._ensure_state(lambda_hat.shape[0], lambda_hat.dtype, lambda_hat.device)

        lambda_k = self._alpha * lambda_prev + (1.0 - self._alpha) * lambda_hat

        self._lambda = lambda_k.detach()
        return lambda_k

    def _ensure_state(self, m: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        if self._lambda is None or self._lambda.shape[0] != m:
            if m > 0:
                self._lambda = torch.full((m,), 1.0 / m, dtype=dtype, device=device)
            else:
                self._lambda = torch.zeros(0, dtype=dtype, device=device)
        return self._lambda
