from __future__ import annotations

from typing import TypeVar

from torch import Tensor

from torchjd.aggregation._mixins import Stateful

from ._weighting_bases import Weighting

_T = TypeVar("_T", contravariant=True, bound=Tensor)


class CRMOGMWeighting(Weighting[_T], Stateful):
    r"""
    :class:`~torchjd.aggregation.Stateful`
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

    where :math:`\lambda_0` is ``initial_weights`` if provided, otherwise
    :math:`\lambda_0 = \hat{\lambda}_1` (so that the first smoothed output equals
    :math:`\hat{\lambda}_1` regardless of :math:`\alpha`).

    Creating the corresponding :class:`~torchjd.aggregation.Aggregator` from a wrapped weighting can
    be done by composing it with the appropriate aggregator subclass
    (:class:`~torchjd.aggregation.WeightedAggregator` or
    :class:`~torchjd.aggregation.GramianWeightedAggregator`)

    The following example shows how to instantiate a Gramian-based weighted aggregator whose
    Gramian weighting is wrapped by CR-MOGM.

    .. testcode:: python

        from torchjd.aggregation import CRMOGMWeighting, GramianWeightedAggregator, UPGradWeighting

        aggregator = GramianWeightedAggregator(CRMOGMWeighting(UPGradWeighting()))

    The following example shows how to instantiate a Matrix-based weighted aggregator whose
    weighting is wrapped by CR-MOGM.

    .. testcode:: python

        from torchjd.aggregation import CRMOGMWeighting, MeanWeighting, WeightedAggregator

        aggregator = WeightedAggregator(CRMOGMWeighting(MeanWeighting()))

    Note that here, :class:`~torchjd.aggregation.MeanWeighting` is used just for the sake of the
    example: the exponential moving average of constant weights will always be equal to the weights
    themselves, so wrapping by ``CRMOGMWeighting`` will have no effect.

    This weighting is stateful: it keeps :math:`\lambda_{k-1}` across calls. Use :meth:`reset`
    to restart the smoothing from the initial state. Note that calling :meth:`reset` will also
    reset the wrapped weighting if it is :class:`~torchjd.aggregation.Stateful`.

    :param weighting: The wrapped weighting whose output is smoothed.
    :param alpha: EMA coefficient on the previous weights. ``alpha=0`` disables smoothing
        (``CRMOGMWeighting`` returns ``weighting``'s output verbatim) and ``alpha=1`` freezes
        the weights at their initial value. The default of ``0.9`` follows the usual EMA
        convention (analogous to Adam's :math:`\beta_1`).
    :param initial_weights: Optional tensor to use as :math:`\lambda_0`. If ``None`` (default),
        :math:`\lambda_0` is set to :math:`\hat{\lambda}_1` on the first forward call, making
        the first smoothed output equal to :math:`\hat{\lambda}_1`.

    .. note::
        ``alpha`` is a fixed ``float`` for simplicity. Corollary 1 of the paper recommends a
        schedule where :math:`\alpha_k` starts near 0 and increases toward 1 as the learning
        rate decays. Update ``alpha`` between forward calls via the setter.

        The following example shows how to update alpha with the suggested scheme from the paper,
        when the aggregator is a Gramian-based weighted aggregator whose Gramian weighting is
        wrapped by CR-MOGM:

        .. testcode:: python

            from torchjd.aggregation import (
                CRMOGMWeighting,
                GramianWeightedAggregator,
                UPGradWeighting,
            )

            aggregator = GramianWeightedAggregator(CRMOGMWeighting(UPGradWeighting()))

            initial_lr = 0.1
            current_lr = 0.05  # e.g. obtained from lr_scheduler.get_lr()[0]

            cr_mogm = aggregator.gramian_weighting
            cr_mogm.alpha = 1 - current_lr / initial_lr
    """

    def __init__(
        self, weighting: Weighting[_T], alpha: float = 0.9, initial_weights: Tensor | None = None
    ) -> None:
        super().__init__()
        self.weighting = weighting
        self.alpha = alpha
        self._initial_weights = initial_weights
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
        r"""
        Clears the EMA state so the next forward restarts from the initial state. Also resets the
        wrapped weighting if it is :class:`~torchjd.aggregation.Stateful`.
        """

        if isinstance(self.weighting, Stateful):
            self.weighting.reset()
        self._lambda = None

    def forward(self, stat: _T, /) -> Tensor:
        lambda_hat = self.weighting(stat)
        lambda_prev = self._ensure_state(lambda_hat)
        self._lambda = self._alpha * lambda_prev + (1.0 - self._alpha) * lambda_hat
        return self._lambda

    def _ensure_state(self, lambda_hat: Tensor) -> Tensor:
        m = lambda_hat.shape[0]
        if self._lambda is None:
            if self._initial_weights is not None:
                if self._initial_weights.shape != (m,):
                    raise ValueError(
                        f"`initial_weights` has shape {tuple(self._initial_weights.shape)}, "
                        f"expected ({m},)."
                    )
                self._lambda = self._initial_weights.to(
                    dtype=lambda_hat.dtype, device=lambda_hat.device
                )
            else:
                self._lambda = lambda_hat
        elif self._lambda.shape[0] != m:
            raise ValueError(
                f"The number of objectives changed from {self._lambda.shape[0]} to {m}. Call "
                f"`reset()` before changing the number of objectives."
            )
        return self._lambda
