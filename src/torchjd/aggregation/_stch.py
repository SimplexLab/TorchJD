# The code of this file was adapted from
# https://github.com/Xi-L/STCH/blob/main/STCH_MTL/LibMTL/weighting/STCH.py.
# It is therefore also subject to the following license.
#
# MIT License
#
# Copyright (c) 2024 Xi Lin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import Tensor

from torchjd._linalg import PSDMatrix

from ._aggregator_bases import GramianWeightedAggregator
from ._weighting_bases import Weighting


class STCH(GramianWeightedAggregator):
    r"""
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` implementing the Smooth Tchebycheff
    scalarization as proposed in `Smooth Tchebycheff Scalarization for Multi-Objective Optimization
    <https://arxiv.org/abs/2402.19078>`_.

    This aggregator uses the log-sum-exp (smooth maximum) function to compute weights that focus
    more on poorly performing tasks (tasks with larger gradient norms). The ``mu`` parameter
    controls the smoothness: as ``mu`` approaches 0, the weights converge to a hard maximum
    (focusing entirely on the worst task); as ``mu`` increases, the weights approach uniform
    averaging.

    :param mu: The smoothness parameter for the log-sum-exp. Smaller values give more weight to the
        worst-performing task. Must be positive.
    :param warmup_steps: Optional number of steps for the warmup phase. During warmup, gradient
        norms are accumulated to compute a nadir vector for normalization. If ``None`` (default),
        no warmup is performed and raw gradient norms are used directly.
    :param eps: A small value to avoid numerical issues in log computations.

    .. warning::
        If ``warmup_steps`` is set, this aggregator becomes stateful. Its output will depend not
        only on the input matrix, but also on its internal state (previously seen matrices). It
        should be reset between experiments using the :meth:`reset` method.

    .. note::
        The original STCH algorithm operates on loss values. This implementation adapts it for
        gradient-based aggregation using gradient norms (derived from the Gramian diagonal) as
        proxies for task performance.

    Example
    -------

    >>> from torch import tensor
    >>> from torchjd.aggregation import STCH
    >>>
    >>> A = STCH(mu=1.0)
    >>> J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    >>> A(J)
    tensor([1.8188, 1.0000, 1.0000])

    .. note::
        This implementation was adapted from the `official implementation
        <https://github.com/Xi-L/STCH>`_.
    """

    def __init__(
        self,
        mu: float = 1.0,
        warmup_steps: int | None = None,
        eps: float = 1e-20,
    ):
        if mu <= 0.0:
            raise ValueError(f"Parameter `mu` should be a positive float. Found `mu = {mu}`.")

        if warmup_steps is not None and warmup_steps < 1:
            raise ValueError(
                f"Parameter `warmup_steps` should be a positive integer or None. "
                f"Found `warmup_steps = {warmup_steps}`."
            )

        stch_weighting = STCHWeighting(mu=mu, warmup_steps=warmup_steps, eps=eps)
        super().__init__(stch_weighting)

        self._mu = mu
        self._warmup_steps = warmup_steps
        self._eps = eps
        self._stch_weighting = stch_weighting

    def reset(self) -> None:
        """Resets the internal state of the algorithm (step counter and accumulated nadir)."""
        self._stch_weighting.reset()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mu={self._mu}, warmup_steps={self._warmup_steps}, "
            f"eps={self._eps})"
        )

    def __str__(self) -> str:
        mu_str = str(self._mu).rstrip("0").rstrip(".")
        return f"STCH(mu={mu_str})"


class STCHWeighting(Weighting[PSDMatrix]):
    r"""
    :class:`~torchjd.aggregation._weighting_bases.Weighting` giving the weights of
    :class:`~torchjd.aggregation.STCH`.

    The weights are computed using the Smooth Tchebycheff scalarization formula:

    .. math::

        w_i = \frac{\exp\left(\frac{\log(g_i / z_i) - \max_j \log(g_j / z_j)}{\mu}\right)}
              {\sum_k \exp\left(\frac{\log(g_k / z_k) - \max_j \log(g_j / z_j)}{\mu}\right)}

    where :math:`g_i` is the gradient norm for task :math:`i` (computed as :math:`\sqrt{G_{ii}}`
    from the Gramian), :math:`z_i` is the nadir value for task :math:`i`, and :math:`\mu` is the
    smoothness parameter.

    :param mu: The smoothness parameter for the log-sum-exp. Must be positive.
    :param warmup_steps: Optional number of steps for the warmup phase. During warmup, gradient
        norms are accumulated to compute a nadir vector. If ``None``, no warmup is performed.
    :param eps: A small value to avoid numerical issues in log computations.

    .. warning::
        If ``warmup_steps`` is set, this weighting becomes stateful. During warmup, it returns
        uniform weights while accumulating gradient norms. After warmup, the accumulated average
        is used as the nadir vector for normalization.
    """

    def __init__(
        self,
        mu: float = 1.0,
        warmup_steps: int | None = None,
        eps: float = 1e-20,
    ):
        super().__init__()

        if mu <= 0.0:
            raise ValueError(f"Parameter `mu` should be a positive float. Found `mu = {mu}`.")

        if warmup_steps is not None and warmup_steps < 1:
            raise ValueError(
                f"Parameter `warmup_steps` should be a positive integer or None. "
                f"Found `warmup_steps = {warmup_steps}`."
            )

        self.mu = mu
        self.warmup_steps = warmup_steps
        self.eps = eps

        # Internal state for warmup
        self.step = 0
        self.nadir_accumulator: Tensor | None = None
        self.nadir_vector: Tensor | None = None

    def reset(self) -> None:
        """Resets the internal state of the algorithm."""
        self.step = 0
        self.nadir_accumulator = None
        self.nadir_vector = None

    def forward(self, gramian: PSDMatrix) -> Tensor:
        device = gramian.device
        dtype = gramian.dtype
        m = gramian.shape[0]

        # Compute gradient norms from Gramian diagonal (sqrt of diagonal)
        grad_norms = torch.sqrt(torch.diag(gramian).clamp(min=self.eps))

        # Handle warmup phase if warmup_steps is set
        if self.warmup_steps is not None:
            if self.step < self.warmup_steps:
                # During warmup: accumulate gradient norms and return uniform weights
                if self.nadir_accumulator is None:
                    self.nadir_accumulator = grad_norms.detach().clone()
                else:
                    self.nadir_accumulator = (
                        self.nadir_accumulator.to(device=device, dtype=dtype) + grad_norms.detach()
                    )

                self.step += 1

                # Return uniform weights during warmup
                return torch.full(size=[m], fill_value=1.0 / m, device=device, dtype=dtype)

            elif self.nadir_vector is None:
                # First step after warmup: compute nadir vector from accumulated values
                self.nadir_vector = self.nadir_accumulator / self.warmup_steps  # type: ignore
                self.step += 1
            else:
                self.step += 1

        # Normalize by nadir vector if available (after warmup)
        if self.nadir_vector is not None:
            nadir = self.nadir_vector.to(device=device, dtype=dtype)
            normalized = grad_norms / nadir.clamp(min=self.eps)
        else:
            normalized = grad_norms

        # Apply log and compute smooth max weights using log-sum-exp trick for numerical stability
        log_normalized = torch.log(normalized + self.eps)
        max_log = torch.max(log_normalized)
        reg_log = (log_normalized - max_log) / self.mu

        # Softmax weights
        exp_reg = torch.exp(reg_log)
        weights = exp_reg / exp_reg.sum()

        return weights
