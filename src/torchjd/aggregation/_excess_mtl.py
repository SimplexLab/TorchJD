# Partly adapted from https://github.com/uiuctml/ExcessMTL — MIT License, Copyright (c) 2024 UIUC TML Lab.
# See NOTICES for the full license text.
from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from torchjd._mixins import Stateful
from torchjd.aggregation._mixins import _NonDifferentiable
from torchjd.linalg import Matrix

from ._weighting_bases import _MatrixWeighting


class ExcessMTLWeighting(_MatrixWeighting, Stateful, _NonDifferentiable):
    r"""
    :class:`~torchjd.Stateful`
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.Matrix`] from `Robust
    Multi-Task Learning with Excess Risks
    <https://proceedings.mlr.press/v235/he24n.html>`_ (ICML 2024).

    At each call, task weights are updated via an exponentiated gradient step (Equation 9) driven
    by per-task excess risk estimates. The excess risk for task :math:`i` is approximated via a
    second-order Taylor expansion (Equations 6-7):

    :param robust_step_size: Step size :math:`\eta_\alpha` for the exponentiated weight update.
        Must be positive.
    :param n_warmup_steps: Number of forward calls during which weights stay uniform
        (:math:`[1/m, \ldots, 1/m]`) and gradient statistics are collected. The baseline excess
        risk is set to the average excess risk observed during warmup. When ``0`` (default), the
        first call's excess risk is used as the baseline and weights are updated immediately
        (matching the official implementation).

    .. warning::
        The state tensor :math:`S \in \mathbb{R}^{m \times n}` accumulates squared gradients
        across **all** calls, where :math:`n` is the total number of model parameters. For large
        models this can be a significant memory cost. Call :meth:`reset` between experiments.

    .. note::
        The weight update is adapted from the `official implementation
        <https://github.com/uiuctml/ExcessMTL>`_ and `LibMTL
        <https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/ExcessMTL.py>`_.
        The warmup strategy follows Appendix C.1 of the paper, which recommends collecting
        gradient statistics for several epochs before beginning weight updates; set
        ``n_warmup_steps`` accordingly (e.g. ``3 * len(dataloader)``).

    .. admonition:: Example

        .. testcode::

            import torch
            from torch.nn import Linear, MSELoss, ReLU, Sequential
            from torch.optim import SGD

            from torchjd import autojac
            from torchjd.aggregation import ExcessMTLWeighting, WeightedAggregator
            from torchjd.autojac import jac_to_grad

            inputs = torch.randn(8, 5)
            targets = torch.randn(8, 2)

            model = Sequential(Linear(5, 4), ReLU(), Linear(4, 2))
            optimizer = SGD(model.parameters())
            criterion = MSELoss()
            aggregator = WeightedAggregator(ExcessMTLWeighting())

            outputs = model(inputs)
            losses = [criterion(outputs[:, i], targets[:, i]) for i in range(2)]
            autojac.backward(losses)
            jac_to_grad(model.parameters(), aggregator)
            optimizer.step()
            optimizer.zero_grad()
    """

    def __init__(
        self,
        robust_step_size: float = 1.0,
        n_warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.robust_step_size = robust_step_size
        self.n_warmup_steps = n_warmup_steps
        self.register_buffer("_weights", None)
        self.register_buffer("_grad_sum", None)
        self.register_buffer("_initial_w", None)
        self.register_buffer("_warmup_w_sum", None)
        self.register_buffer("_n_steps", torch.zeros((), dtype=torch.long))
        self._state_key: tuple[int, int, torch.dtype, torch.device] | None = None

    @property
    def robust_step_size(self) -> float:
        return self._robust_step_size

    @robust_step_size.setter
    def robust_step_size(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(
                f"Attribute `robust_step_size` must be positive. Found robust_step_size={value!r}."
            )
        self._robust_step_size = value

    @property
    def n_warmup_steps(self) -> int:
        return self._n_warmup_steps

    @n_warmup_steps.setter
    def n_warmup_steps(self, value: int) -> None:
        if value < 0:
            raise ValueError(
                f"Attribute `n_warmup_steps` must be non-negative. Found n_warmup_steps={value!r}."
            )
        self._n_warmup_steps = value

    def reset(self) -> None:
        """Clears all state so the next forward starts from uniform weights and re-enters
        warmup."""

        self._weights = None
        self._grad_sum = None
        self._initial_w = None
        self._warmup_w_sum = None
        self._n_steps.zero_()
        self._state_key = None

    def forward(self, matrix: Matrix, /) -> Tensor:
        self._ensure_state(matrix)

        # Accumulate squared gradients for AdaGrad-style diagonal Hessian (Equation 7)
        grad_sum = cast(Tensor, self._grad_sum)
        grad_sum = grad_sum + matrix.detach() ** 2
        self._grad_sum = grad_sum

        # Excess risk proxy: Ê_i ≈ g_i^T H_i^{-1} g_i (Equation 6)
        h = torch.sqrt(grad_sum + 1e-7)
        w = (matrix.detach() ** 2 / h).sum(dim=1)  # shape [m]

        n_steps = int(self._n_steps.item())
        self._n_steps = self._n_steps + 1

        # Warmup: collect excess risk stats but return uniform weights
        if n_steps < self._n_warmup_steps:
            warmup_w_sum = self._warmup_w_sum
            self._warmup_w_sum = w if warmup_w_sum is None else cast(Tensor, warmup_w_sum) + w
            return cast(Tensor, self._weights)

        # Set baseline on the first non-warmup call
        if self._initial_w is None:
            if self._n_warmup_steps > 0:
                # Average excess risk observed during warmup (Appendix C.1)
                self._initial_w = cast(Tensor, self._warmup_w_sum) / self._n_warmup_steps
                w = w / (cast(Tensor, self._initial_w) + 1e-7)
            else:
                # Official impl behaviour: first call's excess is the baseline; use w raw
                self._initial_w = w
        else:
            w = w / (cast(Tensor, self._initial_w) + 1e-7)

        # Exponentiated gradient weight update (Equation 9)
        weights = cast(Tensor, self._weights)
        weights = weights * torch.exp(w * self._robust_step_size)
        weights = weights / weights.sum()
        self._weights = weights
        return weights

    def _ensure_state(self, matrix: Matrix) -> None:
        key = (matrix.shape[0], matrix.shape[1], matrix.dtype, matrix.device)
        if self._state_key == key and self._grad_sum is not None:
            return
        m, n = matrix.shape
        self._grad_sum = matrix.new_zeros(m, n)
        self._weights = matrix.new_full((m,), 1.0 / m)
        self._initial_w = None
        self._warmup_w_sum = None
        self._n_steps.zero_()
        self._state_key = key

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"robust_step_size={self.robust_step_size!r}, "
            f"n_warmup_steps={self.n_warmup_steps!r})"
        )
