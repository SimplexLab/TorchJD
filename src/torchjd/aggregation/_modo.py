from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from torchjd.aggregation._mixins import Stateful, _NonDifferentiable
from torchjd.linalg import PSDMatrix

from ._weighting_bases import _GramianWeighting


class MoDoWeighting(_GramianWeighting, Stateful, _NonDifferentiable):
    r"""
    :class:`~torchjd.aggregation._mixins.Stateful`
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.PSDMatrix`] from `Three-Way
    Trade-Off in Multi-Objective Learning: Optimization, Generalization and Conflict-Avoidance
    <https://www.jmlr.org/papers/volume25/23-1287/23-1287.pdf>`_ (JMLR 2024), commonly referred
    to as MoDo (Multi-Objective gradient with Double sampling).

    Given a Gramian :math:`G`, the weights :math:`\lambda` are updated at each call by a
    softmax-projected gradient step:

    .. math::

        \lambda_{t+1} = \operatorname{softmax}\!\bigl(
            \lambda_t - \gamma \cdot (G \lambda_t + \rho \lambda_t)
        \bigr)

    The paper specifies hard simplex projection :math:`\Pi_\Delta`; we follow the `official
    LibMTL implementation <https://github.com/median-research-group/LibMTL>`_ and use
    :func:`torch.softmax` as the projection step.

    The state :math:`\lambda_{t-1}` is initialised lazily to the uniform vector
    :math:`[1/m, \ldots, 1/m]` on the first forward call once :math:`m` is known, and is reset
    automatically when :math:`m`, ``dtype`` or ``device`` of the input Gramian changes. Use
    :meth:`reset` to manually restart from uniform weights.

    .. warning::
        MoDo's convergence guarantees rely on **double sampling**: the Gramian passed to this
        weighting must come from a mini-batch that is independent of the one used for the
        subsequent parameter update. The Gramian can be computed efficiently from a batch of
        losses using the :class:`~torchjd.autogram.Engine`. See the usage example below.

    :param gamma: Learning rate of the task-weight update. Must be positive.
    :param rho: Non-negative :math:`\ell_2` regularisation coefficient.

    .. admonition:: Example

        Train a model using MoDo with two independent mini-batches per step. The first batch
        drives the :math:`\lambda` update via the Gramian; the second batch drives the parameter
        update via the usual backward pass.

        .. code-block:: python

            import torch
            from torch.nn import Linear, MSELoss, ReLU, Sequential
            from torch.optim import SGD

            from torchjd.aggregation import MoDoWeighting
            from torchjd.autogram import Engine

            model = Sequential(Linear(5, 4), ReLU(), Linear(4, 1))
            optimizer = SGD(model.parameters())
            criterion = MSELoss(reduction="none")
            weighting = MoDoWeighting(gamma=0.1, rho=0.0)
            engine = Engine(model, batch_dim=0)

            # loader_1 and loader_2 must yield independent draws from the same distribution.
            for batch_1, batch_2 in zip(loader_1, loader_2):
                input_1, target_1 = batch_1
                input_2, target_2 = batch_2

                # Step 1: Gramian from batch 1 drives the lambda update.
                losses_1 = criterion(model(input_1).squeeze(dim=1), target_1)
                gramian = engine.compute_gramian(losses_1)
                weights = weighting(gramian)

                # Step 2: backward on batch 2 with those weights drives the parameter update.
                losses_2 = criterion(model(input_2).squeeze(dim=1), target_2)
                losses_2.backward(weights)
                optimizer.step()
                optimizer.zero_grad()
    """

    def __init__(self, gamma: float = 0.1, rho: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.rho = rho
        self._lambda: Tensor | None = None
        self._state_key: tuple[int, torch.dtype, torch.device] | None = None

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"Attribute `gamma` must be positive. Found gamma={value!r}.")
        self._gamma = value

    @property
    def rho(self) -> float:
        return self._rho

    @rho.setter
    def rho(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"Attribute `rho` must be non-negative. Found rho={value!r}.")
        self._rho = value

    def reset(self) -> None:
        """Clears the stored task weights so the next forward starts from uniform."""

        self._lambda = None
        self._state_key = None

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        self._ensure_state(gramian)
        lambd = cast(Tensor, self._lambda)

        grad = gramian @ lambd + self._rho * lambd
        lambd = torch.softmax(lambd - self._gamma * grad, dim=-1)

        self._lambda = lambd
        return lambd

    def _ensure_state(self, gramian: PSDMatrix) -> None:
        key = (gramian.shape[0], gramian.dtype, gramian.device)
        if self._state_key == key and self._lambda is not None:
            return
        self._lambda = gramian.new_full((gramian.shape[0],), 1.0 / gramian.shape[0])
        self._state_key = key

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gamma={self.gamma!r}, rho={self.rho!r})"
