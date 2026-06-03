# Partly adapted from https://github.com/heshandevaka/Trade-Off-MOL — MIT License, Copyright (c) 2023 Heshan Fernando.
# See NOTICES for the full license text.
from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from torchjd.aggregation._mixins import Stateful, _NonDifferentiable
from torchjd.linalg import Matrix

from ._weighting_bases import _MatrixWeighting


class MoDoWeighting(_MatrixWeighting, Stateful, _NonDifferentiable):
    r"""
    :class:`~torchjd.aggregation._mixins.Stateful`
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.Matrix`] from `Three-Way
    Trade-Off in Multi-Objective Learning: Optimization, Generalization and Conflict-Avoidance
    <https://www.jmlr.org/papers/volume25/23-1287/23-1287.pdf>`_ (JMLR 2024).

    .. warning::
        The input matrix must be :math:`G = J_1 J_2^\top`, computed from two **independent**
        mini-batches via :func:`torchjd.autojac.jac`. Using a single-batch Gramian
        (:math:`J_1 J_1^\top`) breaks the convergence guarantee. See the usage examples below.

    :param gamma: Learning rate of the task-weight update. Must be positive.
    :param rho: Non-negative :math:`\ell_2` regularisation coefficient.

    .. note::
        The Euclidean projection onto the simplex used in the :math:`\lambda` update is adapted from
        the `official implementation <https://github.com/heshandevaka/Trade-Off-MOL/blob/main/LibMTL/LibMTL/weighting/MoDo.py>`_.

    .. admonition:: Example (two batches per step)

        The following example reproduces basic MoDo using two independent mini-batches per step.
        This matches MoDo as described in the paper, and the behavior of the official
        implementation when ``three_grads`` is ``False``.

        .. testcode::

            import torch
            from torch.nn import Linear, MSELoss, ReLU, Sequential
            from torch.optim import SGD

            from torchjd.aggregation import MoDoWeighting
            from torchjd.autojac import jac

            # Generate data (8 batches of 16 examples of dim 5) for the sake of the example.
            inputs = torch.randn(8, 16, 5)
            targets = torch.randn(8, 16)

            model = Sequential(Linear(5, 4), ReLU(), Linear(4, 1))
            optimizer = SGD(model.parameters())
            criterion = MSELoss(reduction="none")
            weighting = MoDoWeighting(gamma=0.1, rho=0.0)
            params = list(model.parameters())

            # Consume two consecutive (independent) batches per step.
            for i in range(len(inputs) // 2):
                input_1, input_2 = inputs[2 * i], inputs[2 * i + 1]
                target_1, target_2 = targets[2 * i], targets[2 * i + 1]

                # retain_graph=True so both graphs survive for the backward step below.
                losses_1 = criterion(model(input_1).squeeze(dim=1), target_1)
                jacs_1 = jac(losses_1, params, retain_graph=True)
                J_1 = torch.cat([j.flatten(1) for j in jacs_1], dim=1)

                losses_2 = criterion(model(input_2).squeeze(dim=1), target_2)
                jacs_2 = jac(losses_2, params, retain_graph=True)
                J_2 = torch.cat([j.flatten(1) for j in jacs_2], dim=1)

                G = J_1 @ J_2.T
                weights = weighting(G)

                # Equation 2.9b: the parameter update uses the mean of both batches' losses.
                losses = (losses_1 + losses_2) / 2.0
                losses.backward(weights)
                optimizer.step()
                optimizer.zero_grad()

    .. admonition:: Example (three batches per step)

        The following example reproduces basic MoDo using three independent mini-batches per step,
        keeping the :math:`\lambda` update and the parameter update on separate draws. This matches
        the behavior of LibMTL and of the official implementation when ``three_grads`` is ``True``.

        .. testcode::

            import torch
            from torch.nn import Linear, MSELoss, ReLU, Sequential
            from torch.optim import SGD

            from torchjd.aggregation import MoDoWeighting
            from torchjd.autojac import jac

            # Generate data (9 batches of 16 examples of dim 5) for the sake of the example.
            inputs = torch.randn(9, 16, 5)
            targets = torch.randn(9, 16)

            model = Sequential(Linear(5, 4), ReLU(), Linear(4, 1))
            optimizer = SGD(model.parameters())
            criterion = MSELoss(reduction="none")
            weighting = MoDoWeighting(gamma=0.1, rho=0.0)
            params = list(model.parameters())

            # Consume three consecutive (independent) batches per step.
            for i in range(len(inputs) // 3):
                input_1, input_2, input_3 = inputs[3 * i], inputs[3 * i + 1], inputs[3 * i + 2]
                target_1, target_2, target_3 = targets[3 * i], targets[3 * i + 1], targets[3 * i + 2]

                losses_1 = criterion(model(input_1).squeeze(dim=1), target_1)
                jacs_1 = jac(losses_1, params)
                J_1 = torch.cat([j.flatten(1) for j in jacs_1], dim=1)

                losses_2 = criterion(model(input_2).squeeze(dim=1), target_2)
                jacs_2 = jac(losses_2, params)
                J_2 = torch.cat([j.flatten(1) for j in jacs_2], dim=1)

                G = J_1 @ J_2.T
                weights = weighting(G)

                losses_3 = criterion(model(input_3).squeeze(dim=1), target_3)
                losses_3.backward(weights)
                optimizer.step()
                optimizer.zero_grad()
    """

    def __init__(self, gamma: float = 0.1, rho: float = 0.1) -> None:
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

    def forward(self, matrix: Matrix, /) -> Tensor:
        self._ensure_state(matrix)
        lambd = cast(Tensor, self._lambda)

        grad = matrix @ lambd + self._rho * lambd
        lambd = self._projection2simplex(lambd - self._gamma * grad)

        self._lambda = lambd
        return lambd

    @staticmethod
    def _projection2simplex(y: Tensor) -> Tensor:
        """Euclidean projection of ``y`` onto the probability simplex."""

        m = len(y)
        sorted_y = torch.sort(y, descending=True)[0]
        tmpsum = y.new_zeros(())
        tmax_f = (torch.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum = tmpsum + sorted_y[i]
            tmax = (tmpsum - 1.0) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return torch.max(y - tmax_f, y.new_zeros(m))

    def _ensure_state(self, matrix: Matrix) -> None:
        key = (matrix.shape[0], matrix.dtype, matrix.device)
        if self._state_key == key and self._lambda is not None:
            return
        self._lambda = matrix.new_full((matrix.shape[0],), 1.0 / matrix.shape[0])
        self._state_key = key

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gamma={self.gamma!r}, rho={self.rho!r})"
