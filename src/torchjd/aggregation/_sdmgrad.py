# Partly adapted from https://github.com/OptMN-Lab/SDMGrad — MIT License, Copyright (c) 2023 ml-opt-lab.
# See NOTICES for the full license text.
from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from torchjd._mixins import Stateful
from torchjd.aggregation._mixins import _NonDifferentiable
from torchjd.linalg import Matrix

from ._weighting_bases import _MatrixWeighting


class SDMGradWeighting(_MatrixWeighting, Stateful, _NonDifferentiable):
    r"""
    :class:`~torchjd.Stateful`
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.Matrix`] from `Direction-oriented
    Multi-objective Learning: Simple and Provable Stochastic Algorithms
    <https://arxiv.org/pdf/2305.18409>`_ (NeurIPS 2023).

    .. warning::
        The input matrix must be :math:`A = J_1 J_2^\top`, computed from two **independent**
        mini-batches via :func:`torchjd.autojac.jac`. It is **not** a Gramian and is not symmetric
        or positive semi-definite in general. See the usage examples below.

    :param lr: Learning rate of the inner SGD that solves for the task weights. Must be positive.
    :param momentum: Momentum of the inner SGD. Must be in :math:`[0, 1)`.
    :param n_iter: Number of inner SGD iterations performed at each call. Must be positive.
    :param lamda: Non-negative coefficient controlling how strongly the descent direction is pulled
        toward the preference direction. Must be non-negative.
    :param pref_vector: The preference vector :math:`\tilde w` defining the target direction. If not
        provided, defaults to the uniform vector :math:`[1/m, \ldots, 1/m]` (i.e. the average
        gradient).

    .. note::
        The inner simplex-projected solve is adapted from the `official implementation
        <https://github.com/OptMN-Lab/SDMGrad/blob/main/methods/weight_methods.py>`_. Note that the
        official class default ``lamda=0.6`` is overridden to ``0.3`` in their own experiments, which
        is the value used here (and in `LibMTL
        <https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/SDMGrad.py>`_).

    .. admonition:: Example (two batches per step)

        The following example reproduces SDMGrad using two independent mini-batches per step, reusing
        the second batch for the parameter update.

        .. testcode::

            import torch
            from torch.nn import Linear, MSELoss, ReLU, Sequential
            from torch.optim import SGD

            from torchjd.aggregation import SDMGradWeighting
            from torchjd.autojac import jac

            # Generate data (8 batches of 16 examples of dim 5) for the sake of the example.
            inputs = torch.randn(8, 16, 5)
            targets = torch.randn(8, 16)

            model = Sequential(Linear(5, 4), ReLU(), Linear(4, 1))
            optimizer = SGD(model.parameters())
            criterion = MSELoss(reduction="none")
            weighting = SDMGradWeighting(lamda=0.3)
            params = list(model.parameters())

            # Consume two consecutive (independent) batches per step.
            for i in range(len(inputs) // 2):
                input_1, input_2 = inputs[2 * i], inputs[2 * i + 1]
                target_1, target_2 = targets[2 * i], targets[2 * i + 1]

                losses_1 = criterion(model(input_1).squeeze(dim=1), target_1)
                jacs_1 = jac(losses_1, params)
                J_1 = torch.cat([j.flatten(1) for j in jacs_1], dim=1)

                # retain_graph=True so losses_2's graph survives for the backward step below.
                losses_2 = criterion(model(input_2).squeeze(dim=1), target_2)
                jacs_2 = jac(losses_2, params, retain_graph=True)
                J_2 = torch.cat([j.flatten(1) for j in jacs_2], dim=1)

                A = J_1 @ J_2.T
                weights = weighting(A)

                losses_2.backward(weights)
                optimizer.step()
                optimizer.zero_grad()

    .. admonition:: Example (three batches per step)

        The following example reproduces SDMGrad using three independent mini-batches per step,
        keeping the weight update and the parameter update on separate draws.

        .. testcode::

            import torch
            from torch.nn import Linear, MSELoss, ReLU, Sequential
            from torch.optim import SGD

            from torchjd.aggregation import SDMGradWeighting
            from torchjd.autojac import jac

            # Generate data (9 batches of 16 examples of dim 5) for the sake of the example.
            inputs = torch.randn(9, 16, 5)
            targets = torch.randn(9, 16)

            model = Sequential(Linear(5, 4), ReLU(), Linear(4, 1))
            optimizer = SGD(model.parameters())
            criterion = MSELoss(reduction="none")
            weighting = SDMGradWeighting(lamda=0.3)
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

                A = J_1 @ J_2.T
                weights = weighting(A)

                losses_3 = criterion(model(input_3).squeeze(dim=1), target_3)
                losses_3.backward(weights)
                optimizer.step()
                optimizer.zero_grad()
    """

    def __init__(
        self,
        lr: float = 10.0,
        momentum: float = 0.5,
        n_iter: int = 20,
        lamda: float = 0.3,
        pref_vector: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.n_iter = n_iter
        self.lamda = lamda
        self.pref_vector = pref_vector
        self._w: Tensor | None = None
        self._state_key: tuple[int, torch.dtype, torch.device] | None = None

    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"Attribute `lr` must be positive. Found lr={value!r}.")
        self._lr = value

    @property
    def momentum(self) -> float:
        return self._momentum

    @momentum.setter
    def momentum(self, value: float) -> None:
        if not (0.0 <= value < 1.0):
            raise ValueError(f"Attribute `momentum` must be in [0, 1). Found momentum={value!r}.")
        self._momentum = value

    @property
    def n_iter(self) -> int:
        return self._n_iter

    @n_iter.setter
    def n_iter(self, value: int) -> None:
        if value < 1:
            raise ValueError(
                f"Attribute `n_iter` must be a positive integer. Found n_iter={value!r}."
            )
        self._n_iter = value

    @property
    def lamda(self) -> float:
        return self._lamda

    @lamda.setter
    def lamda(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"Attribute `lamda` must be non-negative. Found lamda={value!r}.")
        self._lamda = value

    @property
    def pref_vector(self) -> Tensor | None:
        return self._pref_vector

    @pref_vector.setter
    def pref_vector(self, value: Tensor | None) -> None:
        if value is not None and value.ndim != 1:
            raise ValueError(
                "Parameter `pref_vector` must be a vector (1D Tensor). Found `pref_vector.ndim = "
                f"{value.ndim}`."
            )
        self._pref_vector = value

    def reset(self) -> None:
        """Clears the stored task weights so the next forward starts from uniform."""

        self._w = None
        self._state_key = None

    def forward(self, matrix: Matrix, /) -> Tensor:
        self._ensure_state(matrix)
        w = cast(Tensor, self._w)
        w_tilde = self._resolve_w_tilde(matrix)

        velocity: Tensor | None = None
        for _ in range(self._n_iter):
            grad = matrix @ (w + self._lamda * w_tilde)
            velocity = grad if velocity is None else self._momentum * velocity + grad
            w = self._projection2simplex(w - self._lr * velocity)

        self._w = w
        return (w + self._lamda * w_tilde) / (1.0 + self._lamda)

    def _resolve_w_tilde(self, matrix: Matrix) -> Tensor:
        m = matrix.shape[0]
        if self._pref_vector is None:
            return matrix.new_full((m,), 1.0 / m)
        if self._pref_vector.shape[0] != m:
            raise ValueError(
                "The length of `pref_vector` must match the number of rows of the input matrix. "
                f"Found len(pref_vector)={self._pref_vector.shape[0]} and matrix.shape[0]={m}."
            )
        return self._pref_vector.to(dtype=matrix.dtype, device=matrix.device)

    def _ensure_state(self, matrix: Matrix) -> None:
        key = (matrix.shape[0], matrix.dtype, matrix.device)
        if self._state_key == key and self._w is not None:
            return
        self._w = matrix.new_full((matrix.shape[0],), 1.0 / matrix.shape[0])
        self._state_key = key

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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(lr={self.lr!r}, momentum={self.momentum!r}, "
            f"n_iter={self.n_iter!r}, lamda={self.lamda!r}, pref_vector={self.pref_vector!r})"
        )
