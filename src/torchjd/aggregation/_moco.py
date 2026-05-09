from typing import cast

import torch
from torch import Tensor

from torchjd.aggregation._mixins import Stateful
from torchjd.linalg import Matrix

from ._aggregator_bases import Aggregator
from ._utils.non_differentiable import raise_non_differentiable_error
from ._weighting_bases import _MatrixWeighting


class MoCoWeighting(_MatrixWeighting, Stateful):
    r"""
    :class:`~torchjd.aggregation._mixins.Stateful`
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.Matrix`] giving the weights of
    :class:`~torchjd.aggregation.MoCo`.

    This weighting is stateful: it keeps the moving gradient estimate :math:`Y` and the task weights
    :math:`\lambda` across calls. Use :meth:`reset` between independent runs.

    .. warning::
        MoCo aggregates the moving estimate :math:`Y`, not the current matrix. Therefore, using
        these weights directly on the current matrix does not generally reproduce
        :class:`~torchjd.aggregation.MoCo`.

    :param beta: Learning rate of the moving gradient estimate.
    :param beta_sigma: Decay exponent of ``beta``.
    :param gamma: Learning rate of the task weights.
    :param gamma_sigma: Decay exponent of ``gamma``.
    :param rho: Non-negative :math:`\ell_2` regularization parameter for the task-weight update.
    """

    def __init__(
        self,
        beta: float = 0.5,
        beta_sigma: float = 0.5,
        gamma: float = 0.1,
        gamma_sigma: float = 0.5,
        rho: float = 0.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.beta_sigma = beta_sigma
        self.gamma = gamma
        self.gamma_sigma = gamma_sigma
        self.rho = rho
        self.reset()

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"Attribute `beta` must be non-negative. Found beta={value!r}.")
        self._beta = value

    @property
    def beta_sigma(self) -> float:
        return self._beta_sigma

    @beta_sigma.setter
    def beta_sigma(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(
                f"Attribute `beta_sigma` must be non-negative. Found beta_sigma={value!r}."
            )
        self._beta_sigma = value

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"Attribute `gamma` must be non-negative. Found gamma={value!r}.")
        self._gamma = value

    @property
    def gamma_sigma(self) -> float:
        return self._gamma_sigma

    @gamma_sigma.setter
    def gamma_sigma(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(
                f"Attribute `gamma_sigma` must be non-negative. Found gamma_sigma={value!r}."
            )
        self._gamma_sigma = value

    @property
    def rho(self) -> float:
        return self._rho

    @rho.setter
    def rho(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"Attribute `rho` must be non-negative. Found rho={value!r}.")
        self._rho = value

    def reset(self) -> None:
        """Clears the moving gradient estimate and resets the task weights."""

        self.step = 0
        self._y: Tensor | None = None
        self._lambd: Tensor | None = None
        self._state_key: tuple[int, int, torch.device, torch.dtype] | None = None

    def forward(self, matrix: Matrix, /) -> Tensor:
        if matrix.shape[0] == 0:
            self.reset()
            self._y = matrix.detach().clone()
            self._state_key = (matrix.shape[0], matrix.shape[1], matrix.device, matrix.dtype)
            return matrix.new_empty((0,))

        self._ensure_state(matrix)
        self.step += 1

        y = cast(Tensor, self._y)
        lambd = cast(Tensor, self._lambd)

        beta_step = self.beta / (self.step**self.beta_sigma)
        gamma_step = self.gamma / (self.step**self.gamma_sigma)

        with torch.no_grad():
            y = y - beta_step * (y - matrix.detach())
            yy_t = y @ y.T
            if self.rho != 0.0:
                eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
                yy_t = yy_t + self.rho * eye
            lambd = torch.softmax(lambd - gamma_step * (yy_t @ lambd), dim=-1)

        self._y = y
        self._lambd = lambd

        return lambd

    @property
    def y(self) -> Tensor:
        if self._y is None:
            raise RuntimeError("The moving gradient estimate is not initialized yet.")
        return self._y

    def _ensure_state(self, matrix: Matrix) -> None:
        key = (matrix.shape[0], matrix.shape[1], matrix.device, matrix.dtype)
        if self._state_key == key and self._y is not None and self._lambd is not None:
            return

        self._y = torch.zeros_like(matrix)
        self._lambd = matrix.new_full((matrix.shape[0],), 1.0 / matrix.shape[0])
        self._state_key = key


class MoCo(Aggregator, Stateful):
    r"""
    :class:`~torchjd.aggregation._mixins.Stateful`
    :class:`~torchjd.aggregation.Aggregator` implementing MoCo from `Mitigating Gradient Bias in
    Multi-objective Learning: A Provably Convergent Approach (ICLR 2023)
    <https://openreview.net/forum?id=dLAYGdKTi2>`_.

    This aggregator is stateful: it keeps the moving gradient estimate :math:`Y` and the task
    weights :math:`\lambda` across calls. Use :meth:`reset` between independent runs.

    .. warning::
        The output depends on previously seen matrices. Call :meth:`reset` between independent
        experiments.

    :param beta: Learning rate of the moving gradient estimate.
    :param beta_sigma: Decay exponent of ``beta``.
    :param gamma: Learning rate of the task weights.
    :param gamma_sigma: Decay exponent of ``gamma``.
    :param rho: Non-negative :math:`\ell_2` regularization parameter for the task-weight update.
    """

    weighting: MoCoWeighting

    def __init__(
        self,
        beta: float = 0.5,
        beta_sigma: float = 0.5,
        gamma: float = 0.1,
        gamma_sigma: float = 0.5,
        rho: float = 0.0,
    ) -> None:
        super().__init__()
        self.weighting = MoCoWeighting(
            beta=beta,
            beta_sigma=beta_sigma,
            gamma=gamma,
            gamma_sigma=gamma_sigma,
            rho=rho,
        )
        self.register_full_backward_pre_hook(raise_non_differentiable_error)

    def forward(self, matrix: Matrix, /) -> Tensor:
        weights = self.weighting(matrix)
        if matrix.shape[0] == 0:
            return matrix.sum(dim=0)

        vector = weights @ self.weighting.y
        if matrix.requires_grad:
            vector = vector + 0.0 * matrix.sum(dim=0)
        return vector

    @property
    def beta(self) -> float:
        return self.weighting.beta

    @beta.setter
    def beta(self, value: float) -> None:
        self.weighting.beta = value

    @property
    def beta_sigma(self) -> float:
        return self.weighting.beta_sigma

    @beta_sigma.setter
    def beta_sigma(self, value: float) -> None:
        self.weighting.beta_sigma = value

    @property
    def gamma(self) -> float:
        return self.weighting.gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        self.weighting.gamma = value

    @property
    def gamma_sigma(self) -> float:
        return self.weighting.gamma_sigma

    @gamma_sigma.setter
    def gamma_sigma(self, value: float) -> None:
        self.weighting.gamma_sigma = value

    @property
    def rho(self) -> float:
        return self.weighting.rho

    @rho.setter
    def rho(self, value: float) -> None:
        self.weighting.rho = value

    def reset(self) -> None:
        """Clears the moving gradient estimate and resets the task weights."""

        self.weighting.reset()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(beta={self.beta!r}, beta_sigma={self.beta_sigma!r}, "
            f"gamma={self.gamma!r}, gamma_sigma={self.gamma_sigma!r}, rho={self.rho!r})"
        )
