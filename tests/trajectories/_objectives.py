from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import torch
from torch import Tensor


class Objective(ABC):
    def __init__(self, n_params: int, n_values: int) -> None:
        self.n_params = n_params
        self.n_values = n_values

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """Compute the value of the objective function at x. It has to be a vector."""

    @abstractmethod
    def jacobian(self, x: Tensor) -> Tensor:
        """
        Compute the value of the Jacobian of the objective function at x. It is a matrix of shape
        [n_values, n_params].
        """

    def __str__(self) -> str:
        """Return a string representation of the objective function."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.n_values})"


class SPSMapping(ABC):
    """Strong Pareto stationary mapping of an objective."""

    @abstractmethod
    def __call__(self, w: Tensor) -> Tensor:
        """
        Map a vector with (strictly) positive coordinates to the corresponding strongly pareto
        stationary point.
        """


class WithSPSMappingMixin(ABC):
    """Mixin adding the possibility to get the Strong Pareto stationary mapping."""

    @property
    @abstractmethod
    def sps_mapping(self) -> SPSMapping:
        pass


class QuadraticFunction(Objective, WithSPSMappingMixin):
    def __init__(self, As: list[Tensor], us: list[Tensor]) -> None:
        if len(As) != len(us):
            raise ValueError("As and us must have the same length.")

        if len(As) < 1:
            raise ValueError("As and us must have at least one element.")

        super().__init__(n_params=len(us[0]), n_values=len(As))
        # Note that if A is not PSD, the objective is not convex.
        self.As = As
        self.us = us

    def __call__(self, x: Tensor) -> Tensor:
        objective_values = [self.quad(x, A, u) for A, u in zip(self.As, self.us, strict=False)]
        return torch.stack(objective_values)

    def jacobian(self, x: Tensor) -> Tensor:
        return torch.vstack([2 * (x - u) @ A for A, u in zip(self.As, self.us, strict=False)])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(As={self.As}, us={self.us})"

    @staticmethod
    def quad(x: Tensor, A: Tensor, u: Tensor) -> Tensor:
        x_minus_u = x - u
        return x_minus_u @ A @ x_minus_u

    @property
    def sps_mapping(self) -> QuadraticSPSMapping:
        return QuadraticSPSMapping(self.As, self.us)


class QuadraticSPSMapping(SPSMapping):
    def __init__(self, As: list[Tensor], us: list[Tensor]) -> None:
        self.As = As
        self.us = us

    def __call__(self, w: Tensor) -> Tensor:
        G = torch.stack([weight * A for weight, A in zip(w, self.As, strict=False)]).sum(dim=0)
        b = torch.stack(
            [weight * A @ u for weight, A, u in zip(w, self.As, self.us, strict=False)]
        ).sum(dim=0)
        return torch.linalg.lstsq(G, b, driver="gelsd").solution


class HomogenousQuadraticFunction(QuadraticFunction):
    def __init__(self, A: Tensor, scales: Tensor, us: list[Tensor]) -> None:
        self.A = A
        self.scales = scales
        As = [A * scale for scale in scales]
        super().__init__(As=As, us=us)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(A={self.A}, scales={self.scales}, us={self.us})"


class ElementWiseQuadratic(Objective, WithSPSMappingMixin):
    def __init__(self, n_dim: int) -> None:
        super().__init__(n_params=n_dim, n_values=n_dim)

    def __call__(self, x: Tensor) -> Tensor:
        if len(x) != self.n_values:
            raise ValueError("x must have the same length as the number of values.")
        return x**2

    def jacobian(self, x: Tensor) -> Tensor:
        return torch.diag(torch.stack([2 * x[0], 2 * x[1]]))

    @property
    def sps_mapping(self) -> ElementWiseQuadraticSPSMapping:
        return ElementWiseQuadraticSPSMapping(self.n_values)


class ElementWiseQuadraticSPSMapping(SPSMapping):
    def __init__(self, n_values: int) -> None:
        self.n_values = n_values

    def __call__(self, w: Tensor) -> Tensor:  # noqa: ARG002
        return torch.zeros(self.n_values)


class ValleyBowl(Objective):
    """Two-objective toy problem from CAGrad (Liu et al., 2021), also used by Nash-MTL
    (Navon et al., 2022).

    Above :math:`x_2 = 0`, each objective is a log-shaped valley; below it, a quadratic bowl.

    Reference implementations:

    - CAGrad: https://github.com/Cranial-XIX/CAGrad/blob/main/toy.py
    - Nash-MTL: https://github.com/AvivNavon/nash-mtl/tree/main/experiments/toy

    :param scale: Multiplier for the first objective.
    :param scale_both_losses: Multiplier applied to both objectives.
    """

    # Value used for the Pareto set in CAGrad's plotting code (``toy.py``, ``plot_contour``).
    PARETO_SET_X2 = -8.3552

    def __init__(self, scale: float = 1.0, scale_both_losses: float = 1.0) -> None:
        super().__init__(n_params=2, n_values=2)
        self.scale = scale
        self.scale_both_losses = scale_both_losses

    def __call__(self, x: Tensor) -> Tensor:
        if x.shape != (2,):
            raise ValueError("x must have shape (2,).")
        x1, x2 = x.unbind()
        valley1 = (0.5 * (-x1 - 7) - torch.tanh(-x2)).abs().clamp_min(5e-6).log() + 6
        valley2 = (0.5 * (-x1 + 3) + torch.tanh(-x2) + 2).abs().clamp_min(5e-6).log() + 6
        above = torch.tanh(x2 * 0.5).clamp_min(0)
        below = torch.tanh(-x2 * 0.5).clamp_min(0)
        bowl1 = ((-x1 + 7).square() + 0.1 * (-x2 - 8).square()) / 10 - 20
        bowl2 = ((-x1 - 7).square() + 0.1 * (-x2 - 8).square()) / 10 - 20
        return (
            torch.stack(
                ((valley1 * above + bowl1 * below) * self.scale, valley2 * above + bowl2 * below)
            )
            * self.scale_both_losses
        )

    def jacobian(self, x: Tensor) -> Tensor:
        return cast(Tensor, torch.func.jacrev(self)(x))

    def __repr__(self) -> str:
        return f"ValleyBowl(scale={self.scale}, scale_both_losses={self.scale_both_losses})"

    def pareto_set(self, n: int = 1000) -> Tensor:
        """Sample the reference Pareto set used by the Valley-Bowl example.

        :param n: Number of points to sample along the reference set.
        """
        return torch.stack(
            (torch.linspace(-7.0, 7.0, n), torch.full((n,), self.PARETO_SET_X2)), dim=1
        )
