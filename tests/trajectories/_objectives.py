from abc import ABC, abstractmethod

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


class WithSPSMappingMixin(ABC):
    """Mixin adding the possibility to get the Strong Pareto stationary mapping."""

    class SPSMapping(ABC):
        @abstractmethod
        def __call__(self, w: Tensor) -> Tensor:
            """
            Map a vector with (strictly) positive coordinates to the corresponding strongly pareto
            stationary point.
            """

    @property
    @abstractmethod
    def sps_mapping(self) -> "WithSPSMappingMixin.SPSMapping":
        pass


class QuadraticForm(Objective, WithSPSMappingMixin):
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

    class SPSMapping(WithSPSMappingMixin.SPSMapping):
        def __init__(self, As: list[Tensor], us: list[Tensor]) -> None:
            self.As = As
            self.us = us

        def __call__(self, w: Tensor) -> Tensor:
            G = torch.stack([weight * A for weight, A in zip(w, self.As, strict=False)]).sum(dim=0)
            b = torch.stack(
                [weight * A @ u for weight, A, u in zip(w, self.As, self.us, strict=False)]
            ).sum(dim=0)
            return torch.linalg.lstsq(G, b, driver="gelsd").solution

    @property
    def sps_mapping(self) -> "QuadraticForm.SPSMapping":
        return self.SPSMapping(self.As, self.us)


class HomogenousQuadraticForm(QuadraticForm):
    def __init__(self, A: Tensor, scales: Tensor, us: list[Tensor]) -> None:
        self.A = A
        self.scales = scales
        As = [A * scale for scale in scales]
        super().__init__(As=As, us=us)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(A={self.A}, scales={self.scales}, us={self.us})"


class ConvexQuadraticForm(QuadraticForm):
    def __init__(self, Bs: list[Tensor], us: list[Tensor]) -> None:
        self.Bs = Bs
        super().__init__(As=[B @ B.T for B in self.Bs], us=us)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(Bs={self.Bs}, us={self.us})"


class ElementWiseQuadratic(Objective, WithSPSMappingMixin):
    def __init__(self, n_dim: int) -> None:
        super().__init__(n_params=n_dim, n_values=n_dim)

    def __call__(self, x: Tensor) -> Tensor:
        if len(x) != self.n_values:
            raise ValueError("x must have the same length as the number of values.")
        return x**2

    def jacobian(self, x: Tensor) -> Tensor:
        return torch.diag(torch.stack([2 * x[0], 2 * x[1]]))

    class SPSMapping(WithSPSMappingMixin.SPSMapping):
        def __init__(self, n_values: int) -> None:
            self.n_values = n_values

        def __call__(self, w: Tensor) -> Tensor:  # noqa: ARG002
            return torch.zeros(self.n_values)

    @property
    def sps_mapping(self) -> "ElementWiseQuadratic.SPSMapping":
        return self.SPSMapping(self.n_values)


class Multinorm(Objective, WithSPSMappingMixin):
    def __init__(self, a: Tensor) -> None:
        n = len(a)
        super().__init__(n_params=n, n_values=n)
        self.a = a

    def __call__(self, x: Tensor) -> Tensor:
        if len(x) != self.n_values:
            raise ValueError("x must have the same length as the number of values.")

        # f_i(x) = a_i * || x - a_i * e_i  ||²
        return self.a * torch.norm(x.expand(len(x), len(x)) - torch.diag(self.a), dim=1) ** 2

    def jacobian(self, x: Tensor) -> Tensor:
        return self.a * 2 * (x.expand(len(x), len(x)) - torch.diag(self.a))

    class SPSMapping(WithSPSMappingMixin.SPSMapping):
        def __init__(self, a: Tensor) -> None:
            self.a = a

        def __call__(self, w: Tensor) -> Tensor:
            return w * self.a

    @property
    def sps_mapping(self) -> "Multinorm.SPSMapping":
        return self.SPSMapping(self.a)
