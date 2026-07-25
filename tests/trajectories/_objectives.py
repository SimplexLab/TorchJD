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
    def sps_mapping(self) -> "QuadraticFunction.SPSMapping":
        return self.SPSMapping(self.As, self.us)


class HomogenousQuadraticFunction(QuadraticFunction):
    def __init__(self, A: Tensor, scales: Tensor, us: list[Tensor]) -> None:
        self.A = A
        self.scales = scales
        As = [A * scale for scale in scales]
        super().__init__(As=As, us=us)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(A={self.A}, scales={self.scales}, us={self.us})"


class PowerNormFunction(Objective, WithSPSMappingMixin):
    """
    Objective whose values are powers of distances to fixed points:
    f_i(x) = scales_i * ||x - us_i||^powers_i.

    Each f_i is convex, and non-quadratic whenever powers_i != 2.

    :param powers: The exponents. Must all be >= 2 for the objective to be smooth.
    :param scales: The positive factors multiplying each powered distance.
    :param us: The points from which the distances are computed.
    """

    def __init__(self, powers: Tensor, scales: Tensor, us: list[Tensor]) -> None:
        if not (len(powers) == len(scales) == len(us)):
            raise ValueError("powers, scales and us must have the same length.")

        if bool((powers < 2.0).any()):
            raise ValueError("powers must all be >= 2 for the objective to be smooth.")

        super().__init__(n_params=len(us[0]), n_values=len(us))
        self.powers = powers
        self.scales = scales
        self.us = us

    def __call__(self, x: Tensor) -> Tensor:
        objective_values = [
            s * (x - u).dot(x - u) ** (p / 2)
            for p, s, u in zip(self.powers, self.scales, self.us, strict=False)
        ]
        return torch.stack(objective_values)

    def jacobian(self, x: Tensor) -> Tensor:
        return torch.vstack(
            [
                s * p * (x - u).dot(x - u) ** ((p - 2) / 2) * (x - u)
                for p, s, u in zip(self.powers, self.scales, self.us, strict=False)
            ]
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(powers={self.powers}, scales={self.scales}, us={self.us})"
        )

    class SPSMapping(WithSPSMappingMixin.SPSMapping):
        """
        SPS mapping for a pair of powered distance functions. The Pareto set is the segment between
        the two points, so the weighted gradient balance equation is solved by bisection along it.
        """

        def __init__(self, powers: Tensor, scales: Tensor, us: list[Tensor]) -> None:
            if len(us) != 2:
                raise ValueError("SPSMapping is only defined for objectives with 2 values.")
            self.powers = powers.to(dtype=torch.float64)
            self.scales = scales.to(dtype=torch.float64)
            self.us = [u.to(dtype=torch.float64) for u in us]

        def __call__(self, w: Tensor) -> Tensor:
            distance = (self.us[1] - self.us[0]).norm()

            def gradient_norm_imbalance(t: Tensor) -> Tensor:
                norms = []
                for i, d in enumerate([t * distance, (1 - t) * distance]):
                    norms.append(w[i] * self.scales[i] * self.powers[i] * d ** (self.powers[i] - 1))
                return norms[0] - norms[1]

            low = torch.tensor(0.0, dtype=torch.float64)
            high = torch.tensor(1.0, dtype=torch.float64)
            for _ in range(100):
                mid = (low + high) / 2
                if gradient_norm_imbalance(mid) < 0.0:
                    low = mid
                else:
                    high = mid
            t = (low + high) / 2
            return self.us[0] + t * (self.us[1] - self.us[0])

    @property
    def sps_mapping(self) -> "PowerNormFunction.SPSMapping":
        return self.SPSMapping(self.powers, self.scales, self.us)


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
