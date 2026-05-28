from abc import ABC, abstractmethod

from torch import Tensor, nn


class Scalarizer(nn.Module, ABC):
    """
    Abstract base class for all scalarizers. Reduces a tensor of values of any shape into a single
    scalar value.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, values: Tensor, /) -> Tensor:
        """Computes the scalarization from input tensor."""

    def __call__(self, values: Tensor, /) -> Tensor:
        """
        Computes the scalar value from the input tensor of values and applies all registered hooks.

        :param values: The tensor of values to scalarize. May be of any shape.
        """
        return super().__call__(values)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"
