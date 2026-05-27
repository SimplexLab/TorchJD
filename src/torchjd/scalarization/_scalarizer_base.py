from abc import ABC, abstractmethod

from torch import Tensor, nn


class Scalarizer(nn.Module, ABC):
    """
    Abstract base class for all scalarizers. Reduces a tensor of losses of any shape into a single
    scalar loss that can be passed to :meth:`~torch.Tensor.backward`.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, losses: Tensor, /) -> Tensor:
        """Computes the scalarization from input tensor."""

    def __call__(self, losses: Tensor, /) -> Tensor:
        """
        Computes the scalar loss from the input tensor of losses and applies all registered hooks.

        :param losses: The tensor of losses to scalarize. May be of any shape.
        """
        return super().__call__(losses)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"
