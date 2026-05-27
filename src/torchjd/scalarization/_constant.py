from torch import Tensor

from ._scalarizer_base import Scalarizer


class Constant(Scalarizer):
    """
    :class:`~torchjd.scalarization.Scalarizer` that combines the input tensor of losses with
    constant, pre-determined weights.

    :param weights: The weights to apply to the losses. Must have the same shape as the losses
        passed at call time.
    """

    def __init__(self, weights: Tensor) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, losses: Tensor, /) -> Tensor:
        if losses.shape != self.weights.shape:
            raise ValueError(
                f"Parameter `losses` should have shape {tuple(self.weights.shape)} (matching the "
                f"shape of the weights). Found `losses.shape = {tuple(losses.shape)}`.",
            )
        return (self.weights * losses).sum()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weights={repr(self.weights)})"
