from torch import Tensor

from ._scalarizer_base import Scalarizer


class Mean(Scalarizer):
    """
    class:`~torchjd.scalarization.Scalarizer` that returns the mean of the input tensor of losses.
    """

    def forward(self, losses: Tensor, /) -> Tensor:
        return losses.mean()
