from torch import Tensor

from ._scalarizer_base import Scalarizer


class Sum(Scalarizer):
    """
    :class:`~torchjd.scalarization.Scalarizer` that returns the sum of the input tensor of losses.
    """

    def forward(self, losses: Tensor, /) -> Tensor:
        return losses.sum()
