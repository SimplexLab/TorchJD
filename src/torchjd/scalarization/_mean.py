from torch import Tensor

from ._scalarizer_base import Scalarizer


class Mean(Scalarizer):
    """
    :class:`~torchjd.scalarization.Scalarizer` that returns the mean of the input tensor of values.
    """

    def forward(self, values: Tensor, /) -> Tensor:
        return values.mean()
