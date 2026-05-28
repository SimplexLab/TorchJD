from torch import Tensor

from ._scalarizer_base import Scalarizer


class Sum(Scalarizer):
    """
    :class:`~torchjd.scalarization.Scalarizer` that returns the sum of the input tensor of values.
    """

    def forward(self, values: Tensor, /) -> Tensor:
        return values.sum()
