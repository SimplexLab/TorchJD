import torch
from torch import Tensor
from torch.nn.functional import softmax

from ._scalarizer_base import Scalarizer


class Random(Scalarizer):
    """
    :class:`~torchjd.scalarization.Scalarizer` that combines the input tensor of values with
    positive random weights summing to 1, as defined in Algorithm 2 of `Reasonable Effectiveness of
    Random Weighting: A Litmus Test for Multi-Task Learning
    <https://arxiv.org/pdf/2111.10603.pdf>`_.
    """

    def forward(self, values: Tensor, /) -> Tensor:
        flat = torch.randn(values.numel(), device=values.device, dtype=values.dtype)
        weights = softmax(flat, dim=-1).reshape(values.shape)
        return (weights * values).sum()
