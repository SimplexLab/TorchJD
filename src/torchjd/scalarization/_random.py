import torch
from torch import Tensor
from torch.nn import functional as F

from ._scalarizer_base import Scalarizer


class Random(Scalarizer):
    """
    class:`~torchjd.scalarization.Scalarizer` that combines the input tensor of losses with
    positive random weights summing to 1, as defined in Algorithm 2 of `Reasonable Effectiveness of
    Random Weighting: A Litmus Test for Multi-Task Learning
    <https://arxiv.org/pdf/2111.10603.pdf>`_.
    """

    def forward(self, losses: Tensor, /) -> Tensor:
        flat = torch.randn(losses.numel(), device=losses.device, dtype=losses.dtype)
        weights = F.softmax(flat, dim=-1).reshape(losses.shape)
        return (weights * losses).sum()
