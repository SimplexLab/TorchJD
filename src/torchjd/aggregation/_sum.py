import torch
from torch import Tensor

from torchjd._linalg import Matrix

from ._aggregator_bases import WeightedAggregator
from ._weighting_bases import Weighting


class Sum(WeightedAggregator):
    """
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` that sums of the rows of the input
    matrices.
    """

    def __init__(self):
        super().__init__(weighting=SumWeighting())


class SumWeighting(Weighting[Matrix]):
    r"""
    :class:`~torchjd.aggregation._weighting_bases.Weighting` that gives the weights
    :math:`\begin{bmatrix} 1 & \dots & 1 \end{bmatrix}^T \in \mathbb{R}^m`.
    """

    def forward(self, matrix: Tensor, /) -> Tensor:
        device = matrix.device
        dtype = matrix.dtype
        weights = torch.ones(matrix.shape[0], device=device, dtype=dtype)
        return weights
