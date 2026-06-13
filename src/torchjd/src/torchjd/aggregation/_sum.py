import torch
from torch import Tensor

from ._aggregator_bases import WeightedAggregator
from ._weighting_bases import _MatrixWeighting


class SumWeighting(_MatrixWeighting):
    r"""
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.Matrix`] that gives the weights
    :math:`\begin{bmatrix} 1 & \dots & 1 \end{bmatrix}^T \in \mathbb{R}^m`.
    """

    def forward(self, matrix: Tensor, /) -> Tensor:
        device = matrix.device
        dtype = matrix.dtype
        weights = torch.ones(matrix.shape[0], device=device, dtype=dtype)
        return weights


class Sum(WeightedAggregator):
    """
    :class:`~torchjd.aggregation.WeightedAggregator` that sums of the rows of the input
    matrices.
    """

    weighting: SumWeighting

    def __init__(self) -> None:
        super().__init__(weighting=SumWeighting())
