import torch
from torch import Tensor

from ._aggregator_bases import WeightedAggregator
from ._weighting_bases import _MatrixWeighting


class MeanWeighting(_MatrixWeighting):
    r"""
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.Matrix`] that gives the weights
    :math:`\begin{bmatrix} \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^T \in
    \mathbb{R}^m`.
    """

    def forward(self, matrix: Tensor, /) -> Tensor:
        device = matrix.device
        dtype = matrix.dtype
        m = matrix.shape[0]
        weights = torch.full(size=[m], fill_value=1 / m, device=device, dtype=dtype)
        return weights


class Mean(WeightedAggregator):
    """
    :class:`~torchjd.aggregation.WeightedAggregator` that averages the rows of the input
    matrices.
    """

    weighting: MeanWeighting

    def __init__(self) -> None:
        super().__init__(weighting=MeanWeighting())
