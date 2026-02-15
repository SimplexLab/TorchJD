import torch
from torch import Tensor

from torchjd._linalg import Matrix

from ._aggregator_bases import WeightedAggregator
from ._weighting_bases import Weighting


class Mean(WeightedAggregator):
    """
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` that averages the rows of the input
    matrices.
    """

    def __init__(self):
        super().__init__(weighting=MeanWeighting())


class MeanWeighting(Weighting[Matrix]):
    r"""
    :class:`~torchjd.aggregation._weighting_bases.Weighting` that gives the weights
    :math:`\begin{bmatrix} \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^T \in
    \mathbb{R}^m`.
    """

    def forward(self, matrix: Tensor, /) -> Tensor:
        device = matrix.device
        dtype = matrix.dtype
        m = matrix.shape[0]
        weights = torch.full(size=[m], fill_value=1 / m, device=device, dtype=dtype)
        return weights
