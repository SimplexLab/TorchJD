import torch
from torch import Tensor

from torchjd._linalg import Structure
from torchjd.aggregation._weighting_bases import FromStructureWeighting

from ._aggregator_bases import WeightedAggregator
from ._weighting_bases import Weighting


class _MeanWeighting(Weighting[Structure]):
    def forward(self, structure: Structure, /) -> Tensor:
        device = structure.device
        dtype = structure.dtype
        m = structure.m
        weights = torch.full(size=[m], fill_value=1 / m, device=device, dtype=dtype)
        return weights


class MeanWeighting(FromStructureWeighting):
    r"""
    :class:`~torchjd.aggregation._weighting_bases.Weighting` that gives the weights
    :math:`\begin{bmatrix} \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^T \in
    \mathbb{R}^m`.
    """

    def __init__(self) -> None:
        super().__init__(_MeanWeighting())


class Mean(WeightedAggregator):
    """
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` that averages the rows of the input
    matrices.
    """

    weighting: MeanWeighting

    def __init__(self) -> None:
        super().__init__(weighting=MeanWeighting())
