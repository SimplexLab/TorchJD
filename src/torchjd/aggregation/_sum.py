import torch
from torch import Tensor

from torchjd._linalg import Structure
from torchjd.aggregation._weighting_bases import FromStructureWeighting

from ._aggregator_bases import WeightedAggregator
from ._weighting_bases import Weighting


class _SumWeighting(Weighting[Structure]):
    def forward(self, structure: Structure, /) -> Tensor:
        weights = torch.ones(structure.m, device=structure.device, dtype=structure.dtype)
        return weights


class SumWeighting(FromStructureWeighting):
    r"""
    :class:`~torchjd.aggregation._weighting_bases.Weighting` that gives the weights
    :math:`\begin{bmatrix} 1 & \dots & 1 \end{bmatrix}^T \in \mathbb{R}^m`.
    """

    def __init__(self) -> None:
        super().__init__(_SumWeighting())


class Sum(WeightedAggregator):
    """
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` that sums of the rows of the input
    matrices.
    """

    weighting: SumWeighting

    def __init__(self) -> None:
        super().__init__(weighting=SumWeighting())
