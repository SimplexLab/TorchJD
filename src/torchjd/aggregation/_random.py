import torch
from torch import Tensor
from torch.nn import functional as F

from torchjd._linalg import Matrix

from ._aggregator_bases import WeightedAggregator
from ._weighting_bases import Weighting


class Random(WeightedAggregator):
    """
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` that computes a random combination of
    the rows of the provided matrices, as defined in algorithm 2 of `Reasonable Effectiveness of
    Random Weighting: A Litmus Test for Multi-Task Learning
    <https://arxiv.org/pdf/2111.10603.pdf>`_.
    """

    def __init__(self) -> None:
        super().__init__(RandomWeighting())


class RandomWeighting(Weighting[Matrix]):
    """
    :class:`~torchjd.aggregation._weighting_bases.Weighting` that generates positive random weights
    at each call.
    """

    def forward(self, matrix: Tensor, /) -> Tensor:
        random_vector = torch.randn(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        weights = F.softmax(random_vector, dim=-1)
        return weights
