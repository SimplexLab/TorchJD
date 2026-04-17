import torch
from torch import Tensor
from torch.nn import functional as F

from torchjd._linalg import Matrix

from ._aggregator_bases import WeightedAggregator
from ._mixins import Stochastic
from ._weighting_bases import Weighting


class RandomWeighting(Weighting[Matrix], Stochastic):
    """
    :class:`~torchjd.aggregation._weighting_bases.Weighting` that generates positive random weights
    at each call.

    :param seed: Seed for the internal random number generator. If ``None``, a seed is drawn from
        the global PyTorch RNG to fork an independent stream.
    """

    def __init__(self, seed: int | None = None) -> None:
        Weighting.__init__(self)
        Stochastic.__init__(self, seed=seed)

    def forward(self, matrix: Tensor, /) -> Tensor:
        random_vector = torch.randn(
            matrix.shape[0], device=matrix.device, dtype=matrix.dtype, generator=self.generator
        )
        weights = F.softmax(random_vector, dim=-1)
        return weights


class Random(WeightedAggregator, Stochastic):
    """
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` that computes a random combination of
    the rows of the provided matrices, as defined in algorithm 2 of `Reasonable Effectiveness of
    Random Weighting: A Litmus Test for Multi-Task Learning
    <https://arxiv.org/pdf/2111.10603.pdf>`_.

    :param seed: Seed for the internal random number generator. If ``None``, a seed is drawn from
        the global PyTorch RNG to fork an independent stream.
    """

    def __init__(self, seed: int | None = None) -> None:
        weighting = RandomWeighting(seed=seed)
        WeightedAggregator.__init__(self, weighting)
        Stochastic.__init__(self, generator=weighting.generator)
