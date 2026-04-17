from typing import cast

import torch
from torch import Tensor

from torchjd._linalg import PSDMatrix

from ._aggregator_bases import GramianWeightedAggregator
from ._mixins import Stochastic
from ._utils.non_differentiable import raise_non_differentiable_error
from ._weighting_bases import Weighting


class PCGrad(GramianWeightedAggregator, Stochastic):
    """
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` as defined in algorithm 1 of
    `Gradient Surgery for Multi-Task Learning <https://arxiv.org/pdf/2001.06782.pdf>`_.

    :param seed: Seed for the internal random number generator. If ``None``, a seed is drawn from
        the global PyTorch RNG to fork an independent stream.
    """

    def __init__(self, seed: int | None = None) -> None:
        weighting = PCGradWeighting(seed=seed)
        GramianWeightedAggregator.__init__(self, weighting)
        Stochastic.__init__(self, generator=weighting.generator)

        # This prevents running into a RuntimeError due to modifying stored tensors in place.
        self.register_full_backward_pre_hook(raise_non_differentiable_error)


class PCGradWeighting(Weighting[PSDMatrix], Stochastic):
    """
    :class:`~torchjd.aggregation._weighting_bases.Weighting` giving the weights of
    :class:`~torchjd.aggregation.PCGrad`.

    :param seed: Seed for the internal random number generator. If ``None``, a seed is drawn from
        the global PyTorch RNG to fork an independent stream.
    """

    def __init__(self, seed: int | None = None) -> None:
        Weighting.__init__(self)
        Stochastic.__init__(self, seed=seed)

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        # Move all computations on cpu to avoid moving memory between cpu and gpu at each iteration
        device = gramian.device
        dtype = gramian.dtype
        cpu = torch.device("cpu")
        gramian = cast(PSDMatrix, gramian.to(device=cpu))

        dimension = gramian.shape[0]
        weights = torch.zeros(dimension, device=cpu, dtype=dtype)

        for i in range(dimension):
            permutation = torch.randperm(dimension, generator=self.generator)
            current_weights = torch.zeros(dimension, device=cpu, dtype=dtype)
            current_weights[i] = 1.0

            for j in permutation:
                if j == i:
                    continue

                # Compute the inner product between g_i^{PC} and g_j
                inner_product = gramian[j] @ current_weights

                if inner_product < 0.0:
                    current_weights[j] -= inner_product / (gramian[j, j])

            weights = weights + current_weights

        return weights.to(device)
