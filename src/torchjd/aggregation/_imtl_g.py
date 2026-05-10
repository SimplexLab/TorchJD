import torch
from torch import Tensor

from torchjd.linalg import PSDMatrix

from ._aggregator_bases import GramianWeightedAggregator
from ._mixins import _NonDifferentiable
from ._weighting_bases import _GramianWeighting


class IMTLGWeighting(_NonDifferentiable, _GramianWeighting):
    """
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.PSDMatrix`]
    giving the weights of :class:`~torchjd.aggregation.IMTLG`.
    """

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        d = torch.sqrt(torch.diagonal(gramian))
        v = torch.linalg.pinv(gramian) @ d
        v_sum = v.sum()

        weights = torch.zeros_like(v) if v_sum.abs() < 1e-12 else v / v_sum

        return weights


class IMTLG(_NonDifferentiable, GramianWeightedAggregator):
    """
    :class:`~torchjd.aggregation.GramianWeightedAggregator` generalizing the method described in
    `Towards Impartial Multi-task Learning <https://discovery.ucl.ac.uk/id/eprint/10120667/>`_.
    This generalization, defined formally in `Jacobian Descent For Multi-Objective Optimization
    <https://arxiv.org/pdf/2406.16232>`_, supports matrices with some linearly dependent rows.
    """

    gramian_weighting: IMTLGWeighting

    def __init__(self) -> None:
        super().__init__(IMTLGWeighting())
