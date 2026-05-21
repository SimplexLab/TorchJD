import torch
from torch import Tensor

from torchjd._linalg import DualConeProjector, projector_or_default
from torchjd.linalg import PSDMatrix

from ._aggregator_bases import GramianWeightedAggregator
from ._mean import MeanWeighting
from ._mixins import _NonDifferentiable
from ._utils.pref_vector import pref_vector_to_str_suffix, pref_vector_to_weighting
from ._weighting_bases import _GramianWeighting


# Non-differentiable: the QP solver operates on numpy arrays, breaking the autograd graph.
class UPGradWeighting(_GramianWeighting, _NonDifferentiable):
    r"""
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.PSDMatrix`]
    giving the weights of :class:`~torchjd.aggregation.UPGrad`.

    :param pref_vector: The preference vector to use. If not provided, defaults to
        :math:`\begin{bmatrix} \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^T \in \mathbb{R}^m`.
    :param projector: The :class:`~torchjd.linalg.DualConeProjector` used to compute the projection.
    """

    def __init__(
        self,
        pref_vector: Tensor | None = None,
        projector: DualConeProjector | None = None,
    ) -> None:
        super().__init__()
        self.pref_vector = pref_vector
        self.projector = projector_or_default(projector)

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        U = torch.diag(self.weighting(gramian))
        W = self.projector(U, gramian)
        return torch.sum(W, dim=0)

    @property
    def pref_vector(self) -> Tensor | None:
        return self._pref_vector

    @pref_vector.setter
    def pref_vector(self, value: Tensor | None) -> None:
        self.weighting = pref_vector_to_weighting(value, default=MeanWeighting())
        self._pref_vector = value

    @property
    def projector(self) -> DualConeProjector:
        return self._projector

    @projector.setter
    def projector(self, value: DualConeProjector | None) -> None:
        self._projector = projector_or_default(value)


class UPGrad(GramianWeightedAggregator, _NonDifferentiable):
    r"""
    :class:`~torchjd.aggregation.GramianWeightedAggregator` that projects each row of the input
    matrix onto the dual cone of all rows of this matrix, and that combines the result, as proposed
    in `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

    :param pref_vector: The preference vector used to combine the projected rows.  If not provided,
        defaults to :math:`\begin{bmatrix} \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^T \in
        \mathbb{R}^m`.
    :param projector: The :class:`~torchjd.linalg.DualConeProjector` used to compute the projection.
    """

    gramian_weighting: UPGradWeighting

    def __init__(
        self,
        pref_vector: Tensor | None = None,
        projector: DualConeProjector | None = None,
    ) -> None:
        super().__init__(
            UPGradWeighting(pref_vector, projector=projector),
        )

    @property
    def pref_vector(self) -> Tensor | None:
        return self.gramian_weighting.pref_vector

    @pref_vector.setter
    def pref_vector(self, value: Tensor | None) -> None:
        self.gramian_weighting.pref_vector = value

    @property
    def projector(self) -> DualConeProjector:
        return self.gramian_weighting.projector

    @projector.setter
    def projector(self, value: DualConeProjector | None) -> None:
        self.gramian_weighting.projector = value

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(pref_vector={repr(self.pref_vector)}, projector="
            f"{repr(self.projector)})"
        )

    def __str__(self) -> str:
        return f"UPGrad{pref_vector_to_str_suffix(self.pref_vector)}"
