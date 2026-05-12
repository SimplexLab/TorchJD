from torch import Tensor

from torchjd._linalg import DualConeProjector, projector_or_default
from torchjd.linalg import PSDMatrix

from ._aggregator_bases import GramianWeightedAggregator
from ._mean import MeanWeighting
from ._mixins import _NonDifferentiable
from ._utils.pref_vector import pref_vector_to_str_suffix, pref_vector_to_weighting
from ._weighting_bases import _GramianWeighting


# Non-differentiable: the QP solver operates on numpy arrays, breaking the autograd graph.
class DualProjWeighting(_NonDifferentiable, _GramianWeighting):
    r"""
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.PSDMatrix`]
    giving the weights of :class:`~torchjd.aggregation.DualProj`.

    :param pref_vector: The preference vector to use. If not provided, defaults to
        :math:`\begin{bmatrix} \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^T \in \mathbb{R}^m`.
    :param projector: The :class:`~torchjd.linalg.DualConeProjector` used tocompute the projection.
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
        u = self.weighting(gramian)
        w = self.projector(u, gramian)
        return w

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


class DualProj(_NonDifferentiable, GramianWeightedAggregator):
    r"""
    :class:`~torchjd.aggregation.GramianWeightedAggregator` that averages the rows of the input
    matrix, and projects the result onto the dual cone of the rows of the matrix. This corresponds
    to the solution to Equation 11 of `Gradient Episodic Memory for Continual Learning
    <https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf>`_.

    :param pref_vector: The preference vector used to combine the rows. If not provided, defaults to
        :math:`\begin{bmatrix} \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^T \in \mathbb{R}^m`.
    :param projector: The :class:`~torchjd.linalg.DualConeProjector` used tocompute the projection.
    """

    gramian_weighting: DualProjWeighting

    def __init__(
        self,
        pref_vector: Tensor | None = None,
        projector: DualConeProjector | None = None,
    ) -> None:
        super().__init__(
            DualProjWeighting(pref_vector, projector=projector),
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
        return f"DualProj{pref_vector_to_str_suffix(self.pref_vector)}"
