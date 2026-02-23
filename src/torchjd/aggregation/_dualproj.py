from torch import Tensor

from torchjd._linalg import PSDMatrix, normalize, regularize

from ._aggregator_bases import GramianWeightedAggregator
from ._mean import MeanWeighting
from ._utils.dual_cone import project_weights
from ._utils.non_differentiable import raise_non_differentiable_error
from ._utils.pref_vector import pref_vector_to_str_suffix, pref_vector_to_weighting
from ._weighting_bases import Weighting


class DualProj(GramianWeightedAggregator):
    r"""
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` that averages the rows of the input
    matrix, and projects the result onto the dual cone of the rows of the matrix. This corresponds
    to the solution to Equation 11 of `Gradient Episodic Memory for Continual Learning
    <https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf>`_.

    :param pref_vector: The preference vector used to combine the rows. If not provided, defaults to
        :math:`\begin{bmatrix} \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^T \in \mathbb{R}^m`.
    :param norm_eps: A small value to avoid division by zero when normalizing.
    """

    def __init__(
        self,
        pref_vector: Tensor | None = None,
        norm_eps: float = 0.0001,
    ) -> None:
        self._pref_vector = pref_vector
        self._norm_eps = norm_eps

        super().__init__(
            DualProjWeighting(pref_vector, norm_eps=norm_eps),
        )

        # This prevents considering the computed weights as constant w.r.t. the matrix.
        self.register_full_backward_pre_hook(raise_non_differentiable_error)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(pref_vector={repr(self._pref_vector)}, "
            f"norm_eps={self._norm_eps})"
        )

    def __str__(self) -> str:
        return f"DualProj{pref_vector_to_str_suffix(self._pref_vector)}"


class DualProjWeighting(Weighting[PSDMatrix]):
    r"""
    :class:`~torchjd.aggregation._weighting_bases.Weighting` giving the weights of
    :class:`~torchjd.aggregation.DualProj`.

    :param pref_vector: The preference vector to use. If not provided, defaults to
        :math:`\begin{bmatrix} \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^T \in \mathbb{R}^m`.
    :param norm_eps: A small value to avoid division by zero when normalizing.
    """

    def __init__(
        self,
        pref_vector: Tensor | None = None,
        norm_eps: float = 0.0001,
    ) -> None:
        super().__init__()
        self._pref_vector = pref_vector
        self.weighting = pref_vector_to_weighting(pref_vector, default=MeanWeighting())
        self.norm_eps = norm_eps

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        u = self.weighting(gramian)
        G = normalize(gramian, self.norm_eps)
        if self.norm_eps > 0:
            G = regularize(G, 1e-4)
        return project_weights(u, G)
