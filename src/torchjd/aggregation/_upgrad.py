import torch
from torch import Tensor

from torchjd._linalg import PSDMatrix, normalize, regularize

from ._aggregator_bases import GramianWeightedAggregator
from ._mean import MeanWeighting
from ._utils.dual_cone import SUPPORTED_SOLVER, project_weights
from ._utils.non_differentiable import raise_non_differentiable_error
from ._utils.pref_vector import pref_vector_to_str_suffix, pref_vector_to_weighting
from ._weighting_bases import GramianWeighting


class UPGradWeighting(GramianWeighting):
    r"""
    :class:`~torchjd.aggregation._weighting_bases.Weighting` giving the weights of
    :class:`~torchjd.aggregation.UPGrad`.

    :param pref_vector: The preference vector to use. If not provided, defaults to
        :math:`\begin{bmatrix} \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^T \in \mathbb{R}^m`.
    :param norm_eps: A small value to avoid division by zero when normalizing.
    :param reg_eps: A small value to add to the diagonal of the gramian of the matrix. Due to
        numerical errors when computing the gramian, it might not exactly be positive definite.
        This issue can make the optimization fail. Adding ``reg_eps`` to the diagonal of the gramian
        ensures that it is positive definite.
    :param solver: The solver used to optimize the underlying optimization problem.
    """

    def __init__(
        self,
        pref_vector: Tensor | None = None,
        norm_eps: float = 0.0001,
        reg_eps: float = 0.0001,
        solver: SUPPORTED_SOLVER = "quadprog",
    ) -> None:
        super().__init__()
        self.pref_vector = pref_vector
        self.norm_eps = norm_eps
        self.reg_eps = reg_eps
        self.solver: SUPPORTED_SOLVER = solver

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        U = torch.diag(self.weighting(gramian))
        G = regularize(normalize(gramian, self.norm_eps), self.reg_eps)
        W = project_weights(U, G, self.solver)
        return torch.sum(W, dim=0)

    @property
    def pref_vector(self) -> Tensor | None:
        return self._pref_vector

    @pref_vector.setter
    def pref_vector(self, value: Tensor | None) -> None:
        self.weighting = pref_vector_to_weighting(value, default=MeanWeighting())
        self._pref_vector = value

    @property
    def norm_eps(self) -> float:
        return self._norm_eps

    @norm_eps.setter
    def norm_eps(self, value: float) -> None:

        if value < 0:
            raise ValueError(f"norm_eps must be non-negative, but got {value}.")

        self._norm_eps = value

    @property
    def reg_eps(self) -> float:
        return self._reg_eps

    @reg_eps.setter
    def reg_eps(self, value: float) -> None:

        if value < 0:
            raise ValueError(f"reg_eps must be non-negative, but got {value}.")

        self._reg_eps = value


class UPGrad(GramianWeightedAggregator):
    r"""
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` that projects each row of the input
    matrix onto the dual cone of all rows of this matrix, and that combines the result, as proposed
    in `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

    :param pref_vector: The preference vector used to combine the projected rows.  If not provided,
        defaults to :math:`\begin{bmatrix} \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^T \in
        \mathbb{R}^m`.
    :param norm_eps: A small value to avoid division by zero when normalizing.
    :param reg_eps: A small value to add to the diagonal of the gramian of the matrix. Due to
        numerical errors when computing the gramian, it might not exactly be positive definite.
        This issue can make the optimization fail. Adding ``reg_eps`` to the diagonal of the gramian
        ensures that it is positive definite.
    :param solver: The solver used to optimize the underlying optimization problem.
    """

    gramian_weighting: UPGradWeighting

    def __init__(
        self,
        pref_vector: Tensor | None = None,
        norm_eps: float = 0.0001,
        reg_eps: float = 0.0001,
        solver: SUPPORTED_SOLVER = "quadprog",
    ) -> None:
        self._solver: SUPPORTED_SOLVER = solver

        super().__init__(
            UPGradWeighting(pref_vector, norm_eps=norm_eps, reg_eps=reg_eps, solver=solver),
        )

        # This prevents considering the computed weights as constant w.r.t. the matrix.
        self.register_full_backward_pre_hook(raise_non_differentiable_error)

    @property
    def pref_vector(self) -> Tensor | None:
        return self.gramian_weighting.pref_vector

    @pref_vector.setter
    def pref_vector(self, value: Tensor | None) -> None:
        self.gramian_weighting.pref_vector = value

    @property
    def norm_eps(self) -> float:
        return self.gramian_weighting.norm_eps

    @norm_eps.setter
    def norm_eps(self, value: float) -> None:
        self.gramian_weighting.norm_eps = value

    @property
    def reg_eps(self) -> float:
        return self.gramian_weighting.reg_eps

    @reg_eps.setter
    def reg_eps(self, value: float) -> None:
        self.gramian_weighting.reg_eps = value

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(pref_vector={repr(self.pref_vector)}, norm_eps="
            f"{self.norm_eps}, reg_eps={self.reg_eps}, solver={repr(self._solver)})"
        )

    def __str__(self) -> str:
        return f"UPGrad{pref_vector_to_str_suffix(self.pref_vector)}"
