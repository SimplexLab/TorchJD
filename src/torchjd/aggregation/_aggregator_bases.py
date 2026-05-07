from abc import ABC, abstractmethod
from typing import cast

from torch import Tensor, nn

from torchjd._linalg import Matrix, compute_gramian, is_matrix

from ._weighting_bases import GramianWeighting, MatrixWeighting


class Aggregator(nn.Module, ABC):
    r"""
    Abstract base class for all aggregators. It has the role of aggregating matrices of dimension
    :math:`m \times n` into row vectors of dimension :math:`n`.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _check_is_matrix(matrix: Tensor) -> None:
        if not is_matrix(matrix):
            raise ValueError(
                "Parameter `matrix` should be a tensor of dimension 2. Found `matrix.shape = "
                f"{matrix.shape}`.",
            )

    @abstractmethod
    def forward(self, matrix: Matrix, /) -> Tensor:
        """Computes the aggregation from the input matrix."""

    def __call__(self, matrix: Tensor, /) -> Tensor:
        """
        Computes the aggregation from the input matrix and applies all registered hooks.

        :param matrix: The Jacobian to aggregate.
        """
        Aggregator._check_is_matrix(matrix)
        return super().__call__(matrix)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


class WeightedAggregator(Aggregator):
    """
    Aggregator that combines the rows of the input jacobian matrix with weights given by applying a
    :class:`~torchjd.aggregation.MatrixWeighting` to it.

    :param weighting: The object responsible for extracting the vector of weights from the matrix.
    """

    def __init__(self, weighting: MatrixWeighting) -> None:
        super().__init__()
        self.weighting = weighting

    @staticmethod
    def combine(matrix: Matrix, weights: Tensor) -> Tensor:
        """
        Aggregates a matrix by making a linear combination of its rows, using the provided vector of
        weights.
        """

        vector = weights @ matrix
        return vector

    def forward(self, matrix: Matrix, /) -> Tensor:
        weights = self.weighting(matrix)
        vector = self.combine(matrix, weights)
        return vector


class GramianWeightedAggregator(WeightedAggregator):
    """
    :class:`~torchjd.aggregation.WeightedAggregator` that computes the gramian of the input
    jacobian matrix before applying a :class:`~torchjd.aggregation.GramianWeighting` to it.

    :param gramian_weighting: The object responsible for extracting the vector of weights from the
        gramian.
    """

    def __init__(self, gramian_weighting: GramianWeighting) -> None:
        super().__init__(cast(MatrixWeighting, gramian_weighting << compute_gramian))
        self.gramian_weighting = gramian_weighting
