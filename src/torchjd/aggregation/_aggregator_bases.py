from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

from torch import Tensor, nn

from torchjd._linalg import compute_gramian, is_matrix
from torchjd.linalg import Matrix, PSDMatrix

from ._weighting_bases import Weighting


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
    Aggregator that combines the rows of the input Jacobian matrix with weights given by applying a
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.Matrix`] to it.

    :param weighting: The object responsible for extracting the vector of weights from the matrix.
    """

    def __init__(self, weighting: Weighting[Matrix]) -> None:
        super().__init__()
        self.weighting = weighting

    @staticmethod
    def _combine(matrix: Matrix, weights: Tensor) -> Tensor:
        """
        Aggregates a matrix by making a linear combination of its rows, using the provided vector of
        weights.
        """

        vector = weights @ matrix
        return vector

    def forward(self, matrix: Matrix, /) -> Tensor:
        weights = self.weighting(matrix)
        vector = self._combine(matrix, weights)
        return vector


class GramianWeightedAggregator(WeightedAggregator):
    """
    :class:`~torchjd.aggregation.WeightedAggregator` that computes the gramian of the input
    Jacobian matrix before applying a :class:`~torchjd.aggregation.Weighting`
    [:class:`~torchjd.linalg.PSDMatrix`] to it.

    :param gramian_weighting: The object responsible for extracting the vector of weights from the
        gramian.
    """

    def __init__(self, gramian_weighting: Weighting[PSDMatrix]) -> None:
        super().__init__(gramian_weighting << compute_gramian)
        self.gramian_weighting = gramian_weighting


A = TypeVar("A", bound=nn.Module)
F = TypeVar("F", bound=Callable)


class Composition(nn.Module, Generic[A, F]):
    """
    Wraps an aggregator and a function into a composite nn.Module.
    """

    def __init__(self, outer: A, inner: F) -> None:
        super().__init__()
        self.outer = outer
        self.inner = inner

    def forward(self, *args, **kwargs):
        return self.outer(self.inner(*args, **kwargs))

    def __str__(self) -> str:
        return str(self.outer) + " << " + str(self.inner)


def compose(self: A, inner: F) -> Composition[A, F]:
    return Composition(self, inner)


# Bind the << operator to nn.Module
nn.Module.__lshift__ = compose
