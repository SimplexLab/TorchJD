from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic, TypeVar

from torch import Tensor, nn

from torchjd.linalg import Matrix, PSDMatrix

_T = TypeVar("_T", contravariant=True, bound=Tensor)
_FnInputT = TypeVar("_FnInputT", bound=Tensor)
_FnOutputT = TypeVar("_FnOutputT", bound=Tensor)


class Weighting(nn.Module, ABC, Generic[_T]):
    r"""
    Abstract base class for all weighting methods. It has the role of extracting a vector of weights
    of dimension :math:`m` from some statistic of a matrix of dimension :math:`m \times n`,
    generally its Gramian, of dimension :math:`m \times m`.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, stat: _T, /) -> Tensor:
        """Computes the vector of weights from the input stat."""

    def __call__(self, stat: Tensor, /) -> Tensor:
        """
        Computes the vector of weights from the input stat and applies all registered hooks.

        :param stat: The stat from which the weights must be extracted.
        """

        return super().__call__(stat)

    def _compose(self, fn: Callable[[_FnInputT], _T]) -> Weighting[_FnInputT]:
        return _Composition(self, fn)

    __lshift__ = _compose


class _Composition(Weighting[_T]):
    """
    Weighting that composes a Weighting with a function, so that the Weighting is applied to the
    output of the function.
    """

    def __init__(self, weighting: Weighting[_FnOutputT], fn: Callable[[_T], _FnOutputT]) -> None:
        super().__init__()
        self.fn = fn
        self.weighting = weighting

    def forward(self, stat: _T, /) -> Tensor:
        return self.weighting(self.fn(stat))


class _MatrixWeighting(Weighting[Matrix]):
    def __call__(self, matrix: Tensor, /) -> Tensor:
        """
        Computes the vector of weights from the input matrix and applies all registered hooks.

        :param matrix: The matrix from which the weights must be extracted.
        """
        return super().__call__(matrix)


class _GramianWeighting(Weighting[PSDMatrix]):
    def __call__(self, gramian: Tensor, /) -> Tensor:
        """
        Computes the vector of weights from the input Gramian and applies all registered hooks.

        :param gramian: The Gramian from which the weights must be extracted.
        """
        return super().__call__(gramian)
