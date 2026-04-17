from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic, TypeVar

from torch import Tensor, nn

from torchjd._linalg import Matrix, PSDTensor, Structure, extract_structure, is_psd_tensor

_T = TypeVar("_T", contravariant=True)
_FnInputT = TypeVar("_FnInputT")
_FnOutputT = TypeVar("_FnOutputT")


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

    def __call__(self, stat: object, /) -> Tensor:
        """Computes the vector of weights from the input stat and applies all registered hooks."""

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


class FromStructureWeighting(_Composition[Matrix]):
    """
    Weighting that extracts the structure of the input matrix before applying a Weighting to it.

    :param structure_weighting: The object responsible for extracting the vector of weights from the
        structure.
    """

    def __init__(self, structure_weighting: Weighting[Structure]) -> None:
        super().__init__(structure_weighting, extract_structure)
        self.structure_weighting = structure_weighting


class FromNothingWeighting(_Composition[Matrix]):
    """
    Weighting that extracts nothing from the input matrix before applying a Weighting to it (i.e. to
    None).

    :param none_weighting: The object responsible for extracting the vector of weights from nothing.
    """

    def __init__(self, none_weighting: Weighting[None]) -> None:
        super().__init__(none_weighting, lambda _: None)
        self.none_weighting = none_weighting


class GeneralizedWeighting(nn.Module, ABC):
    r"""
    Abstract base class for all weightings that operate on generalized Gramians. It has the role of
    extracting a tensor of weights of dimension :math:`m_1 \times \dots \times m_k` from a
    generalized Gramian of dimension
    :math:`m_1 \times \dots \times m_k \times m_k \times \dots \times m_1`.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, generalized_gramian: PSDTensor, /) -> Tensor:
        """Computes the vector of weights from the input generalized Gramian."""

    def __call__(self, generalized_gramian: Tensor, /) -> Tensor:
        """
        Computes the tensor of weights from the input generalized Gramian and applies all registered
        hooks.
        """

        assert is_psd_tensor(generalized_gramian)
        return super().__call__(generalized_gramian)
