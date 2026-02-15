from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Iterable, Iterator, MutableSet
from typing import TypeVar

_T = TypeVar("_T", bound=Hashable)


class OrderedSet(MutableSet[_T]):
    """Ordered collection of distinct elements."""

    def __init__(self, elements: Iterable[_T]):
        super().__init__()
        self.ordered_dict = OrderedDict[_T, None]([(element, None) for element in elements])

    def difference_update(self, elements: set[_T]) -> None:
        """Removes all specified elements from the OrderedSet."""

        for element in elements:
            self.discard(element)

    def add(self, value: _T) -> None:
        """Adds the specified element to the OrderedSet."""

        self.ordered_dict[value] = None

    def __add__(self, other: OrderedSet[_T]) -> OrderedSet[_T]:
        """Creates a new OrderedSet with the elements of self followed by the elements of other."""

        return OrderedSet([*self, *other])

    def discard(self, value: _T) -> None:
        if value in self:
            del self.ordered_dict[value]

    def __iter__(self) -> Iterator[_T]:
        return self.ordered_dict.__iter__()

    def __len__(self) -> int:
        return len(self.ordered_dict)

    def __contains__(self, x: object) -> bool:
        return x in self.ordered_dict
