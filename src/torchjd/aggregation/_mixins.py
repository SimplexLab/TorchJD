from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class Stateful(ABC):
    """Mixin adding a reset method."""

    @abstractmethod
    def reset(self) -> None:
        """Resets the internal state."""


class _NonDifferentiable(nn.Module):
    """
    Mixin making a nn.Module non-differentiable, preventing autograd graph construction by wrapping
    the call in :func:`torch.no_grad`.

    .. warning::
        This mixin must appear **after** the primary base class (e.g.
        :class:`~torchjd.aggregation.Aggregator`,
        :class:`~torchjd.aggregation._weighting_bases._GramianWeighting`) in the inheritance list,
        so that the primary class's :meth:`__call__` is resolved first and its ``super().__call__``
        call chains through this mixin before reaching :class:`torch.nn.Module`. Placing this mixin
        *before* the primary base will cause it to shadow the primary class's :meth:`__call__`
        signature in generated documentation.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with torch.no_grad():
            return super().__call__(*args, **kwargs)
