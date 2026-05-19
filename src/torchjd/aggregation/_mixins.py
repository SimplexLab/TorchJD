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
        This mixin must appear **before** any :class:`torch.nn.Module` base class in the inheritance
        list. Placing it after will silently have no effect, because :meth:`__call__` would be
        resolved to :class:`torch.nn.Module` before reaching this mixin.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with torch.no_grad():
            return super().__call__(*args, **kwargs)
