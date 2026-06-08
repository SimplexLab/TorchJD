from typing import Any

import torch
from torch import nn

from torchjd._mixins import Stateful as Stateful


class _NonDifferentiable(nn.Module):
    """
    Mixin making a nn.Module non-differentiable, preventing autograd graph construction by wrapping
    the call in :func:`torch.no_grad`.

    .. warning::
        Placing this mixin *before* the primary base will cause it to shadow the primary class's
        :meth:`__call__` signature in generated documentation.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with torch.no_grad():
            return super().__call__(*args, **kwargs)
