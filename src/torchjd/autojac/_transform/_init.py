from collections.abc import Set as AbstractSet

import torch
from torch import Tensor

from ._base import RequirementError, TensorDict, Transform


class Init(Transform):
    """
    Transform from {} returning Gradients filled with ones for each of the provided values.

    :param values: Tensors for which Gradients must be returned.
    """

    def __init__(self, values: AbstractSet[Tensor]):
        self.values = values

    def __call__(self, _input: TensorDict, /) -> TensorDict:
        return {value: torch.ones_like(value) for value in self.values}

    def check_keys(self, input_keys: set[Tensor], /) -> set[Tensor]:
        if not input_keys == set():
            raise RequirementError(
                f"The input_keys should be the empty set. Found input_keys {input_keys}.",
            )
        return set(self.values)
