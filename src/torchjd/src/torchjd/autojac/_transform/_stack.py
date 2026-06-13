from collections.abc import Sequence

import torch
from torch import Tensor

from ._base import TensorDict, Transform
from ._materialize import materialize


class Stack(Transform):
    """
    Transform from something to Jacobians, applying several transforms to the same input, and
    combining the results (by stacking) into a single TensorDict.

    The set of keys of the resulting dict is the union of the sets of keys of the input dicts.

    :param transforms: The transforms to apply. They should all be from the same thing and map to
        Gradients. Their outputs may have different sets of keys. If a key is absent in some output
        dicts, the corresponding stacked tensor is filled with zeroes at the positions corresponding
        to those dicts.
    """

    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, input: TensorDict, /) -> TensorDict:
        results = [transform(input) for transform in self.transforms]
        result = _stack(results)
        return result

    def check_keys(self, input_keys: set[Tensor], /) -> set[Tensor]:
        return {key for transform in self.transforms for key in transform.check_keys(input_keys)}


def _stack(gradient_dicts: list[TensorDict]) -> TensorDict:
    # It is important to first remove duplicate keys before computing their associated
    # stacked tensor. Otherwise, some computations would be duplicated. Therefore, we first compute
    # unique_keys, and only then, we compute the stacked tensors.
    union: TensorDict = {}
    for d in gradient_dicts:
        union |= d
    unique_keys = union.keys()
    result = {key: _stack_one_key(gradient_dicts, key) for key in unique_keys}
    return result


def _stack_one_key(gradient_dicts: list[TensorDict], input: Tensor) -> Tensor:
    """Makes the stacked tensor corresponding to a given key, from a list of tensor dicts."""

    optional_gradients = [gradients.get(input, None) for gradients in gradient_dicts]
    gradients = materialize(optional_gradients, [input] * len(optional_gradients))
    jacobian = torch.stack(gradients, dim=0)
    return jacobian
