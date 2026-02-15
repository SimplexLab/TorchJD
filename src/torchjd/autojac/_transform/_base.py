from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeAlias

from torch import Tensor

TensorDict: TypeAlias = dict[Tensor, Tensor]
# Some interesting cases of TensorDict that are worth defining informally (for performance reasons):
# Gradients: A TensorDict in which the shape of each value must be the same as the shape of its
#   corresponding key.
# Jacobians: A TensorDict in which the values must all have the same first dimension and the rest of
#   the shape of each value must be the same as the shape of its corresponding key.
# GradientVectors: A TensorDict containing flattened gradients: the values must be vectors with the
#   same number of elements as their corresponding key.
# JacobianMatrices: A TensorDict containing matrixified (flattened into matrix shape) jacobians: the
#   values must be matrices with a unique first dimension and with a second dimension equal to the
#   number of elements of their corresponding key.


class RequirementError(ValueError):
    """Inappropriate set of inputs keys."""


class Transform(ABC):
    """
    Abstract base class for all transforms. Transforms are elementary building blocks of a jacobian
    descent backward phase. A transform maps a TensorDict to another.
    """

    def compose(self, other: Transform) -> Transform:
        return Composition(self, other)

    def conjunct(self, other: Transform) -> Transform:
        return Conjunction([self, other])

    def __str__(self) -> str:
        return type(self).__name__

    @abstractmethod
    def __call__(self, input: TensorDict, /) -> TensorDict:
        """Applies the transform to the input."""

    @abstractmethod
    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        """
        Checks that the provided input_keys satisfy the transform's requirements and returns the
        corresponding output keys for recursion.

        If the provided input_keys do not satisfy the transform's requirements, raises a
        RequirementError.

        The output keys are the set of keys of the output TensorDict of the transform when the input
        TensorDict's keys are input_keys.
        """

    __lshift__ = compose
    __or__ = conjunct


class Composition(Transform):
    """
    Transform corresponding to the composition of two transforms inner and outer.

    :param inner: The transform to apply first, to the input.
    :param outer: The transform to apply second, to the result of ``inner``.
    """

    def __init__(self, outer: Transform, inner: Transform):
        self.outer = outer
        self.inner = inner

    def __str__(self) -> str:
        return str(self.outer) + " ∘ " + str(self.inner)

    def __call__(self, input: TensorDict, /) -> TensorDict:
        intermediate = self.inner(input)
        return self.outer(intermediate)

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        intermediate_keys = self.inner.check_keys(input_keys)
        output_keys = self.outer.check_keys(intermediate_keys)
        return output_keys


class Conjunction(Transform):
    """
    Transform applying several transforms to the same input, and combining the results (by union)
    into a single TensorDict.

    :param transforms: The transforms to apply. Their outputs should have disjoint sets of keys.
    """

    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = transforms

    def __str__(self) -> str:
        strings = []
        for t in self.transforms:
            s = str(t)
            if isinstance(t, Conjunction):
                strings.append(s[1:-1])  # Remove parentheses
            else:
                strings.append(s)
        return "(" + " | ".join(strings) + ")"

    def __call__(self, tensor_dict: TensorDict, /) -> TensorDict:
        union: TensorDict = {}
        for transform in self.transforms:
            union |= transform(tensor_dict)
        return union

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        output_keys_list = [key for t in self.transforms for key in t.check_keys(input_keys)]
        output_keys = set(output_keys_list)

        if len(output_keys) != len(output_keys_list):
            raise RequirementError("The sets of output keys of transforms should be disjoint.")

        return output_keys
