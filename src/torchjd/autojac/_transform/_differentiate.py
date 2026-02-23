from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor

from ._base import RequirementError, TensorDict, Transform
from ._materialize import materialize
from ._ordered_set import OrderedSet


class Differentiate(Transform, ABC):
    """
    Abstract base class for transforms responsible for differentiating some outputs with respect to
    some inputs.

    :param outputs: Tensors to differentiate.
    :param inputs: Tensors with respect to which we differentiate.
    :param retain_graph: If False, the graph used to compute the grads will be freed.
    :param create_graph: If True, graph of the derivative will be constructed, allowing to compute
        higher order derivative products.

    .. note:: The order of outputs and inputs only matters because we have no guarantee that
        torch.autograd.grad is *exactly* equivariant to input permutations and invariant to output
        (with their corresponding grad_output) permutations.
    """

    def __init__(
        self,
        outputs: OrderedSet[Tensor],
        inputs: OrderedSet[Tensor],
        retain_graph: bool,
        create_graph: bool,
    ) -> None:
        self.outputs = list(outputs)
        self.inputs = list(inputs)
        self.retain_graph = retain_graph
        self.create_graph = create_graph

    def __call__(self, tensors: TensorDict, /) -> TensorDict:
        tensor_outputs = [tensors[output] for output in self.outputs]

        differentiated_tuple = self._differentiate(tensor_outputs)
        new_differentiations = dict(zip(self.inputs, differentiated_tuple, strict=True))
        return type(tensors)(new_differentiations)

    @abstractmethod
    def _differentiate(self, tensor_outputs: Sequence[Tensor], /) -> tuple[Tensor, ...]:
        """
        Abstract method for differentiating the outputs with respect to the inputs, and applying the
        linear transformations represented by the tensor_outputs to the results.

        The implementation of this method should define what kind of differentiation is performed:
        whether gradients, Jacobians, etc. are computed, and what the dimension of the
        tensor_outputs should be.
        """

    def check_keys(self, input_keys: set[Tensor], /) -> set[Tensor]:
        outputs = set(self.outputs)
        if not outputs == input_keys:
            raise RequirementError(
                f"The input_keys must match the expected outputs. Found input_keys {input_keys} and"
                f"outputs {outputs}.",
            )
        return set(self.inputs)

    def _get_vjp(self, grad_outputs: Sequence[Tensor], retain_graph: bool) -> tuple[Tensor, ...]:
        optional_grads = torch.autograd.grad(
            self.outputs,
            self.inputs,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            create_graph=self.create_graph,
            allow_unused=True,
        )
        grads = materialize(optional_grads, inputs=self.inputs)
        return grads
