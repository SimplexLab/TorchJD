from collections.abc import Sequence

import torch
from torch import Tensor

from ._differentiate import Differentiate
from ._ordered_set import OrderedSet


class Grad(Differentiate):
    """
    Transform from Gradients to Gradients, computing the gradient of each output element with
    respect to each input tensor, and applying the linear transformations represented by provided
    the grad_outputs to the results.

    :param outputs: Tensors to differentiate.
    :param inputs: Tensors with respect to which we differentiate.
    :param retain_graph: If False, the graph used to compute the grads will be freed. Defaults to
        False.
    :param create_graph: If True, graph of the derivative will be constructed, allowing to compute
        higher order derivative products. Defaults to False.

    .. note:: The order of outputs and inputs only matters because we have no guarantee that
        torch.autograd.grad is *exactly* equivariant to input permutations and invariant to output
        (with their corresponding grad_output) permutations.
    """

    def __init__(
        self,
        outputs: OrderedSet[Tensor],
        inputs: OrderedSet[Tensor],
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> None:
        super().__init__(outputs, inputs, retain_graph, create_graph)

    def _differentiate(self, grad_outputs: Sequence[Tensor], /) -> tuple[Tensor, ...]:
        """
        Computes the gradient of each output element with respect to each input tensor, and applies
        the linear transformations represented by the grad_outputs to the results.

        Returns one gradient per input, corresponding to the sum of the scaled gradients with
        respect to this input.

        :param grad_outputs: The sequence of tensors to scale the obtained gradients with. Its
            length should be equal to the length of ``outputs``. Each grad_output should have the
            same shape as the corresponding output.
        """

        if len(self.inputs) == 0:
            return ()

        if len(self.outputs) == 0:
            return tuple(torch.zeros_like(input) for input in self.inputs)

        grads = self._get_vjp(grad_outputs, self.retain_graph)
        return grads
