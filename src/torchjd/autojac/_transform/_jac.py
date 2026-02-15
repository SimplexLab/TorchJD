import math
from collections.abc import Callable, Sequence
from functools import partial

import torch
from torch import Tensor

from ._differentiate import Differentiate
from ._ordered_set import OrderedSet


class Jac(Differentiate):
    """
    Transform from Jacobians to Jacobians computing the jacobian of each output with respect to each
    input, and applying the linear transformations represented by the argument jac_outputs to the
    results.

    :param outputs: Tensors to differentiate.
    :param inputs: Tensors with respect to which we differentiate.
    :param chunk_size: The number of scalars to differentiate simultaneously. If set to ``None``,
        all outputs will be differentiated in parallel at once. If set to ``1``, all will be
        differentiated sequentially. A larger value results in faster differentiation, but also
        higher memory usage. Defaults to ``None``.
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
        chunk_size: int | None,
        retain_graph: bool = False,
        create_graph: bool = False,
    ):
        super().__init__(outputs, inputs, retain_graph, create_graph)
        self.chunk_size = chunk_size

    def _differentiate(self, jac_outputs: Sequence[Tensor], /) -> tuple[Tensor, ...]:
        """
        Computes the jacobian of each output with respect to each input, and applies the linear
        transformations represented by the jac_outputs to the results.

        Returns one jacobian per input. The ith jacobian will be of shape
        ``(m,) + inputs[i].shape``, where ``m`` is the first dimension of the jac_outputs.

        :param jac_outputs: The sequence of tensors indicating how to scale the obtained jacobians.
            Its length should be equal to the length of ``outputs``. ``jac_outputs[i]`` should be of
            shape ``(m,) + outputs[i].shape``, where ``m`` is any value constant over the
            jac_outputs.
        """

        if len(self.inputs) == 0:
            return ()

        if len(self.outputs) == 0:
            return tuple(
                [
                    torch.empty((0, *input.shape), device=input.device, dtype=input.dtype)
                    for input in self.inputs
                ],
            )

        # If the jac_outputs are correct, this value should be the same for all jac_outputs.
        m = jac_outputs[0].shape[0]
        max_chunk_size = self.chunk_size if self.chunk_size is not None else m
        n_chunks = math.ceil(m / max_chunk_size)

        # One tuple per chunk (i), with one value per input (j), of shape [k_i] + shape[j],
        # where k_i is the number of rows in the chunk (the k_i's sum to m)
        jacs_chunks: list[tuple[Tensor, ...]] = []

        # First differentiations: always retain graph
        get_vjp_retain = partial(self._get_vjp, retain_graph=True)
        for i in range(n_chunks - 1):
            start = i * max_chunk_size
            end = (i + 1) * max_chunk_size
            jac_outputs_chunk = [jac_output[start:end] for jac_output in jac_outputs]
            jacs_chunks.append(_get_jacs_chunk(jac_outputs_chunk, get_vjp_retain))

        # Last differentiation: retain the graph only if self.retain_graph==True
        get_vjp_last = partial(self._get_vjp, retain_graph=self.retain_graph)
        start = (n_chunks - 1) * max_chunk_size
        jac_outputs_chunk = [jac_output[start:] for jac_output in jac_outputs]
        jacs_chunks.append(_get_jacs_chunk(jac_outputs_chunk, get_vjp_last))

        n_inputs = len(self.inputs)
        if len(jacs_chunks) == 1:
            # Avoid using cat to avoid doubling memory usage, if it's not needed
            jacs = jacs_chunks[0]
        else:
            jacs = tuple(torch.cat([chunks[i] for chunks in jacs_chunks]) for i in range(n_inputs))

        return jacs


def _get_jacs_chunk(
    jac_outputs_chunk: list[Tensor],
    get_vjp: Callable[[Sequence[Tensor]], tuple[Tensor, ...]],
) -> tuple[Tensor, ...]:
    """
    Computes the jacobian matrix chunk corresponding to the provided get_vjp function, either by
    calling get_vjp directly or by wrapping it into a call to ``torch.vmap``, depending on the shape
    of the provided ``jac_outputs_chunk``. Because of the numerous issues of vmap, we use it only if
    necessary (i.e. when the ``jac_outputs_chunk`` have more than 1 row).
    """

    chunk_size = jac_outputs_chunk[0].shape[0]
    if chunk_size == 1:
        grad_outputs = [tensor.squeeze(0) for tensor in jac_outputs_chunk]
        gradients = get_vjp(grad_outputs)
        return tuple(gradient.unsqueeze(0) for gradient in gradients)
    return torch.vmap(get_vjp, chunk_size=chunk_size)(jac_outputs_chunk)
