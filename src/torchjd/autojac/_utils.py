from collections import deque
from collections.abc import Iterable, Sequence, Sized
from typing import cast

from torch import Tensor
from torch.autograd.graph import Node
from torch.overrides import is_tensor_like

from ._transform import OrderedSet


def check_optional_positive_chunk_size(parallel_chunk_size: int | None) -> None:
    if not (parallel_chunk_size is None or parallel_chunk_size > 0):
        raise ValueError(
            "`parallel_chunk_size` should be `None` or greater than `0`. (got "
            f"{parallel_chunk_size})",
        )


def as_checked_ordered_set(
    tensors: Sequence[Tensor] | Tensor,
    variable_name: str,
) -> OrderedSet[Tensor]:
    if is_tensor_like(tensors):
        tensors = (cast(Tensor, tensors),)

    original_length = len(tensors)
    output = OrderedSet(tensors)

    if len(output) != original_length:
        raise ValueError(f"`{variable_name}` should contain unique elements.")

    return OrderedSet(tensors)


def check_matching_length(
    seq1: Sized,
    seq2: Sized,
    variable_name1: str,
    variable_name2: str,
) -> None:
    """
    Checks that two sequences have the same length.

    :param seq1: First sequence to validate.
    :param seq2: Second sequence to validate.
    :param variable_name1: Name of the first variable to include in the error message.
    :param variable_name2: Name of the second variable to include in the error message.
    """
    if len(seq1) != len(seq2):
        raise ValueError(
            f"`{variable_name1}` should have the same length as `{variable_name2}`. "
            f"(got {len(seq1)} and {len(seq2)})",
        )


def check_matching_jac_shapes(
    jacobians: Iterable[Tensor],
    tensors: Iterable[Tensor],
    jacobian_variable_name: str,
    tensor_variable_name: str,
) -> None:
    """
    Checks that the shape of each Jacobian (excluding first dimension) matches the corresponding
    tensor shape.

    :param jacobians: Sequence of Jacobian tensors to validate.
    :param tensors: Sequence of tensors whose shapes should match.
    :param jacobian_variable_name: Name of the Jacobian variable for error messages.
    :param tensor_variable_name: Name of the tensor variable for error messages.
    """
    for i, (jacobian, tensor) in enumerate(zip(jacobians, tensors, strict=True)):
        if jacobian.shape[1:] != tensor.shape:
            raise ValueError(
                f"Shape mismatch: `{jacobian_variable_name}[{i}]` has shape {tuple(jacobian.shape)} "
                f"but `{tensor_variable_name}[{i}]` has shape {tuple(tensor.shape)}. "
                f"The shape of `{jacobian_variable_name}[{i}]` (excluding the first dimension) "
                f"should match the shape of `{tensor_variable_name}[{i}]`.",
            )


def check_matching_grad_shapes(
    gradients: Iterable[Tensor],
    tensors: Iterable[Tensor],
    gradient_variable_name: str,
    tensor_variable_name: str,
) -> None:
    """
    Checks that the shape of each gradient matches the corresponding tensor shape.

    :param gradients: Sequence of gradient tensors to validate.
    :param tensors: Sequence of tensors whose shapes should match.
    :param gradient_variable_name: Name of the gradient variable for error messages.
    :param tensor_variable_name: Name of the tensor variable for error messages.
    """
    for i, (gradient, tensor) in enumerate(zip(gradients, tensors, strict=True)):
        if gradient.shape != tensor.shape:
            raise ValueError(
                f"Shape mismatch: `{gradient_variable_name}[{i}]` has shape {tuple(gradient.shape)} "
                f"but `{tensor_variable_name}[{i}]` has shape {tuple(tensor.shape)}. "
                f"The shape of `{gradient_variable_name}[{i}]` should match the shape of "
                f"`{tensor_variable_name}[{i}]`.",
            )


def check_consistent_first_dimension(
    jacobians: Sequence[Tensor],
    variable_name: str,
) -> None:
    """
    Checks that all Jacobians have the same first dimension (number of rows).

    :param jacobians: Sequence of Jacobian tensors to validate.
    :param variable_name: Name of the variable to include in the error message.
    """
    if len(jacobians) > 0 and not all(
        jacobian.shape[0] == jacobians[0].shape[0] for jacobian in jacobians[1:]
    ):
        raise ValueError(f"All Jacobians in `{variable_name}` should have the same number of rows.")


def get_leaf_tensors(tensors: Iterable[Tensor], excluded: Iterable[Tensor]) -> OrderedSet[Tensor]:
    """
    Gets the leaves of the autograd graph of all specified ``tensors``.

    :param tensors: Tensors from which the graph traversal should start. They should all require
        grad and not be leaves.
    :param excluded: Tensors whose grad_fn should be excluded from the graph traversal. They should
        all require grad and not be leaves.

    """

    if any(tensor.grad_fn is None for tensor in tensors):
        raise ValueError("All `tensors` should have a `grad_fn`.")

    if any(tensor.grad_fn is None for tensor in excluded):
        raise ValueError("All `excluded` tensors should have a `grad_fn`.")

    accumulate_grads = _get_descendant_accumulate_grads(
        roots=cast(OrderedSet[Node], OrderedSet([tensor.grad_fn for tensor in tensors])),
        excluded_nodes=cast(set[Node], {tensor.grad_fn for tensor in excluded}),
    )

    # accumulate_grads contains instances of AccumulateGrad, which contain a `variable` field.
    # They cannot be typed as such because AccumulateGrad is not public.
    leaves = OrderedSet([g.variable for g in accumulate_grads])  # type: ignore[attr-defined]

    return leaves


def _get_descendant_accumulate_grads(
    roots: OrderedSet[Node],
    excluded_nodes: set[Node],
) -> OrderedSet[Node]:
    """
    Gets the AccumulateGrad descendants of the specified nodes.

    :param roots: Root nodes from which the graph traversal should start.
    :param excluded_nodes: Nodes excluded from the graph traversal.
    """

    excluded_nodes = set(excluded_nodes)  # Re-instantiate set to avoid modifying input
    result: OrderedSet[Node] = OrderedSet([])
    roots.difference_update(excluded_nodes)
    nodes_to_traverse = deque(roots)

    # This implementation more or less follows what is advised in
    # https://discuss.pytorch.org/t/autograd-graph-traversal/213658 and what was suggested in
    # https://github.com/SimplexLab/TorchJD/issues/216.
    while nodes_to_traverse:
        node = nodes_to_traverse.popleft()  # Breadth-first

        if node.__class__.__name__ == "AccumulateGrad":
            result.add(node)

        for child, _ in node.next_functions:
            if child is not None and child not in excluded_nodes:
                nodes_to_traverse.append(child)  # Append to the right
                excluded_nodes.add(child)

    return result
