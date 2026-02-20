from collections.abc import Sequence
from typing import cast

from torch import Tensor
from torch.overrides import is_tensor_like

from torchjd.autojac._transform._base import Transform
from torchjd.autojac._transform._diagonalize import Diagonalize
from torchjd.autojac._transform._init import Init
from torchjd.autojac._transform._jac import Jac
from torchjd.autojac._transform._ordered_set import OrderedSet
from torchjd.autojac._utils import (
    as_checked_ordered_set,
    check_consistent_first_dimension,
    check_matching_jac_shapes,
    check_matching_length,
    check_optional_positive_chunk_size,
)


def jac(
    outputs: Sequence[Tensor] | Tensor,
    inputs: Sequence[Tensor] | Tensor,
    *,
    jac_outputs: Sequence[Tensor] | Tensor | None = None,
    retain_graph: bool = False,
    parallel_chunk_size: int | None = None,
) -> tuple[Tensor, ...]:
    r"""
    Computes the Jacobians of ``outputs`` with respect to ``inputs``, left-multiplied by
    ``jac_outputs`` (or identity if ``jac_outputs`` is ``None``), and returns the result as a tuple,
    with one Jacobian per input tensor. The returned Jacobian with respect to input ``t`` has shape
    ``[m] + t.shape``.

    :param outputs: The tensor or tensors to differentiate. Should be non-empty.
    :param inputs: The tensor or tensors with respect to which the Jacobian must be computed. These
        must have their ``requires_grad`` flag set to ``True``.
    :param jac_outputs: The initial Jacobians to backpropagate, analog to the ``grad_outputs``
        parameter of :func:`torch.autograd.grad`. If provided, it must have the same structure as
        ``outputs`` and each tensor in ``jac_outputs`` must match the shape of the corresponding
        tensor in ``outputs``, with an extra leading dimension representing the number of rows of
        the resulting Jacobian (e.g. the number of losses). If ``None``, defaults to the identity
        matrix. In this case, the standard Jacobian of ``outputs`` is computed, with one row for
        each value in the ``outputs``.
    :param retain_graph: If ``False``, the graph used to compute the grad will be freed. Defaults to
        ``False``.
    :param parallel_chunk_size: The number of scalars to differentiate simultaneously in the
        backward pass. If set to ``None``, all coordinates of ``outputs`` will be differentiated in
        parallel at once. If set to ``1``, all coordinates will be differentiated sequentially. A
        larger value results in faster differentiation, but also higher memory usage. Defaults to
        ``None``.

    .. note::
        The only difference between this function and :func:`torchjd.autojac.backward`, is that it
        returns the Jacobians as a tuple, while :func:`torchjd.autojac.backward` stores them in the
        ``.jac`` fields of the inputs.

    .. admonition::
        Example

        The following example shows how to use ``jac``.

            >>> import torch
            >>>
            >>> from torchjd.autojac import jac
            >>>
            >>> param = torch.tensor([1., 2.], requires_grad=True)
            >>> # Compute arbitrary quantities that are function of param
            >>> y1 = torch.tensor([-1., 1.]) @ param
            >>> y2 = (param ** 2).sum()
            >>>
            >>> jacobians = jac([y1, y2], param)
            >>>
            >>> jacobians
            (tensor([[-1., 1.],
                    [ 2., 4.]]),)

    .. admonition::
        Example

        The following example shows how to compute jacobians, combine them into a single Jacobian
        matrix, and compute its Gramian.

            >>> import torch
            >>>
            >>> from torchjd.autojac import jac
            >>>
            >>> weight = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)  # shape: [2, 2]
            >>> bias = torch.tensor([0.5, -0.5], requires_grad=True)  # shape: [2]
            >>> # Compute arbitrary quantities that are function of weight and bias
            >>> input_vec = torch.tensor([1., -1.])
            >>> y1 = weight @ input_vec + bias  # shape: [2]
            >>> y2 = (weight ** 2).sum() + (bias ** 2).sum()  # shape: [] (scalar)
            >>>
            >>> jacobians = jac([y1, y2], [weight, bias])  # shapes: [3, 2, 2], [3, 2]
            >>> jacobian_matrices = tuple(J.flatten(1) for J in jacobians)  # shapes: [3, 4], [3, 2]
            >>> combined_jacobian_matrix = torch.concat(jacobian_matrices, dim=1)  # shape: [3, 6]
            >>> gramian = combined_jacobian_matrix @ combined_jacobian_matrix.T  # shape: [3, 3]
            >>> gramian
            tensor([[  3.,   0.,  -1.],
                    [  0.,   3.,  -3.],
                    [ -1.,  -3., 122.]])

        The obtained gramian is a symmetric matrix containing the dot products between all pairs of
        gradients. It's a strong indicator of gradient norm (the diagonal elements are the squared
        norms of the gradients) and conflict (a negative off-diagonal value means that the gradients
        conflict). In fact, most aggregators base their decision entirely on the gramian.

        In this case, we can see that the first two gradients (those of y1) both have a squared norm
        of 3, while the third gradient (that of y2) has a squared norm of 122. The first two
        gradients are exactly orthogonal (they have an inner product of 0), but they conflict with
        the third gradient (inner product of -1 and -3).

    .. admonition::
        Example

        This example shows how to apply chain rule using the ``jac_outputs`` parameter to compute
        the Jacobian in two steps.

            >>> import torch
            >>>
            >>> from torchjd.autojac import jac
            >>>
            >>> x = torch.tensor([1., 2.], requires_grad=True)
            >>> # Compose functions: x -> h -> y
            >>> h = x ** 2
            >>> y1 = h.sum()
            >>> y2 = torch.tensor([1., -1.]) @ h
            >>>
            >>> # Step 1: Compute d[y1,y2]/dh
            >>> jac_h = jac([y1, y2], [h])[0]  # Shape: [2, 2]
            >>>
            >>> # Step 2: Use chain rule to compute d[y1,y2]/dx = (d[y1,y2]/dh) @ (dh/dx)
            >>> jac_x = jac(h, x, jac_outputs=jac_h)[0]
            >>>
            >>> jac_x
            tensor([[ 2.,  4.],
                    [ 2., -4.]])

        This two-step computation is equivalent to directly computing ``jac([y1, y2], x)``.

    .. warning::
        To differentiate in parallel, ``jac`` relies on ``torch.vmap``, which has some
        limitations: `it does not work on the output of compiled functions
        <https://github.com/pytorch/pytorch/issues/138422>`_, `when some tensors have
        <https://github.com/TorchJD/torchjd/issues/184>`_ ``retains_grad=True`` or `when using an
        RNN on CUDA <https://github.com/TorchJD/torchjd/issues/220>`_, for instance. If you
        experience issues with ``jac`` try to use ``parallel_chunk_size=1`` to avoid relying on
        ``torch.vmap``.
    """

    check_optional_positive_chunk_size(parallel_chunk_size)

    outputs_ = as_checked_ordered_set(outputs, "outputs")
    if len(outputs_) == 0:
        raise ValueError("`outputs` cannot be empty.")

    # Preserve repetitions to duplicate jacobians at the return statement
    inputs_with_repetition = cast(Sequence[Tensor], (inputs,) if is_tensor_like(inputs) else inputs)
    inputs_ = OrderedSet(inputs_with_repetition)

    jac_outputs_dict = _create_jac_outputs_dict(outputs_, jac_outputs)
    transform = _create_transform(outputs_, inputs_, parallel_chunk_size, retain_graph)
    result = transform(jac_outputs_dict)
    return tuple(result[input] for input in inputs_with_repetition)


def _create_jac_outputs_dict(
    outputs: OrderedSet[Tensor],
    opt_jac_outputs: Sequence[Tensor] | Tensor | None,
) -> dict[Tensor, Tensor]:
    """
    Creates a dictionary mapping outputs to their corresponding Jacobians.

    :param outputs: The tensors to differentiate.
    :param opt_jac_outputs: The initial Jacobians to backpropagate. If ``None``, defaults to
        identity.
    """
    if opt_jac_outputs is None:
        # Transform that creates gradient outputs containing only ones.
        init = Init(outputs)
        # Transform that turns the gradients into Jacobians.
        diag = Diagonalize(outputs)
        return (diag << init)({})
    jac_outputs = cast(
        Sequence[Tensor], (opt_jac_outputs,) if is_tensor_like(opt_jac_outputs) else opt_jac_outputs
    )
    check_matching_length(jac_outputs, outputs, "jac_outputs", "outputs")
    check_matching_jac_shapes(jac_outputs, outputs, "jac_outputs", "outputs")
    check_consistent_first_dimension(jac_outputs, "jac_outputs")
    return dict(zip(outputs, jac_outputs, strict=True))


def _create_transform(
    outputs: OrderedSet[Tensor],
    inputs: OrderedSet[Tensor],
    parallel_chunk_size: int | None,
    retain_graph: bool,
) -> Transform:
    return Jac(outputs, inputs, parallel_chunk_size, retain_graph)
