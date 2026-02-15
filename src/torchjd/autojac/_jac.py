from collections.abc import Iterable, Sequence

from torch import Tensor

from torchjd.autojac._transform._base import Transform
from torchjd.autojac._transform._diagonalize import Diagonalize
from torchjd.autojac._transform._init import Init
from torchjd.autojac._transform._jac import Jac
from torchjd.autojac._transform._ordered_set import OrderedSet
from torchjd.autojac._utils import (
    as_checked_ordered_set,
    check_optional_positive_chunk_size,
    get_leaf_tensors,
)


def jac(
    outputs: Sequence[Tensor] | Tensor,
    inputs: Iterable[Tensor] | None = None,
    retain_graph: bool = False,
    parallel_chunk_size: int | None = None,
) -> tuple[Tensor, ...]:
    r"""
    Computes the Jacobian of all values in ``outputs`` with respect to all ``inputs``. Returns the
    result as a tuple, with one Jacobian per input tensor. The returned Jacobian with respect to
    input ``t`` has shape ``[m] + t.shape``.

    :param outputs: The tensor or tensors to differentiate. Should be non-empty. The Jacobians will
        have one row for each value of each of these tensors.
    :param inputs: The tensors with respect to which the Jacobian must be computed. These must have
        their ``requires_grad`` flag set to ``True``. If not provided, defaults to the leaf tensors
        that were used to compute the ``outputs`` parameter.
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
            >>> jacobians = jac([y1, y2], [param])
            >>>
            >>> jacobians
            (tensor([-1., 1.],
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
        raise ValueError("`outputs` cannot be empty")

    if inputs is None:
        inputs_ = get_leaf_tensors(tensors=outputs_, excluded=set())
        inputs_with_repetition = list(inputs_)
    else:
        inputs_with_repetition = list(inputs)  # Create a list to avoid emptying generator
        inputs_ = OrderedSet(inputs_with_repetition)

    jac_transform = _create_transform(
        outputs=outputs_,
        inputs=inputs_,
        retain_graph=retain_graph,
        parallel_chunk_size=parallel_chunk_size,
    )

    result = jac_transform({})
    return tuple(result[input] for input in inputs_with_repetition)


def _create_transform(
    outputs: OrderedSet[Tensor],
    inputs: OrderedSet[Tensor],
    retain_graph: bool,
    parallel_chunk_size: int | None,
) -> Transform:
    # Transform that creates gradient outputs containing only ones.
    init = Init(outputs)

    # Transform that turns the gradients into Jacobians.
    diag = Diagonalize(outputs)

    # Transform that computes the required Jacobians.
    jac = Jac(outputs, inputs, parallel_chunk_size, retain_graph)

    return jac << diag << init
