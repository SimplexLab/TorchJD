from collections.abc import Iterable, Sequence

from torch import Tensor

from ._transform import AccumulateJac, Diagonalize, Init, Jac, OrderedSet, Transform
from ._utils import (
    as_checked_ordered_set,
    check_consistent_first_dimension,
    check_matching_length,
    check_matching_shapes,
    check_optional_positive_chunk_size,
    get_leaf_tensors,
)


def backward(
    tensors: Sequence[Tensor] | Tensor,
    jac_tensors: Sequence[Tensor] | Tensor | None = None,
    inputs: Iterable[Tensor] | None = None,
    retain_graph: bool = False,
    parallel_chunk_size: int | None = None,
) -> None:
    r"""
    Computes the Jacobians of ``tensors`` with respect to ``inputs``, left-multiplied by
    ``jac_tensors`` (or identity if ``jac_tensors`` is ``None``), and accumulates the results in the
    ``.jac`` fields of the ``inputs``.

    :param tensors: The tensor or tensors to differentiate. Should be non-empty.
    :param jac_tensors: The initial Jacobians to backpropagate, analog to the `grad_tensors`
        parameter of `torch.autograd.backward`. If provided, it must have the same structure as
        ``tensors`` and each tensor in ``jac_tensors`` must match the shape of the corresponding
        tensor in ``tensors``, with an extra leading dimension representing the number of rows of
        the resulting Jacobian (e.g. the number of losses). If ``None``, defaults to the identity
        matrix. In this case, the standard Jacobian of ``tensors`` is computed, with one row for
        each value in the ``tensors``.
    :param inputs: The tensors with respect to which the Jacobians must be computed. These must have
        their ``requires_grad`` flag set to ``True``. If not provided, defaults to the leaf tensors
        that were used to compute the ``tensors`` parameter.
    :param retain_graph: If ``False``, the graph used to compute the grad will be freed. Defaults to
        ``False``.
    :param parallel_chunk_size: The number of scalars to differentiate simultaneously in the
        backward pass. If set to ``None``, all coordinates of ``tensors`` will be differentiated in
        parallel at once. If set to ``1``, all coordinates will be differentiated sequentially. A
        larger value results in faster differentiation, but also higher memory usage. Defaults to
        ``None``.

    .. admonition::
        Example

        This example shows a simple usage of ``backward``.

            >>> import torch
            >>>
            >>> from torchjd.autojac import backward
            >>>
            >>> param = torch.tensor([1., 2.], requires_grad=True)
            >>> # Compute arbitrary quantities that are function of param
            >>> y1 = torch.tensor([-1., 1.]) @ param
            >>> y2 = (param ** 2).sum()
            >>>
            >>> backward([y1, y2])
            >>>
            >>> param.jac
            tensor([[-1.,  1.],
                    [ 2.,  4.]])

        The ``.jac`` field of ``param`` now contains the Jacobian of
        :math:`\begin{bmatrix}y_1 \\ y_2\end{bmatrix}` with respect to ``param``.

    .. admonition::
        Example

        This is the same example as before, except that we explicitly specify the identity
        ``jac_tensors`` (which is equivalent to using the default `None`).

            >>> import torch
            >>>
            >>> from torchjd.autojac import backward
            >>>
            >>> param = torch.tensor([1., 2.], requires_grad=True)
            >>> # Compute arbitrary quantities that are function of param
            >>> y1 = torch.tensor([-1., 1.]) @ param
            >>> y2 = (param ** 2).sum()
            >>>
            >>> J1 = torch.tensor([1.0, 0.0])
            >>> J2 = torch.tensor([0.0, 1.0])
            >>>
            >>> backward([y1, y2], jac_tensors=[J1, J2])
            >>>
            >>> param.jac
            tensor([[-1.,  1.],
                    [ 2.,  4.]])

        Instead of using the identity ``jac_tensors``, you can backpropagate some Jacobians obtained
        by a call to :func:`torchjd.autojac.jac` on a later part of the computation graph.

    .. warning::
        To differentiate in parallel, ``backward`` relies on ``torch.vmap``, which has some
        limitations: `it does not work on the output of compiled functions
        <https://github.com/pytorch/pytorch/issues/138422>`_, `when some tensors have
        <https://github.com/TorchJD/torchjd/issues/184>`_ ``retains_grad=True`` or `when using an
        RNN on CUDA <https://github.com/TorchJD/torchjd/issues/220>`_, for instance. If you
        experience issues with ``backward`` try to use ``parallel_chunk_size=1`` to avoid relying on
        ``torch.vmap``.
    """
    check_optional_positive_chunk_size(parallel_chunk_size)

    tensors_ = as_checked_ordered_set(tensors, "tensors")

    if len(tensors_) == 0:
        raise ValueError("`tensors` cannot be empty")

    if inputs is None:
        inputs_ = get_leaf_tensors(tensors=tensors_, excluded=set())
    else:
        inputs_ = OrderedSet(inputs)

    jac_tensors_dict = _create_jac_tensors_dict(tensors_, jac_tensors)
    transform = _create_transform(tensors_, inputs_, parallel_chunk_size, retain_graph)
    transform(jac_tensors_dict)


def _create_jac_tensors_dict(
    tensors: OrderedSet[Tensor],
    opt_jac_tensors: Sequence[Tensor] | Tensor | None,
) -> dict[Tensor, Tensor]:
    """
    Creates a dictionary mapping tensors to their corresponding Jacobians.

    :param tensors: The tensors to differentiate.
    :param opt_jac_tensors: The initial Jacobians to backpropagate. If ``None``, defaults to
        identity.
    """
    if opt_jac_tensors is None:
        # Transform that creates gradient outputs containing only ones.
        init = Init(tensors)
        # Transform that turns the gradients into Jacobians.
        diag = Diagonalize(tensors)
        return (diag << init)({})
    jac_tensors = [opt_jac_tensors] if isinstance(opt_jac_tensors, Tensor) else opt_jac_tensors
    check_matching_length(jac_tensors, tensors, "jac_tensors", "tensors")
    check_matching_shapes(jac_tensors, tensors, "jac_tensors", "tensors")
    check_consistent_first_dimension(jac_tensors, "jac_tensors")
    return dict(zip(tensors, jac_tensors, strict=True))


def _create_transform(
    tensors: OrderedSet[Tensor],
    inputs: OrderedSet[Tensor],
    parallel_chunk_size: int | None,
    retain_graph: bool,
) -> Transform:
    """Creates the backward transform that computes and accumulates Jacobians."""
    # Transform that computes the required Jacobians.
    jac = Jac(tensors, inputs, parallel_chunk_size, retain_graph)
    # Transform that accumulates the result in the .jac field of the inputs.
    accumulate = AccumulateJac()
    return accumulate << jac
