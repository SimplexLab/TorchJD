from collections.abc import Iterable, Sequence

from torch import Tensor

from ._transform import AccumulateJac, Diagonalize, Init, Jac, OrderedSet, Transform
from ._utils import as_checked_ordered_set, check_optional_positive_chunk_size, get_leaf_tensors


def backward(
    tensors: Sequence[Tensor] | Tensor,
    inputs: Iterable[Tensor] | None = None,
    retain_graph: bool = False,
    parallel_chunk_size: int | None = None,
) -> None:
    r"""
    Computes the Jacobians of all values in ``tensors`` with respect to all ``inputs`` and
    accumulates them in the ``.jac`` fields of the ``inputs``.

    :param tensors: The tensor or tensors to differentiate. Should be non-empty. The Jacobians will
        have one row for each value of each of these tensors.
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

        The following code snippet showcases a simple usage of ``backward``.

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

    .. warning::
        To differentiate in parallel, ``backward`` relies on ``torch.vmap``, which has some
        limitations: `it does not work on the output of compiled functions
        <https://github.com/pytorch/pytorch/issues/138422>`_, `when some tensors have
        <https://github.com/SimplexLab/torchjd/issues/184>`_ ``retains_grad=True`` or `when using an
        RNN on CUDA <https://github.com/SimplexLab/torchjd/issues/220>`_, for instance. If you
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

    backward_transform = _create_transform(
        tensors=tensors_,
        inputs=inputs_,
        retain_graph=retain_graph,
        parallel_chunk_size=parallel_chunk_size,
    )

    backward_transform({})


def _create_transform(
    tensors: OrderedSet[Tensor],
    inputs: OrderedSet[Tensor],
    retain_graph: bool,
    parallel_chunk_size: int | None,
) -> Transform:
    """Creates the backward transform."""

    # Transform that creates gradient outputs containing only ones.
    init = Init(tensors)

    # Transform that turns the gradients into Jacobians.
    diag = Diagonalize(tensors)

    # Transform that computes the required Jacobians.
    jac = Jac(tensors, inputs, parallel_chunk_size, retain_graph)

    # Transform that accumulates the result in the .jac field of the inputs.
    accumulate = AccumulateJac()

    return accumulate << jac << diag << init
