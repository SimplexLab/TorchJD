from collections.abc import Iterable, Sequence

from torch import Tensor

from ._transform import (
    AccumulateGrad,
    AccumulateJac,
    Grad,
    Init,
    Jac,
    OrderedSet,
    Select,
    Stack,
    Transform,
)
from ._utils import (
    as_checked_ordered_set,
    check_matching_grad_shapes,
    check_matching_length,
    check_optional_positive_chunk_size,
    get_leaf_tensors,
)


def mtl_backward(
    tensors: Sequence[Tensor],
    /,
    features: Sequence[Tensor] | Tensor,
    *,
    grad_tensors: Sequence[Tensor] | None = None,
    tasks_params: Sequence[Iterable[Tensor]] | None = None,
    shared_params: Iterable[Tensor] | None = None,
    retain_graph: bool = False,
    parallel_chunk_size: int | None = None,
) -> None:
    """
    In the context of Multi-Task Learning (MTL), we often have a shared feature extractor followed
    by several task-specific heads. A loss can then be computed for each task.

    This function computes the gradient of each task-specific tensor with respect to its
    task-specific parameters and accumulates it in their ``.grad`` fields. It also computes the
    Jacobian of all tensors with respect to the shared parameters and accumulates it in their
    ``.jac`` fields. These Jacobians have one row per task.

    If the ``tensors`` are non-scalar, ``mtl_backward`` requires some initial gradients in
    ``grad_tensors``. This allows to compose ``mtl_backward`` with some other function computing
    the gradients with respect to the tensors (chain rule).

    :param tensors: The task-specific tensors. If these are scalar (e.g. the losses produced by
        every task), no ``grad_tensors`` are needed. If these are non-scalar tensors, providing some
        ``grad_tensors`` is necessary.
    :param features: The last shared representation used for all tasks, as given by the feature
        extractor. Should be non-empty.
    :param grad_tensors: The initial gradients to backpropagate, analog to the ``grad_tensors``
        parameter of :func:`torch.autograd.backward`. If any of the ``tensors`` is non-scalar,
        ``grad_tensors`` must be provided, with the same length and shapes as ``tensors``.
        Otherwise, this parameter is not needed and will default to scalars of 1.
    :param tasks_params: The parameters of each task-specific head. Their ``requires_grad`` flags
        must be set to ``True``. If not provided, the parameters considered for each task will
        default to the leaf tensors that are in the computation graph of its tensor, but that were
        not used to compute the ``features``.
    :param shared_params: The parameters of the shared feature extractor. Their ``requires_grad``
        flags must be set to ``True``. If not provided, defaults to the leaf tensors that are in the
        computation graph of the ``features``.
    :param retain_graph: If ``False``, the graph used to compute the grad will be freed. Defaults to
        ``False``.
    :param parallel_chunk_size: The number of scalars to differentiate simultaneously in the
        backward pass. If set to ``None``, all coordinates of ``tensors`` will be differentiated in
        parallel at once. If set to ``1``, all coordinates will be differentiated sequentially. A
        larger value results in faster differentiation, but also higher memory usage. Defaults to
        ``None``.

    .. admonition::
        Example

        A usage example of ``mtl_backward`` is provided in
        :doc:`Multi-Task Learning (MTL) <../../examples/mtl>`.

    .. note::
        ``shared_params`` should contain no parameter in common with ``tasks_params``. The different
        tasks may have some parameters in common. In this case, the sum of the gradients with
        respect to those parameters will be accumulated into their ``.grad`` fields.

    .. warning::
        To differentiate in parallel, ``mtl_backward`` relies on ``torch.vmap``, which has some
        limitations: `it does not work on the output of compiled functions
        <https://github.com/pytorch/pytorch/issues/138422>`_, `when some tensors have
        <https://github.com/SimplexLab/TorchJD/issues/184>`_ ``retains_grad=True`` or `when using an
        RNN on CUDA <https://github.com/SimplexLab/TorchJD/issues/220>`_, for instance. If you
        experience issues with ``backward`` try to use ``parallel_chunk_size=1`` to avoid relying on
        ``torch.vmap``.
    """

    check_optional_positive_chunk_size(parallel_chunk_size)

    tensors_ = as_checked_ordered_set(tensors, "tensors")
    features_ = as_checked_ordered_set(features, "features")

    if shared_params is None:
        shared_params_ = get_leaf_tensors(tensors=features_, excluded=[])
    else:
        shared_params_ = OrderedSet(shared_params)
    if tasks_params is None:
        tasks_params_ = [get_leaf_tensors(tensors=[t], excluded=features_) for t in tensors_]
    else:
        tasks_params_ = [OrderedSet(task_params) for task_params in tasks_params]

    if len(features_) == 0:
        raise ValueError("`features` cannot be empty.")

    if len(tensors_) == 0:
        raise ValueError("`tensors` cannot be empty.")
    if len(tensors_) != len(tasks_params_):
        raise ValueError("`tensors` and `tasks_params` should have the same size.")

    _check_no_overlap(shared_params_, tasks_params_)

    grad_tensors_dict = _create_grad_tensors_dict(tensors_, grad_tensors)
    backward_transform = _create_transform(
        tensors=tensors_,
        features=features_,
        tasks_params=tasks_params_,
        shared_params=shared_params_,
        retain_graph=retain_graph,
        parallel_chunk_size=parallel_chunk_size,
    )

    backward_transform(grad_tensors_dict)


def _create_grad_tensors_dict(
    tensors: OrderedSet[Tensor],
    opt_grad_tensors: Sequence[Tensor] | None,
) -> dict[Tensor, Tensor]:
    if opt_grad_tensors is None:
        _check_tensors_are_scalar(tensors)
        grad_tensors_dict = Init(tensors)({})
    else:
        check_matching_length(opt_grad_tensors, tensors, "grad_tensors", "tensors")
        check_matching_grad_shapes(opt_grad_tensors, tensors, "grad_tensors", "tensors")
        grad_tensors_dict = dict(zip(tensors, opt_grad_tensors, strict=True))

    return grad_tensors_dict


def _create_transform(
    tensors: OrderedSet[Tensor],
    features: OrderedSet[Tensor],
    tasks_params: list[OrderedSet[Tensor]],
    shared_params: OrderedSet[Tensor],
    retain_graph: bool,
    parallel_chunk_size: int | None,
) -> Transform:
    """
    Creates the backward transform for a multi-task learning problem. It is a hybrid between
    Jacobian descent (for shared parameters) and multiple gradient descent branches (for
    task-specific parameters).
    """

    # Task-specific transforms. Each of them computes and accumulates the gradient of the task's
    # tensor w.r.t. the task's specific parameters, and computes and backpropagates the gradient of
    # the tensor w.r.t. the shared representations.
    task_transforms = [
        _create_task_transform(
            features,
            task_params,
            OrderedSet([t]),
            retain_graph,
        )
        for task_params, t in zip(tasks_params, tensors, strict=True)
    ]

    # Transform that stacks the gradients of the tensors w.r.t. the shared representations into a
    # Jacobian.
    stack = Stack(task_transforms)

    # Transform that computes the Jacobians of the tensors w.r.t. the shared parameters.
    jac = Jac(features, shared_params, parallel_chunk_size, retain_graph)

    # Transform that accumulates the result in the .jac field of the shared parameters.
    accumulate = AccumulateJac()

    return accumulate << jac << stack


def _create_task_transform(
    features: OrderedSet[Tensor],
    task_params: OrderedSet[Tensor],
    tensor: OrderedSet[Tensor],  # contains a single tensor
    retain_graph: bool,
) -> Transform:
    # Tensors with respect to which we compute the gradients.
    to_differentiate = task_params + features

    # Transform that computes the gradients of the tensor w.r.t. the task-specific parameters and
    # the features.
    grad = Grad(tensor, to_differentiate, retain_graph)

    # Transform that accumulates the gradients w.r.t. the task-specific parameters into their
    # .grad fields.
    accumulate = AccumulateGrad() << Select(task_params)

    # Transform that backpropagates the gradients of the tensor w.r.t. the features.
    backpropagate = Select(features)

    # Transform that accumulates the gradient of the tensor w.r.t. the task-specific parameters into
    # their .grad fields and backpropagates the gradient of the tensor w.r.t. to the features.
    backward_task = (backpropagate | accumulate) << grad << Select(tensor)
    return backward_task


def _check_tensors_are_scalar(tensors: Iterable[Tensor]) -> None:
    for t in tensors:
        if t.ndim > 0:
            raise ValueError(
                "When `tensors` are non-scalar, the `grad_tensors` parameter must be provided."
            )


def _check_no_overlap(
    shared_params: Iterable[Tensor],
    tasks_params: Sequence[Iterable[Tensor]],
) -> None:
    task_param_set = {param for task_params in tasks_params for param in task_params}
    shared_param_set = set(shared_params)
    intersection = task_param_set.intersection(shared_param_set)

    if len(intersection) != 0:
        raise ValueError("`tasks_params` should contain no tensor in common with `shared_params`.")
