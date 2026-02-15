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
from ._utils import as_checked_ordered_set, check_optional_positive_chunk_size, get_leaf_tensors


def mtl_backward(
    losses: Sequence[Tensor],
    features: Sequence[Tensor] | Tensor,
    tasks_params: Sequence[Iterable[Tensor]] | None = None,
    shared_params: Iterable[Tensor] | None = None,
    retain_graph: bool = False,
    parallel_chunk_size: int | None = None,
) -> None:
    """
    In the context of Multi-Task Learning (MTL), we often have a shared feature extractor followed
    by several task-specific heads. A loss can then be computed for each task.

    This function computes the gradient of each task-specific loss with respect to its task-specific
    parameters and accumulates it in their ``.grad`` fields. Then, it computes the Jacobian of all
    losses with respect to the shared parameters and accumulates it in their ``.jac`` fields.

    :param losses: The task losses. The Jacobians will have one row per loss.
    :param features: The last shared representation used for all tasks, as given by the feature
        extractor. Should be non-empty.
    :param tasks_params: The parameters of each task-specific head. Their ``requires_grad`` flags
        must be set to ``True``. If not provided, the parameters considered for each task will
        default to the leaf tensors that are in the computation graph of its loss, but that were not
        used to compute the ``features``.
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

    losses_ = as_checked_ordered_set(losses, "losses")
    features_ = as_checked_ordered_set(features, "features")

    if shared_params is None:
        shared_params_ = get_leaf_tensors(tensors=features_, excluded=[])
    else:
        shared_params_ = OrderedSet(shared_params)
    if tasks_params is None:
        tasks_params_ = [get_leaf_tensors(tensors=[loss], excluded=features_) for loss in losses_]
    else:
        tasks_params_ = [OrderedSet(task_params) for task_params in tasks_params]

    if len(features_) == 0:
        raise ValueError("`features` cannot be empty.")

    _check_no_overlap(shared_params_, tasks_params_)
    _check_losses_are_scalar(losses_)

    if len(losses_) == 0:
        raise ValueError("`losses` cannot be empty")
    if len(losses_) != len(tasks_params_):
        raise ValueError("`losses` and `tasks_params` should have the same size.")

    backward_transform = _create_transform(
        losses=losses_,
        features=features_,
        tasks_params=tasks_params_,
        shared_params=shared_params_,
        retain_graph=retain_graph,
        parallel_chunk_size=parallel_chunk_size,
    )

    backward_transform({})


def _create_transform(
    losses: OrderedSet[Tensor],
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
    # loss w.r.t. the task's specific parameters, and computes and backpropagates the gradient of
    # the losses w.r.t. the shared representations.
    task_transforms = [
        _create_task_transform(
            features,
            task_params,
            OrderedSet([loss]),
            retain_graph,
        )
        for task_params, loss in zip(tasks_params, losses, strict=True)
    ]

    # Transform that stacks the gradients of the losses w.r.t. the shared representations into a
    # Jacobian.
    stack = Stack(task_transforms)

    # Transform that computes the Jacobians of the losses w.r.t. the shared parameters.
    jac = Jac(features, shared_params, parallel_chunk_size, retain_graph)

    # Transform that accumulates the result in the .jac field of the shared parameters.
    accumulate = AccumulateJac()

    return accumulate << jac << stack


def _create_task_transform(
    features: OrderedSet[Tensor],
    task_params: OrderedSet[Tensor],
    loss: OrderedSet[Tensor],  # contains a single scalar loss
    retain_graph: bool,
) -> Transform:
    # Tensors with respect to which we compute the gradients.
    to_differentiate = task_params + features

    # Transform that initializes the gradient output to 1.
    init = Init(loss)

    # Transform that computes the gradients of the loss w.r.t. the task-specific parameters and
    # the features.
    grad = Grad(loss, to_differentiate, retain_graph)

    # Transform that accumulates the gradients w.r.t. the task-specific parameters into their
    # .grad fields.
    accumulate = AccumulateGrad() << Select(task_params)

    # Transform that backpropagates the gradients of the losses w.r.t. the features.
    backpropagate = Select(features)

    # Transform that accumulates the gradient of the losses w.r.t. the task-specific parameters into
    # their .grad fields and backpropagates the gradient of the losses w.r.t. to the features.
    backward_task = (backpropagate | accumulate) << grad << init
    return backward_task


def _check_losses_are_scalar(losses: Iterable[Tensor]) -> None:
    for loss in losses:
        if loss.ndim > 0:
            raise ValueError("`losses` should contain only scalars.")


def _check_no_overlap(
    shared_params: Iterable[Tensor],
    tasks_params: Sequence[Iterable[Tensor]],
) -> None:
    task_param_set = {param for task_params in tasks_params for param in task_params}
    shared_param_set = set(shared_params)
    intersection = task_param_set.intersection(shared_param_set)

    if len(intersection) != 0:
        raise ValueError("`tasks_params` should contain no tensor in common with `shared_params`.")
