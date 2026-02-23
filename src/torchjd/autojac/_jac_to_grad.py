from collections import deque
from collections.abc import Iterable
from typing import TypeGuard, cast, overload

import torch
from torch import Tensor, nn

from torchjd._linalg import Matrix, PSDMatrix, compute_gramian
from torchjd.aggregation import Aggregator, Weighting
from torchjd.aggregation._aggregator_bases import GramianWeightedAggregator, WeightedAggregator

from ._accumulation import TensorWithJac, accumulate_grads, is_tensor_with_jac
from ._utils import check_consistent_first_dimension


@overload
def jac_to_grad(
    tensors: Iterable[Tensor],
    /,
    aggregator: GramianWeightedAggregator,
    *,
    retain_jac: bool = False,
    optimize_gramian_computation: bool = False,
) -> Tensor: ...


@overload
def jac_to_grad(
    tensors: Iterable[Tensor],
    /,
    aggregator: WeightedAggregator,  # Not a GramianWA, because overloads are checked in order
    *,
    retain_jac: bool = False,
) -> Tensor: ...


@overload
def jac_to_grad(
    tensors: Iterable[Tensor],
    /,
    aggregator: Aggregator,  # Not a WeightedAggregator, because overloads are checked in order
    *,
    retain_jac: bool = False,
) -> None: ...


def jac_to_grad(
    tensors: Iterable[Tensor],
    /,
    aggregator: Aggregator,
    *,
    retain_jac: bool = False,
    optimize_gramian_computation: bool = False,
) -> Tensor | None:
    r"""
    Aggregates the Jacobians stored in the ``.jac`` fields of ``tensors`` and accumulates the result
    into their ``.grad`` fields.

    :param tensors: The tensors whose ``.jac`` fields should be aggregated. All Jacobians must
        have the same first dimension (e.g. number of losses).
    :param aggregator: The aggregator used to reduce the Jacobians into gradients. If it uses a
        :class:`Weighting <torchjd.aggregation._weighting_bases.Weighting>` to combine the rows of
        the Jacobians, ``jac_to_grad`` will also return the computed weights.
    :param retain_jac: Whether to preserve the ``.jac`` fields of the tensors after they have been
        used. Defaults to ``False``.
    :param optimize_gramian_computation: When the ``aggregator`` computes weights based on the
        Gramian of the Jacobian, it's possible to skip the concatenation of the Jacobians and to
        instead compute the Gramian as the sum of the Gramians of the individual Jacobians. This
        saves memory (up to 50% memory saving) but can be slightly slower (up to 15%) on CUDA. We
        advise to try this optimization if memory is an issue for you. Defaults to ``False``.

    .. note::
        This function starts by "flattening" the ``.jac`` fields into matrices (i.e. flattening all
        of their dimensions except the first one), then concatenates those matrices into a combined
        Jacobian matrix. The aggregator is then used on this matrix, which returns a combined
        gradient vector, that is split and reshaped to fit into the ``.grad`` fields of the tensors.

    .. admonition::
        Example

        This example shows how to use ``jac_to_grad`` after a call to ``backward``

            >>> import torch
            >>>
            >>> from torchjd.aggregation import UPGrad
            >>> from torchjd.autojac import backward, jac_to_grad
            >>>
            >>> param = torch.tensor([1., 2.], requires_grad=True)
            >>> # Compute arbitrary quantities that are function of param
            >>> y1 = torch.tensor([-1., 1.]) @ param
            >>> y2 = (param ** 2).sum()
            >>>
            >>> backward([y1, y2])  # param now has a .jac field
            >>> weights = jac_to_grad([param], UPGrad())  # param now has a .grad field
            >>> param.grad
            tensor([0.5000, 2.5000])
            >>> weights
            tensor([0.5,  0.5])

        The ``.grad`` field of ``param`` now contains the aggregation (by UPGrad) of the Jacobian of
        :math:`\begin{bmatrix}y_1 \\ y_2\end{bmatrix}` with respect to ``param``. In this case, the
        weights used to combine the Jacobian are equal because there was no conflict.
    """

    tensors_ = list[TensorWithJac]()
    for t in tensors:
        if not is_tensor_with_jac(t):
            raise ValueError(
                "Some `jac` fields were not populated. Did you use `autojac.backward` or "
                "`autojac.mtl_backward` before calling `jac_to_grad`?",
            )
        tensors_.append(t)

    if len(tensors_) == 0:
        raise ValueError("The `tensors` parameter cannot be empty.")

    jacobians = deque(t.jac for t in tensors_)
    check_consistent_first_dimension(jacobians, "tensors.jac")

    if not retain_jac:
        _free_jacs(tensors_)

    if optimize_gramian_computation:
        if not _can_skip_jacobian_combination(aggregator):
            raise ValueError(
                "In order to use `jac_to_grad` with `optimize_gramian_computation=True`, you must "
                "provide an `Aggregator` that computes weights based on the Gramian of the Jacobian"
                " (e.g. `UPGrad`) and that doesn't have any forward hooks attached to it."
            )

        gradients, weights = _gramian_based(aggregator, jacobians)
    else:
        gradients, weights = _jacobian_based(aggregator, jacobians, tensors_)
    accumulate_grads(tensors_, gradients)

    return weights


def _can_skip_jacobian_combination(aggregator: Aggregator) -> TypeGuard[GramianWeightedAggregator]:
    return (
        isinstance(aggregator, GramianWeightedAggregator)
        and not _has_forward_hook(aggregator)
        and not _has_forward_hook(aggregator.weighting)
    )


def _has_forward_hook(module: nn.Module) -> bool:
    """Return whether the module has any forward hook registered."""
    return len(module._forward_hooks) > 0 or len(module._forward_pre_hooks) > 0


def _jacobian_based(
    aggregator: Aggregator,
    jacobians: deque[Tensor],
    tensors: list[TensorWithJac],
) -> tuple[list[Tensor], Tensor | None]:
    jacobian_matrix = _unite_jacobians(jacobians)
    weights: Tensor | None = None

    if isinstance(aggregator, WeightedAggregator):

        def capture_hook(_m: Weighting[Matrix], _i: tuple[Tensor], output: Tensor) -> None:
            nonlocal weights
            weights = output

        # Append the weight-capturing post-hook to the outer weighting to ensure that all other
        # post-hooks of the outer and inner weighting are run (potentially with effect on the
        # weights) prior to capturing the weights.
        handle = aggregator.weighting.register_forward_hook(capture_hook)

        # Using a try-finally here in case an exception is raised by the aggregator.
        try:
            gradient_vector = aggregator(jacobian_matrix)
        finally:
            handle.remove()
    else:
        gradient_vector = aggregator(jacobian_matrix)
    gradients = _disunite_gradient(gradient_vector, tensors)
    return gradients, weights


def _gramian_based(
    aggregator: GramianWeightedAggregator,
    jacobians: deque[Tensor],
) -> tuple[list[Tensor], Tensor]:
    weighting = aggregator.gramian_weighting
    gramian = _compute_gramian_sum(jacobians)
    weights = weighting(gramian)

    gradients = list[Tensor]()
    while jacobians:
        jacobian = jacobians.popleft()  # get jacobian + dereference it to free memory asap
        gradients.append(torch.tensordot(weights, jacobian, dims=1))

    return gradients, weights


def _compute_gramian_sum(jacobians: deque[Tensor]) -> PSDMatrix:
    gramian = sum([compute_gramian(matrix) for matrix in jacobians])
    return cast(PSDMatrix, gramian)


def _unite_jacobians(jacobians: deque[Tensor]) -> Tensor:
    jacobian_matrices = list[Tensor]()
    while jacobians:
        jacobian = jacobians.popleft()  # get jacobian + dereference it to free memory asap
        jacobian_matrices.append(jacobian.reshape(jacobian.shape[0], -1))
    jacobian_matrix = torch.concat(jacobian_matrices, dim=1)
    return jacobian_matrix


def _disunite_gradient(
    gradient_vector: Tensor,
    tensors: list[TensorWithJac],
) -> list[Tensor]:
    gradient_vectors = gradient_vector.split([t.numel() for t in tensors])
    gradients = [g.reshape(t.shape) for g, t in zip(gradient_vectors, tensors, strict=True)]
    return gradients


def _free_jacs(tensors: Iterable[TensorWithJac]) -> None:
    """
    Deletes the ``.jac`` field of the provided tensors.

    :param tensors: The tensors whose ``.jac`` fields should be cleared.
    """

    for t in tensors:
        del t.jac
