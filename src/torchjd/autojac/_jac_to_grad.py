from collections.abc import Iterable
from typing import overload

import torch
from torch import Tensor

from torchjd.aggregation import Aggregator
from torchjd.aggregation._aggregator_bases import WeightedAggregator

from ._accumulation import TensorWithJac, accumulate_grads, is_tensor_with_jac
from ._utils import check_consistent_first_dimension


@overload
def jac_to_grad(
    tensors: Iterable[Tensor],
    /,
    aggregator: WeightedAggregator,
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
            tensor([0.5],  0.5])

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
        return None

    jacobians = [t.jac for t in tensors_]

    check_consistent_first_dimension(jacobians, "tensors.jac")

    if not retain_jac:
        _free_jacs(tensors_)

    jacobian_matrix = _unite_jacobians(jacobians)
    if isinstance(aggregator, WeightedAggregator):
        weights = aggregator.weighting(jacobian_matrix)
        gradient_vector = weights @ jacobian_matrix
    else:
        weights = None
        gradient_vector = aggregator(jacobian_matrix)
    gradients = _disunite_gradient(gradient_vector, tensors_)
    accumulate_grads(tensors_, gradients)
    return weights


def _unite_jacobians(jacobians: list[Tensor]) -> Tensor:
    jacobian_matrices = [jacobian.reshape(jacobian.shape[0], -1) for jacobian in jacobians]
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
