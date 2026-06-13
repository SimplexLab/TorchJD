from collections.abc import Iterable
from typing import TypeGuard

from torch import Tensor


class TensorWithJac(Tensor):
    """
    Tensor known to have a populated jac field.

    Should not be directly instantiated, but can be used as a type hint and can be casted to.
    """

    jac: Tensor


def is_tensor_with_jac(t: Tensor) -> TypeGuard[TensorWithJac]:
    return hasattr(t, "jac")


def accumulate_jacs(params: Iterable[Tensor], jacobians: Iterable[Tensor]) -> None:
    for param, jac in zip(params, jacobians, strict=True):
        _check_expects_grad(param, field_name=".jac")
        # We that the shape is correct to be consistent with torch, that checks that the grad
        # shape is correct before assigning it.
        if jac.shape[1:] != param.shape:
            raise RuntimeError(
                f"attempting to assign a jacobian of size '{list(jac.shape)}' to a tensor of "
                f"size '{list(param.shape)}'. Please ensure that the tensor and each row of the"
                " jacobian are the same size",
            )

        if is_tensor_with_jac(param):
            param.jac += jac
        else:
            # We do not clone the value to save memory and time, so subsequent modifications of
            # the value of key.jac (subsequent accumulations) will also affect the value of
            # jacobians[key] and outside changes to the value of jacobians[key] will also affect
            # the value of key.jac. So to be safe, the values of jacobians should not be used
            # anymore after being passed to this function.
            #
            # We do not detach from the computation graph because the value can have grad_fn
            # that we want to keep track of (in case it was obtained via create_graph=True).
            param.__setattr__("jac", jac)


def accumulate_grads(params: Iterable[Tensor], gradients: Iterable[Tensor]) -> None:
    for param, grad in zip(params, gradients, strict=True):
        _check_expects_grad(param, field_name=".grad")
        if hasattr(param, "grad") and param.grad is not None:
            param.grad += grad
        else:
            param.grad = grad


def _check_expects_grad(tensor: Tensor, field_name: str) -> None:
    if not _expects_grad(tensor):
        raise ValueError(
            f"Cannot populate the {field_name} field of a Tensor that does not satisfy:\n"
            "`tensor.requires_grad and (tensor.is_leaf or tensor.retains_grad)`.",
        )


def _expects_grad(tensor: Tensor) -> bool:
    """
    Determines whether a Tensor expects its .grad attribute to be populated.
    See https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf for more information.
    """

    return tensor.requires_grad and (tensor.is_leaf or tensor.retains_grad)
