from torch import Tensor, nn


class NonDifferentiableError(RuntimeError):
    def __init__(self, module: nn.Module) -> None:
        super().__init__(f"Trying to differentiate through {module}, which is not differentiable.")


def raise_non_differentiable_error(module: nn.Module, _: tuple[Tensor, ...] | Tensor) -> None:
    raise NonDifferentiableError(module)
