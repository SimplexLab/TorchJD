from abc import ABC, abstractmethod
from typing import cast

from torch import Tensor
from torch.utils._pytree import PyTree

from torchjd._linalg import Matrix, PSDMatrix, compute_gramian
from torchjd.autogram._jacobian_computer import JacobianComputer


class GramianComputer(ABC):
    @abstractmethod
    def __call__(
        self,
        rg_outputs: tuple[Tensor, ...],
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
    ) -> PSDMatrix | None:
        """Compute what we can for a module and optionally return the gramian if it's ready."""

    @abstractmethod
    def track_forward_call(self) -> None:
        """Track that the module's forward was called. Necessary in some implementations."""

    @abstractmethod
    def reset(self) -> None:
        """Reset state if any. Necessary in some implementations."""


class JacobianBasedGramianComputer(GramianComputer, ABC):
    def __init__(self, jacobian_computer: JacobianComputer) -> None:
        self.jacobian_computer = jacobian_computer


class JacobianBasedGramianComputerWithCrossTerms(JacobianBasedGramianComputer):
    """
    Stateful JacobianBasedGramianComputer that waits for all usages to be counted before returning
    the gramian.
    """

    def __init__(self, jacobian_computer: JacobianComputer) -> None:
        super().__init__(jacobian_computer)
        self.remaining_counter = 0
        self.summed_jacobian: Matrix | None = None

    def reset(self) -> None:
        self.remaining_counter = 0
        self.summed_jacobian = None

    def track_forward_call(self) -> None:
        self.remaining_counter += 1

    def __call__(
        self,
        rg_outputs: tuple[Tensor, ...],
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
    ) -> PSDMatrix | None:
        """Compute what we can for a module and optionally return the gramian if it's ready."""

        jacobian_matrix = self.jacobian_computer(rg_outputs, grad_outputs, args, kwargs)

        if self.summed_jacobian is None:
            self.summed_jacobian = jacobian_matrix
        else:
            self.summed_jacobian = cast(Matrix, self.summed_jacobian + jacobian_matrix)

        self.remaining_counter -= 1

        if self.remaining_counter == 0:
            gramian = compute_gramian(self.summed_jacobian)
            del self.summed_jacobian
            return gramian
        return None
