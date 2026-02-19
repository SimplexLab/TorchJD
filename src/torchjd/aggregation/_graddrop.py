from collections.abc import Callable

import torch
from torch import Tensor

from torchjd._linalg import Matrix

from ._aggregator_bases import Aggregator
from ._utils.non_differentiable import raise_non_differentiable_error


def _identity(P: Tensor) -> Tensor:
    return P


class GradDrop(Aggregator):
    """
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` that applies the gradient combination
    steps from GradDrop, as defined in lines 10 to 15 of Algorithm 1 of `Just Pick a Sign:
    Optimizing Deep Multitask Models with Gradient Sign Dropout
    <https://arxiv.org/pdf/2010.06808.pdf>`_.

    :param f: The function to apply to the Gradient Positive Sign Purity. It should be monotically
        increasing. Defaults to identity.
    :param leak: The tensor of leak values, determining how much each row is allowed to leak
        through. Defaults to None, which means no leak.
    """

    def __init__(self, f: Callable = _identity, leak: Tensor | None = None):
        if leak is not None and leak.dim() != 1:
            raise ValueError(
                "Parameter `leak` should be a 1-dimensional tensor. Found `leak.shape = "
                f"{leak.shape}`.",
            )

        super().__init__()
        self.f = f
        self.leak = leak

        # This prevents computing gradients that can be very wrong.
        self.register_full_backward_pre_hook(raise_non_differentiable_error)

    def forward(self, matrix: Matrix, /) -> Tensor:
        self._check_matrix_has_enough_rows(matrix)

        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return torch.zeros(matrix.shape[1], dtype=matrix.dtype, device=matrix.device)

        leak = self.leak if self.leak is not None else torch.zeros_like(matrix[:, 0])

        P = 0.5 * (torch.ones_like(matrix[0]) + matrix.sum(dim=0) / matrix.abs().sum(dim=0))
        fP = self.f(P)
        U = torch.rand(P.shape, dtype=matrix.dtype, device=matrix.device)

        vector = torch.zeros_like(matrix[0])
        for i in range(len(matrix)):
            M_i = (fP > U) * (matrix[i] > 0) + (fP < U) * (matrix[i] < 0)
            vector += (leak[i] + (1 - leak[i]) * M_i) * matrix[i]

        return vector

    def _check_matrix_has_enough_rows(self, matrix: Tensor) -> None:
        n_rows = matrix.shape[0]
        if self.leak is not None and n_rows != len(self.leak):
            raise ValueError(
                f"Parameter `matrix` should be a matrix of exactly {len(self.leak)} rows (i.e. the "
                f"number of leak scalars). Found `matrix` of shape `{matrix.shape}`.",
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(f={repr(self.f)}, leak={repr(self.leak)})"

    def __str__(self) -> str:
        if self.leak is None:
            leak_str = ""
        else:
            leak_str = f"([{', '.join([f'{l_:.2f}'.rstrip('0') for l_ in self.leak])}])"
        return f"GradDrop{leak_str}"
