from typing import Literal, cast, overload

import torch
from torch import Tensor

from ._matrix import Matrix, PSDMatrix, PSDTensor


@overload
def compute_gramian(t: Tensor) -> PSDMatrix:
    pass


@overload
def compute_gramian(t: Tensor, contracted_dims: Literal[-1]) -> PSDMatrix:
    pass


@overload
def compute_gramian(t: Matrix, contracted_dims: Literal[1]) -> PSDMatrix:
    pass


@overload
def compute_gramian(t: Tensor, contracted_dims: int) -> PSDTensor:
    pass


def compute_gramian(t: Tensor, contracted_dims: int = -1) -> PSDTensor:
    """
    Computes the `Gramian matrix <https://en.wikipedia.org/wiki/Gram_matrix>`_ of the input.

    `contracted_dims` specifies the number of trailing dimensions to contract. If negative,
    it indicates the number of leading dimensions to preserve (e.g., ``-1`` preserves the
    first dimension).
    """

    # Optimization: it's faster to do that than moving dims and using tensordot, and this case
    # happens very often, sometimes hundreds of times for a single jac_to_grad.
    if contracted_dims == -1:
        matrix = t.unsqueeze(1) if t.ndim == 1 else t.flatten(start_dim=1)

        gramian = matrix @ matrix.T

    else:
        contracted_dims = contracted_dims if contracted_dims >= 0 else contracted_dims + t.ndim
        indices_source = list(range(t.ndim - contracted_dims))
        indices_dest = list(range(t.ndim - 1, contracted_dims - 1, -1))
        transposed = t.movedim(indices_source, indices_dest)
        gramian = torch.tensordot(t, transposed, dims=contracted_dims)

    return cast(PSDTensor, gramian)


def normalize(gramian: PSDMatrix, eps: float) -> PSDMatrix:
    """
    Normalizes the gramian `G=AA^T` with respect to the Frobenius norm of `A`.

    If `G=A A^T`, then the Frobenius norm of `A` is the square root of the trace of `G`, i.e., the
    sqrt of the sum of the diagonal elements. The gramian of the (Frobenius) normalization of `A` is
    therefore `G` divided by the sum of its diagonal elements.
    """

    squared_frobenius_norm = gramian.diagonal().sum()
    condition = squared_frobenius_norm < eps

    # Use torch.where rather than a if-else to avoid cuda synchronization.
    output = torch.where(condition, torch.zeros_like(gramian), gramian / squared_frobenius_norm)
    return cast(PSDMatrix, output)


def regularize(gramian: PSDMatrix, eps: float) -> PSDMatrix:
    """
    Adds a regularization term to the gramian to enforce positive definiteness.

    Because of numerical errors, `gramian` might have slightly negative eigenvalue(s). Adding a
    regularization term which is a small proportion of the identity matrix ensures that the gramian
    is positive definite.
    """

    regularization_matrix = eps * torch.eye(
        gramian.shape[0],
        dtype=gramian.dtype,
        device=gramian.device,
    )
    output = gramian + regularization_matrix
    return cast(PSDMatrix, output)
