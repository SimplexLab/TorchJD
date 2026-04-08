import torch
from torch import Tensor


def fix_zero_basis_vectors(physical: Tensor, basis: Tensor) -> tuple[Tensor, Tensor]:
    """
    Remove basis vectors that are all 0 and sum the corresponding elements in the physical tensor.
    """

    are_vectors_zero = (basis == 0).all(dim=0)

    if not are_vectors_zero.any():
        return physical, basis

    zero_column_indices = torch.arange(len(are_vectors_zero))[are_vectors_zero].tolist()
    physical = physical.sum(dim=zero_column_indices)
    basis = basis[:, ~are_vectors_zero]
    return physical, basis
