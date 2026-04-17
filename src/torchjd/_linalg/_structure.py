from dataclasses import dataclass

import torch

from torchjd._linalg import Matrix


@dataclass
class Structure:
    m: int
    device: torch.device
    dtype: torch.dtype


def extract_structure(matrix: Matrix) -> Structure:
    return Structure(m=matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
