"""
This module provides type annotation classes representing tensors with specific structural
properties.
"""

from torchjd._linalg import (
    DualConeProjector,
    Matrix,
    PSDMatrix,
    QuadprogProjector,
)

__all__ = [
    "DualConeProjector",
    "Matrix",
    "PSDMatrix",
    "QuadprogProjector",
]
