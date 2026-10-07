"""
This module provides utility linear algebra methods as well as types to represent specific
structural properties.
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
