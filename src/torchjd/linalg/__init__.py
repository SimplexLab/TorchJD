"""
This module provides utilitary linear algebra methods as well as types to represent specific
structural properties.
"""

from torchjd._linalg import (
    DualConeProjector,
    Matrix,
    ProxsuiteProjector,
    PSDMatrix,
    QuadprogProjector,
)

__all__ = [
    "DualConeProjector",
    "Matrix",
    "PSDMatrix",
    "QuadprogProjector",
    "ProxsuiteProjector",
]
