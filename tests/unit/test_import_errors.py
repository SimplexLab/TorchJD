from pytest import mark, raises
from utils.optional_deps import (
    IS_CAGRAD_AVAILABLE,
    IS_NASH_MTL_AVAILABLE,
    IS_QUADPROG_PROJ_AVAILABLE,
)

from torchjd.aggregation import CAGrad, NashMTL
from torchjd.linalg import QuadprogProjector


@mark.skipif(IS_CAGRAD_AVAILABLE, reason="CAGrad deps are available.")
def test_cagrad_import_error_at_init() -> None:
    with raises(ImportError):
        _ = CAGrad(c=0.5)


@mark.skipif(IS_NASH_MTL_AVAILABLE, reason="NashMTL deps are available.")
def test_nash_mtl_import_error_at_init() -> None:
    with raises(ImportError):
        _ = NashMTL(n_tasks=2)


@mark.skipif(IS_QUADPROG_PROJ_AVAILABLE, reason="QuadprogProjector deps are available.")
def test_quadprog_projector_import_error_at_init() -> None:
    with raises(ImportError):
        _ = QuadprogProjector()
