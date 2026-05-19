from importlib.util import find_spec

import pytest

from torchjd._mixins import _WithOptionalDeps
from torchjd.aggregation import (
    IMTLG,
    CAGradWeighting,
    GramianWeightedAggregator,
    IMTLGWeighting,
    UPGrad,
    UPGradWeighting,
)
from torchjd.aggregation._nash_mtl import _NashMTLWeighting
from torchjd.aggregation._weighting_bases import _GramianWeighting
from torchjd.linalg import QuadprogProjector


def deps_are_installed(cls: type[_WithOptionalDeps]) -> bool:
    """Returns a boolean indicating whether all of cls dependencies are installed."""

    return all(find_spec(d) is not None for d in cls._REQUIRED_DEPS)


IS_QUADPROG_PROJ_AVAILABLE = deps_are_installed(QuadprogProjector)
IS_CAGRAD_AVAILABLE = deps_are_installed(CAGradWeighting)
IS_NASH_MTL_AVAILABLE = deps_are_installed(_NashMTLWeighting)


def skip_if_deps_not_installed(cls: type[_WithOptionalDeps]) -> None:
    """Skip the tests in the file of this call if not all dependencies of cls are installed."""

    for dependency_name in cls._REQUIRED_DEPS:
        pytest.importorskip(dependency_name)


def base_agg() -> GramianWeightedAggregator:
    """Returns the aggregator we want to run most tests with, depending on availability."""

    return UPGrad() if IS_QUADPROG_PROJ_AVAILABLE else IMTLG()


def base_weighting() -> _GramianWeighting:
    """Returns the _GramianWeighting we want to run most tests with, depending on availability."""

    return UPGradWeighting() if IS_QUADPROG_PROJ_AVAILABLE else IMTLGWeighting()
