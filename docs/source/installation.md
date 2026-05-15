# Installation

```{include} ../../README.md
:start-after: <!-- start installation -->
:end-before: <!-- end installation -->
```

Note that `torchjd` requires Python 3.10, 3.11, 3.12, 3.13 or 3.14 and `torch>=2.0`.

Some aggregators have additional dependencies that are not included by default when installing
`torchjd`. The following table lists the optional dependency groups and the aggregators they enable:

| Group | Classes | Dependencies | Install command |
|-------|---------|--------------|-----------------|
| `quadprog_projector` | QuadprogProjector (used in UPGrad and DualProj) | `numpy` (BSD-3-Clause), `quadprog` (GPL-2.0+), `qpsolvers` (LGPL-3.0) | `pip install "torchjd[quadprog_projector]"` |
| `cagrad` | CAGrad | `numpy` (BSD-3-Clause), `cvxpy` (Apache-2.0) | `pip install "torchjd[cagrad]"` |
| `nash_mtl` | NashMTL | `numpy` (BSD-3-Clause), `cvxpy` (Apache-2.0), `ecos` (GPL-3.0) | `pip install "torchjd[nash_mtl]"` |

To install `torchjd` with all of its optional dependencies, you can also use:
```
pip install "torchjd[full]"
```
