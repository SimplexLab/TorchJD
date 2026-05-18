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
| `quadprog_projector` | QuadprogProjector (used in UPGrad and DualProj) | `numpy`, `quadprog`, `qpsolvers` | `pip install "torchjd[quadprog_projector]"` |
| `cagrad` | CAGrad | `numpy`, `cvxpy` | `pip install "torchjd[cagrad]"` |
| `nash_mtl` | NashMTL | `numpy`, `cvxpy`, `ecos` | `pip install "torchjd[nash_mtl]"` |

To install `torchjd` with all of its optional dependencies, you can also use:
```
pip install "torchjd[full]"
```
