# Installation

```{include} ../../README.md
:start-after: <!-- start installation -->
:end-before: <!-- end installation -->
```

Note that `torchjd` requires Python 3.10, 3.11, 3.12, 3.13 or 3.14 and `torch>=2.0`.

Some aggregators have additional dependencies that are not included by default when installing
`torchjd`. The following table lists the optional dependency groups and the aggregators they enable:

Group | Classes | Dependencies | Install command |
|-----|---------|--------------|-----------------|
| `quadprog_projector` | {class}`~torchjd.linalg.QuadprogProjector` (used in {class}`~torchjd.aggregation.UPGrad` and {class}`~torchjd.aggregation.DualProj`) | [numpy](https://github.com/numpy/numpy), [quadprog](https://github.com/quadprog/quadprog), [qpsolvers](https://github.com/qpsolvers/qpsolvers) | `pip install "torchjd[quadprog_projector]"` |
| `cagrad` | {class}`~torchjd.aggregation.CAGrad` | [numpy](https://github.com/numpy/numpy), [cvxpy](https://github.com/cvxpy/cvxpy/) | `pip install "torchjd[cagrad]"` |
| `nash_mtl` | {class}`~torchjd.aggregation.NashMTL` | [numpy](https://github.com/numpy/numpy), [cvxpy](https://github.com/cvxpy/cvxpy/), [ecos](https://github.com/embotech/ecos) | `pip install "torchjd[nash_mtl]"` |

To install `torchjd` with all of its optional dependencies, you can also use:
```
pip install "torchjd[full]"
```
