# Installation

```{include} ../../README.md
:start-after: <!-- start installation -->
:end-before: <!-- end installation -->
```

Note that `torchjd` requires Python 3.10, 3.11, 3.12, 3.13 or 3.14 and `torch>=2.0`.

Some aggregators (CAGrad and Nash-MTL) have additional dependencies that are not included by default
when installing `torchjd`. To install them, you can use:
```
pip install "torchjd[cagrad]"
```
```
pip install "torchjd[nash_mtl]"
```

To install `torchjd` with all of its optional dependencies, you can also use:
```
pip install "torchjd[full]"
```
