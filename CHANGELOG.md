# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). This
changelog does not include internal changes that do not affect the user.

## [Unreleased]

### Added

- Added the function `torchjd.autojac.jac`. It's the same as `torchjd.autojac.backward` except that
  it returns the Jacobians as a tuple instead of storing them in the `.jac` fields of the inputs.
  Its interface is analog to that of `torch.autograd.grad`.
- Added a `jac_tensors` parameter to `backward`, allowing to pre-multiply the Jacobian computation
  by initial Jacobians. This enables multi-step chain rule computations and is analogous to the
  `grad_tensors` parameter in `torch.autograd.backward`.
- Added a `grad_tensors` parameter to `mtl_backward`, allowing to use non-scalar `losses` (now
  renamed to `tensors`). This is analogous to the `grad_tensors` parameter of
  `torch.autograd.backward`. When using `scalar` losses, the usage does not change.
- Added a `jac_outputs` parameter to `jac`, allowing to pre-multiply the Jacobian computation by
  initial Jacobians. This is analogous to the `grad_outputs` parameter in `torch.autograd.grad`.
- Added a `scale_mode` parameter to `AlignedMTL` and `AlignedMTLWeighting`, allowing to choose
  between `"min"`, `"median"`, and `"rmse"` scaling.
- Added an attribute `gramian_weighting` to all aggregators that use a gramian-based `Weighting`.
  Usage is still the same, `aggregator.gramian_weighting` is just an alias for the (quite confusing)
  `aggregator.weighting.weighting` field.

### Changed

- **BREAKING**: Removed from `backward` and `mtl_backward` the responsibility to aggregate the
  Jacobian. Now, these functions compute and populate the `.jac` fields of the parameters, and a new
  function `torchjd.autojac.jac_to_grad` should then be called to aggregate those `.jac` fields into
  `.grad` fields.
  This means that users now have more control on what they do with the Jacobians (they can easily
  aggregate them group by group or even param by param if they want), but it now requires an extra
  line of code to do the Jacobian descent step. To update, please change:
  ```python
  backward(losses, aggregator)
  ```
  to
  ```python
  backward(losses)
  jac_to_grad(model.parameters(), aggregator)
  ```
  and
  ```python
  mtl_backward(losses, features, aggregator)
  ```
  to
  ```python
  mtl_backward(losses, features)
  jac_to_grad(shared_module.parameters(), aggregator)
  ```
- **BREAKING**: Made some parameters of the public interface of `torchjd` positional-only or keyword-only:
  - `backward`: The `tensors` parameter is now positional-only. Suggested change: `backward(tensors=losses)` => `backward(losses)`. All other parameters are now keyword-only.
  - `mtl_backward`: The `tensors` parameter (previously named `losses`) is now positional-only. Suggested change: `mtl_backward(losses=losses, features=features)` => `mtl_backward(losses, features=features)`. The `features` parameter remains usable as positional or keyword. All other parameters are now keyword-only.
  - `Aggregator.__call__`: The `matrix` parameter is now positonal-only. Suggested change: `aggregator(matrix=matrix)` => `aggregator(matrix)`.
  - `Weighting.__call__`: The `stat` parameter is now positional-only. Suggested change: `weighting(stat=gramian)` => `weighting(gramian)`.
  - `GeneralizedWeighting.__call__`: The `generalized_gramian` parameter is now positional-only. Suggested change: `generalized_weighting(generalized_gramian=generalized_gramian)` => `generalized_weighting(generalized_gramian)`.
- Removed an unnecessary memory duplication. This should significantly improve the memory efficiency
  of `autojac`.
- Removed an unnecessary internal cloning of gradient. This should slightly improve the memory
  efficiency of `autojac`.
- Increased the lower bounds of the torch (from 2.0.0 to 2.3.0) and numpy (from 1.21.0
  to 1.21.2) dependencies to reflect what really works with torchjd. We now also run torchjd's tests
  with the dependency lower-bounds specified in `pyproject.toml`, so we should now always accurately
  reflect the actual lower-bounds.

## [0.8.1] - 2026-01-07

### Added

- Added `__all__` in the `__init__.py` of packages. This should prevent PyLance from triggering
  warnings when importing from `torchjd`.

## [0.8.0] - 2025-11-13

### Added

- Added the `autogram` package, with the `autogram.Engine`. This is an implementation of Algorithm 3
  from [Jacobian Descent for Multi-Objective Optimization](https://arxiv.org/pdf/2406.16232),
  optimized for batched computations, as in IWRM. Generalized Gramians can also be obtained by using
  the autogram engine on a tensor of losses of arbitrary shape.
- For all `Aggregator`s based on the weighting of the Gramian of the Jacobian, made their
  `Weighting` class public. It can be used directly on a Gramian (computed via the
  `autogram.Engine`) to extract some weights. The list of new public classes is:
  - `Weighting` (abstract base class)
  - `UPGradWeighting`
  - `AlignedMTLWeighting`
  - `CAGradWeighting`
  - `ConstantWeighting`
  - `DualProjWeighting`
  - `IMTLGWeighting`
  - `KrumWeighting`
  - `MeanWeighting`
  - `MGDAWeighting`
  - `PCGradWeighting`
  - `RandomWeighting`
  - `SumWeighting`
- Added `GeneralizedWeighting` (base class) and `Flattening` (implementation) to extract tensors of
  weights from generalized Gramians.
- Added usage example for IWRM with autogram.
- Added usage example for IWRM with partial autogram.
- Added usage example for IWMTL with autogram.
- Added Python 3.14 classifier in pyproject.toml (we now also run tests on Python 3.14 in the CI).

### Changed

- Removed an unnecessary internal reshape when computing Jacobians. This should have no effect but a
  slight performance improvement in `autojac`.
- Revamped documentation.
- Made `backward` and `mtl_backward` importable from `torchjd.autojac` (like it was prior to 0.7.0).
- Deprecated importing `backward` and `mtl_backward` from `torchjd` directly.

## [0.7.0] - 2025-06-04

### Changed

- **BREAKING**: Changed the dependencies of `CAGrad` and `NashMTL` to be optional when installing
  TorchJD. Users of these aggregators will have to use `pip install "torchjd[cagrad]"`, `pip install
  "torchjd[nash_mtl]"` or `pip install "torchjd[full]"` to install TorchJD alongside those
  dependencies. This should make TorchJD more lightweight.
- **BREAKING**: Made the aggregator modules and the `autojac` package protected. The aggregators
  must now always be imported via their package (e.g.
  `from torchjd.aggregation.upgrad import UPGrad` must be changed to
  `from torchjd.aggregation import UPGrad`). The `backward` and `mtl_backward` functions must now
  always be imported directly from the `torchjd` package (e.g.
  `from torchjd.autojac.mtl_backward import mtl_backward` must be changed to
  `from torchjd import mtl_backward`).
- Removed the check that the input Jacobian matrix provided to an aggregator does not contain `nan`,
  `inf` or `-inf` values. This check was costly in memory and in time for large matrices so this
  should improve performance. However, if the optimization diverges for some reason (for instance
  due to a too large learning rate), the resulting exceptions may come from other sources.
- Removed some runtime checks on the shapes of the internal tensors used by the `autojac` engine.
  This should lead to a small performance improvement.

### Fixed

- Made some aggregators (`CAGrad`, `ConFIG`, `DualProj`, `GradDrop`, `IMTLG`, `NashMTL`, `PCGrad`
  and `UPGrad`) raise a `NonDifferentiableError` whenever one tries to differentiate through them.
  Before this change, trying to differentiate through them leaded to wrong gradients or unclear
  errors.

### Added

- Added a `py.typed` file in the top package of `torchjd` to ensure compliance with
  [PEP 561](https://peps.python.org/pep-0561/). This should make it possible for users to use
  [mypy](https://github.com/python/mypy) against the type annotations provided in `torchjd`.

## [0.6.0] - 2025-04-19

### Added

- Added usage example showing how to combine TorchJD with automatic mixed precision (AMP).

### Changed

- Refactored the underlying optimization problem that `UPGrad` and `DualProj` have to solve to
  project onto the dual cone. This should slightly improve the performance and precision of these
  aggregators.
- Refactored internal verifications in the `autojac` engine so that they do not run at runtime
  anymore. This should minimally improve the performance and reduce the memory usage of `backward`
  and `mtl_backward`.
- Refactored internal typing in the `autojac` engine so that fewer casts are made and so that code
  is simplified. This should slightly improve the performance of `backward` and `mtl_backward`.
- Improved the implementation of `ConFIG` to be simpler and safer when normalizing vectors. It
  should slightly improve the performance of `ConFIG` and minimally affect its behavior.
- Simplified the normalization of the Gramian in `UPGrad`, `DualProj` and `CAGrad`. This should
  slightly improve their performance and precision.

### Fixed

- Fixed an issue with `backward` and `mtl_backward` that could make the ordering of the columns of
  the Jacobians non-deterministic, and that could thus lead to slightly non-deterministic results
  with some aggregators.
- Removed arbitrary exception handling in `IMTLG` and `AlignedMTL` when the computation fails. In
  practice, this fix should only affect some matrices with extremely large values, which should
  not usually happen.
- Fixed a bug in `NashMTL` that made it fail (due to a type mismatch) when `update_weights_every`
  was more than 1.

## [0.5.0] - 2025-02-01

### Added

- Added new aggregator `ConFIG` from [ConFIG: Towards Conflict-free Training of Physics
Informed Neural Networks](https://arxiv.org/pdf/2408.11104).

## [0.4.2] - 2025-01-30

### Added

- Added Python 3.13 classifier in pyproject.toml (we now also run tests on Python 3.13 in the CI).

## [0.4.1] - 2025-01-02

### Fixed

- Fixed a bug introduced in v0.4.0 that could cause `backward` and `mtl_backward` to fail with some
  tensor shapes.

## [0.4.0] - 2025-01-02 [YANKED]

### Changed

- Changed how the Jacobians are computed when calling `backward` or `mtl_backward` with
  `parallel_chunk_size=1` to not rely on `torch.autograd.vmap` in this case. Whenever `vmap` does
  not support something (compiled functions, RNN on cuda, etc.), users should now be able to avoid
  using `vmap` by calling `backward` or `mtl_backward` with `parallel_chunk_size=1`.

- Changed the effect of the parameter `retain_graph` of `backward` and `mtl_backward`. When set to
  `False`, it now frees the graph only after all gradients have been computed. In most cases, users
  should now leave the default value `retain_graph=False`, no matter what the value of
  `parallel_chunk_size` is. This will reduce the memory overhead.

### Added

- RNN training usage example in the documentation.

## [0.3.1] - 2024-12-21

### Changed

- Improved the performance of the graph traversal function called by `backward` and `mtl_backward`
  to find the tensors with respect to which differentiation should be done. It now visits every node
  at most once.

## [0.3.0] - 2024-12-10

### Added

- Added a default value to the `inputs` parameter of `backward`. If not provided, the `inputs` will
  default to all leaf tensors that were used to compute the `tensors` parameter. This is in line
  with the behavior of
  [torch.autograd.backward](https://pytorch.org/docs/stable/generated/torch.autograd.backward.html).
- Added a default value to the `shared_params` and to the `tasks_params` arguments of
  `mtl_backward`. If not provided, the `shared_params` will default to all leaf tensors that were
  used to compute the `features`, and the `tasks_params` will default to all leaf tensors that were
  used to compute each of the `losses`, excluding those used to compute the `features`.
- Note in the documentation about the incompatibility of `backward` and `mtl_backward` with tensors
  that retain grad.

### Changed

- **BREAKING**: Changed the name of the parameter `A` to `aggregator` in `backward` and
  `mtl_backward`.
- **BREAKING**: Changed the order of the parameters of `backward` and `mtl_backward` to make it
  possible to have a default value for `inputs` and for `shared_params` and `tasks_params`,
  respectively. Usages of `backward` and `mtl_backward` that rely on the order between arguments
  must be updated.
- Switched to the [PEP 735](https://peps.python.org/pep-0735/) dependency groups format in
  `pyproject.toml` (from a `[tool.pdm.dev-dependencies]` to a `[dependency-groups]` section). This
  should only affect development dependencies.

### Fixed

- **BREAKING**: Added a check in `mtl_backward` to ensure that `tasks_params` and `shared_params`
  have no overlap. Previously, the behavior in this scenario was quite arbitrary.

## [0.2.2] - 2024-11-11

### Added

- PyTorch Lightning integration example.
- Explanation about Jacobian descent in the README.

### Fixed

- Made the dependency on [ecos](https://github.com/embotech/ecos-python) explicit in pyproject.toml
  (before `cvxpy` 1.16.0, it was installed automatically when installing `cvxpy`).

## [0.2.1] - 2024-09-17

### Changed

- Removed upper cap on `numpy` version in the dependencies. This makes `torchjd` compatible with
  the most recent numpy versions too.

### Fixed

- Prevented `IMTLG` from dividing by zero during its weight rescaling step. If the input matrix
  consists only of zeros, it will now return a vector of zeros instead of a vector of `nan`.

## [0.2.0] - 2024-09-05

### Added

- `autojac` package containing the backward pass functions and their dependencies.
- `mtl_backward` function to make a backward pass for multi-task learning.
- Multi-task learning example.

### Changed

- **BREAKING**: Moved the `backward` module to the `autojac` package. Some imports may have to be
  adapted.
- Improved documentation of `backward`.

### Fixed

- Fixed wrong tensor device with `IMTLG` in some rare cases.
- **BREAKING**: Removed the possibility of populating the `.grad` field of a tensor that does not
  expect it when calling `backward`. If an input `t` provided to backward does not satisfy
  `t.requires_grad and (t.is_leaf or t.retains_grad)`, an error is now raised.
- **BREAKING**: When using `backward`, aggregations are now accumulated into the `.grad` fields
  of the inputs rather than replacing those fields if they already existed. This is in line with the
  behavior of `torch.autograd.backward`.

## [0.1.0] - 2024-06-22

### Added

- Basic project structure.
- `aggregation` package:
  - `Aggregator` base class to aggregate Jacobian matrices.
  - `AlignedMTL` from [Independent Component
      Alignment for Multi-Task Learning](
      https://openaccess.thecvf.com/content/CVPR2023/papers/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.pdf).
  - `CAGrad` from [Conflict-Averse Gradient Descent for Multi-task
      Learning](https://arxiv.org/pdf/2110.14048.pdf).
  - `Constant` to aggregate with constant weights.
  - `DualProj` adapted from [Gradient Episodic
      Memory for Continual Learning](https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf).
  - `GradDrop` from [Just Pick a Sign: Optimizing Deep
      Multitask Models with Gradient Sign Dropout](https://arxiv.org/pdf/2010.06808.pdf).
  - `IMTLG` from [Towards Impartial Multi-task Learning](https://discovery.ucl.ac.uk/id/eprint/10120667/).
  - `Krum` from [Machine Learning with Adversaries: Byzantine
      Tolerant Gradient Descent](https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf).
  - `Mean` to average the rows of the matrix.
  - `MGDA` from [Multiple-gradient descent algorithm (MGDA) for multiobjective optimization](https://www.sciencedirect.com/science/article/pii/S1631073X12000738/pdf?md5=2622857e4abde98b6f7ddc8a13a337e1&pid=1-s2.0-S1631073X12000738-main.pdf>).
  - `NashMTL` from [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017.pdf).
  - `PCGrad` from [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782.pdf).
  - `Random` from [Reasonable Effectiveness of Random Weighting: A
      Litmus Test for Multi-Task Learning](https://arxiv.org/pdf/2111.10603.pdf).
  - `Sum` to sum the rows of the matrix.
  - `TrimmedMean` from [Byzantine-Robust Distributed Learning: Towards
      Optimal Statistical Rates](https://proceedings.mlr.press/v80/yin18a/yin18a.pdf).
  - `UPGrad` from [Jacobian Descent for Multi-Objective Optimization](https://arxiv.org/pdf/2406.16232).
- `backward` function to perform a step of Jacobian descent.
- Documentation of the public API and of some usage examples.
- Tests:
  - Unit tests.
  - Documentation tests.
  - Plotting utilities to verify qualitatively that aggregators work as expected.
