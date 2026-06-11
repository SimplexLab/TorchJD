# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). This
changelog does not include internal changes that do not affect the user.

## [Unreleased]

## [0.14.0] - 2026-06-10

### Added

- Added `DWA` (Dynamic Weight Average) from [End-to-End Multi-Task Learning with
  Attention](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf)
  (CVPR 2019), a stateful `Scalarizer` that weights each value by the relative rate at which its
  loss decreased over the two previous epochs. It has no learnable parameters; call its `step()`
  method once per epoch to roll the loss history.
- Added `SDMGradWeighting` from [Direction-oriented Multi-objective Learning: Simple and Provable Stochastic Algorithms](https://arxiv.org/pdf/2305.18409) (NeurIPS 2023). It is a stateful `Weighting` that solves for task weights via a simplex-projected inner loop on a cross-batch matrix `A = J_1 @ J_2.T` (computed from two independent mini-batches using `autojac.jac`), with a direction-oriented regularizer pulling the descent direction toward a preference direction.
- Added `IMTL-L` (the loss-balancing variant of Impartial Multi-Task Learning) from [Towards
  Impartial Multi-Task Learning](https://openreview.net/pdf?id=IMPnRXEWpvr) (ICLR 2021), a stateful
  `Scalarizer` that learns a per-task scale `s_i` and combines the values as
  `Σ (exp(s_i) · L_i − s_i)`.
- Added `UW` (Uncertainty Weighting) from [Multi-Task Learning Using Uncertainty to Weigh Losses
  for Scene Geometry and
  Semantics](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf),
  a `Scalarizer` that combines the values using learned per-task uncertainties. It is the first
  stateful, trainable scalarizer: its log-variances are an `nn.Parameter` that must be passed to
  the optimizer.

## [0.13.0] - 2026-06-07

### Added

- Added `STCH` from [Smooth Tchebycheff Scalarization for Multi-Objective
  Optimization](https://openreview.net/pdf?id=m4dO5L6eCp), a `Scalarizer` that combines the input
  tensor of values into a smooth approximation of their (weighted, shifted) maximum.
- Added `MoDoWeighting` from [Three-Way Trade-Off in Multi-Objective Learning: Optimization, Generalization and Conflict-Avoidance](https://www.jmlr.org/papers/volume25/23-1287/23-1287.pdf) (JMLR 2024). It is a stateful `Weighting` that maintains task weights across calls via a simplex-projected gradient step on a cross-batch matrix `G = J_1 @ J_2.T`, computed from two independent mini-batches using `autojac.jac`.
- Added `GeometricMean` (also known as GLS) studied in [MultiNet++: Multi-Stream Feature
  Aggregation and Geometric Loss Strategy for Multi-Task
  Learning](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf),
  a `Scalarizer` that returns the geometric mean of the input tensor of values.

### Changed

- **BREAKING**: Moved the `Stateful` mixin from `torchjd.aggregation` to the top-level `torchjd`
  namespace, so it can be shared between the aggregation and scalarization packages. Import it as
  `torchjd.Stateful` instead of `torchjd.aggregation.Stateful`.

## [0.12.0] - 2026-05-28

### Added

- Added a new `torchjd.scalarization` package providing the abstract `Scalarizer` base class and
  the concrete implementations `Constant`, `Mean`, `Random`, and `Sum`. These baselines simply
  combine losses into a scalar that can be optimized with a standard backward pass, making them
  useful for comparison with JD-based methods.
- Added `FairGrad` and `FairGradWeighting` from [Fair Resource Allocation in Multi-Task
  Learning](https://arxiv.org/pdf/2402.15638).

### Changed

- **BREAKING**: Removed `numpy`, `quadprog` and `qpsolvers` from the main dependencies of `torchjd`,
  (which now only has `torch` as its main dependency). This makes the base version of `torchjd`
  (installed with `pip install torchjd`) much lighter, but it means that users of `UPGrad` and
  `DualProj` now have to install the new optional dependency group `quadprog_projector` explicitly
  (with e.g. `pip install "torchjd[quadprog_projector]"`).
- **BREAKING**: Removed entirely the concept of generalized Gramians. The `Engine.compute_gramian`
  method now always returns a square matrix of shape `[m, m]`, where `m` is the total number of
  elements of the ``output`` tensor (treating all dimensions uniformly). Previously, an output of
  shape `[m1, m2]` would return a 4D generalized Gramian of shape `[m1, m2, m2, m1]`; it now
  returns a `[m1 * m2, m1 * m2]` matrix.
  This also removes `GeneralizedWeighting` and `Flattening`.
  To update, replace `Flattening(weighting)` with a standard `Weighting` and reshape the resulting
  weight vector yourself:
  ```python
  # Before
  from torchjd.aggregation import Flattening, UPGradWeighting
  weighting = Flattening(UPGradWeighting())
  gramian = engine.compute_gramian(losses)  # shape: [m1, m2, m2, m1]
  weights = weighting(gramian)              # shape: [m1, m2]
  losses.backward(weights)

  # After
  from torchjd.aggregation import UPGradWeighting
  weighting = UPGradWeighting()
  gramian = engine.compute_gramian(losses)           # shape: [m1 * m2, m1 * m2]
  weights = weighting(gramian).reshape(losses.shape) # shape: [m1, m2]
  losses.backward(weights)
  ```

## [0.11.0] - 2026-05-18

### Changed

- **BREAKING**: Removed `norm_eps`, `rep_eps` and `solver` parameters from the `__init__` of
  `UPGrad`, `UPGradWeighting`, `DualProj` and `DualProjWeighting` in favor of a `projector`
  parameter of type `DualConeProjector`. To update:
  ```python
  # Before
  from torchjd.aggregation import UPGrad
  aggregator = UPGrad(norm_eps=1e-6, reg_eps=1e-6, solver="quadprog")

  # After
  from torchjd.aggregation import UPGrad
  from torchjd.linalg import QuadprogProjector
  aggregator = UPGrad(projector=QuadprogProjector(norm_eps=1e-6, reg_eps=1e-6))
  ```
  If you used the default `norm_eps`, `reg_eps` and `solver`, you don't have to change anything and
  you will get the same results.
- `CAGrad`, `CAGradWeighting`, and `NashMTL` are now always importable from `torchjd.aggregation`,
  even when their optional dependencies are not installed. Attempting to instantiate them without the
  required dependencies now raises an `ImportError` with installation instructions, instead of
  raising an `ImportError` at import time.
- Non-differentiable aggregators and weightings (UPGrad, DualProj, PCGrad, GradVac, IMTLG,
  GradDrop, ConFIG, CAGrad, NashMTL) no longer build a computation graph when called on tensors
  that require gradients. Their forward pass is now wrapped in `torch.no_grad()`, so attempting to
  differentiate through them is not possible anymore (while before, it raised a `NonDifferentiableError`).

### Added

- Added `CRMOGMWeighting` from [On the Convergence of Stochastic Multi-Objective Gradient
  Manipulation and Beyond](https://proceedings.neurips.cc/paper_files/paper/2022/file/f91bd64a3620aad8e70a27ad9cb3ca57-Paper-Conference.pdf)
  (NeurIPS 2022). It wraps an existing `Weighting` and stabilises its weights with an exponential
  moving average across calls.
- Added a new abstraction: the `DualConeProjector` abstract base class and its concrete
  `QuadprogProjector` implementation, to do the projection of the gradients onto the dual cone, as
  required in `UPGrad`, and `DualProj`. These classes can be found in `torchjd.linalg`.
- Made `WeightedAggregator` and `GramianWeightedAggregator` public. These abstract base classes are
  now importable from `torchjd.aggregation` and documented. They can be extended to easily implement
  custom `Aggregator`s.
- Made `Matrix` and `PSDMatrix` public. These type annotation classes are now importable from
  `torchjd.linalg` and documented. Users can now subclass `Weighting[Matrix]` or
  `Weighting[PSDMatrix]` to implement custom `Weighting`s.
- Added getters and setters for the constructor parameters of all aggregators and weightings, so
  that they can be changed after initialization. This includes: `pref_vector`,
  `norm_eps` and `reg_eps` in `UPGrad`, `UPGradWeighting`, `DualProj` and `DualProjWeighting`;
  `pref_vector` and `scale_mode` in `AlignedMTL` and `AlignedMTLWeighting`; `c` and `norm_eps` in
  `CAGrad` and `CAGradWeighting`; `pref_vector` in `ConFIG`; `leak` in `GradDrop`, `n_byzantine` and
  `n_selected` in `Krum` and `KrumWeighting`; `epsilon` and `max_iters` in `MGDA` and
  `MGDAWeighting`; `n_tasks`, `max_norm`, `update_weights_every` and `optim_niter` in `NashMTL`;
  `trim_number` in `TrimmedMean`. Setters validate their inputs matching the existing constructor
  checks. Note that setters for `GradVac` and `GradVacWeighting` already existed.

## [0.10.0] - 2026-04-16

### Added

- Added `GradVac` and `GradVacWeighting` from
  [Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models](https://arxiv.org/pdf/2010.05874).
- Documented per-parameter-group aggregation (GradVac-style grouping) in a new Grouping example.

### Fixed

- Added a fallback for when the inner optimization of `NashMTL` fails (which can happen for example
  on the matrix [[0., 0.], [0., 1.]]).

## [0.9.0] - 2026-02-24

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
- **BREAKING**: Made some parameters of the public interface of `torchjd` positional-only or
  keyword-only:
  - `backward`: The `tensors` parameter is now positional-only. Suggested change:
    `backward(tensors=losses)` => `backward(losses)`. All other parameters are now keyword-only.
  - `mtl_backward`: The `tensors` parameter (previously named `losses`) is now positional-only.
    Suggested change: `mtl_backward(losses=losses, features=features)` =>
    `mtl_backward(losses, features=features)`. The `features` parameter remains usable as positional
    or keyword. All other parameters are now keyword-only.
  - `Aggregator.__call__`: The `matrix` parameter is now positional-only. Suggested change:
    `aggregator(matrix=matrix)` => `aggregator(matrix)`.
  - `Weighting.__call__`: The `stat` parameter is now positional-only. Suggested change:
    `weighting(stat=gramian)` => `weighting(gramian)`.
  - `GeneralizedWeighting.__call__`: The `generalized_gramian` parameter is now positional-only.
    Suggested change: `generalized_weighting(generalized_gramian=generalized_gramian)` =>
    `generalized_weighting(generalized_gramian)`.
- Removed several unnecessary memory duplications. This should significantly improve the memory
  efficiency and speed of `autojac`.
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
  Before this change, trying to differentiate through them led to wrong gradients or unclear
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
  - `MGDA` from [Multiple-gradient descent algorithm (MGDA) for multiobjective optimization](https://comptes-rendus.academie-sciences.fr/mathematique/articles/10.1016/j.crma.2012.03.014/).
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
