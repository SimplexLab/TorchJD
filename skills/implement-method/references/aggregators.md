# Aggregators Reference

This document describes the architecture, conventions, and patterns for implementing a new
aggregator in `torchjd.aggregation`. Read this alongside the existing implementations
(`_cr_mogm.py`, `_modo.py`, `_sdmgrad.py`, `_excess_mtl.py`, `_gradvac.py`, `_nash_mtl.py`)
before writing any code.

---

## What is an Aggregator?

An `Aggregator` maps a Jacobian matrix `J ∈ ℝ^{m×n}` (m tasks × n parameters) to a single
gradient vector `g ∈ ℝ^n`. It answers: *given the per-task gradients, which single direction
should we update the model parameters in?*

Most aggregators work by computing a weight vector `λ ∈ ℝ^m` and returning `λ @ J`. This is
the `WeightedAggregator` pattern.

---

## Class Hierarchy

```
nn.Module
└── Aggregator                         # ABC: validates input, calls forward()
    ├── WeightedAggregator             # forward = weighting(J) @ J
    └── GramianWeightedAggregator      # forward = gramian_weighting(J @ J^T) @ J
                                       # (pre-computes the Gramian before calling weighting)

Weighting[_T]                          # ABC: takes a statistic _T, returns weights [m]
├── _MatrixWeighting                   # _T = Matrix  → takes raw Jacobian J
└── _GramianWeighting                  # _T = PSDMatrix → takes Gramian J @ J^T
```

**Key rule:** if your method needs coordinate-wise information (e.g. per-element gradient
history like ExcessMTL) or a cross-Jacobian product `J_1 @ J_2^T` (like MoDo, SDMGrad),
use `_MatrixWeighting`. If your method only needs task-task inner products (the Gramian),
use `_GramianWeighting`.

---

## The Paired Class Convention

Almost every algorithm ships as **two classes**:

1. `FooWeighting(_MatrixWeighting or _GramianWeighting, ...)` — the core computation.
   Prefixed with `_` to signal if it is private. Takes either `Matrix` or `PSDMatrix` as input.
2. `Foo(WeightedAggregator or GramianWeightedAggregator, ...)` — the public-facing aggregator
   wrapping the weighting.

**Exception:** if the method is a *modifier* that wraps any existing weighting (like CR-MOGM),
ship only the `*Weighting` class. Do not create a convenience aggregator — the user composes
it themselves with `WeightedAggregator` or `GramianWeightedAggregator`.

---

## Stateful Aggregators

If the weighting maintains state across calls (e.g. EMA of weights, accumulated gradient
history, warm-started weights), inherit from `Stateful` and implement `reset()`.

```python
from torchjd._mixins import Stateful

class _FooWeighting(_MatrixWeighting, Stateful, _NonDifferentiable):
    def __init__(self, ...):
        super().__init__()
        # Register ALL state tensors as buffers — never as plain Python attributes.
        # register_buffer ensures .to(device) moves them correctly.
        self.register_buffer("_my_state", None)  # None = lazily initialized
        self._state_key: tuple[...] | None = None  # plain attribute, not a tensor

    def reset(self) -> None:
        """Clears all state so the next forward starts fresh."""
        self._my_state = None
        self._state_key = None
```

**State key:** use `(m, dtype, device)` when state shape depends only on `m` (number of
tasks). Use `(m, n, dtype, device)` when state shape is `[m, n]` (e.g. ExcessMTL's
`_grad_sum`). Auto-reset state when the key changes — never raise an error.

**Lazy initialisation:** since `m` (and sometimes `n`) is only known at `forward` time,
initialize state tensors in `_ensure_state`, not in `__init__`.

```python
def _ensure_state(self, matrix: Matrix) -> None:
    key = (matrix.shape[0], matrix.dtype, matrix.device)
    if self._state_key == key and self._my_state is not None:
        return
    m = matrix.shape[0]
    self._my_state = matrix.new_zeros(m)  # or new_full, etc.
    self._state_key = key
```

---

## Non-Differentiable Weightings

If the weights are computed from detached statistics (no gradient should flow through the
weighting), inherit from `_NonDifferentiable`:

```python
from torchjd.aggregation._mixins import _NonDifferentiable
```

This is the case for most stateful aggregators (GradVac, NashMTL, CR-MOGM, MoDo, SDMGrad,
ExcessMTL). The `_NonDifferentiable` wraps the `__call__` in a `torch.no_grad` context, so it's useless to detach the input matrix in such a method.

---

## Property-Based Validation

All constructor parameters must be exposed as properties with validating setters. This
allows safe mutation after construction and gives immediate, clear error messages.

```python
@property
def alpha(self) -> float:
    return self._alpha

@alpha.setter
def alpha(self, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"Attribute `alpha` must be in [0, 1]. Found alpha={value!r}."
        )
    self._alpha = value
```

Common constraints:
- Step sizes / learning rates: `> 0`
- Momentum: `in [0, 1)`
- Regularisation coefficients: `>= 0`
- Iteration counts: `>= 1` (int)
- Preference vectors: `ndim == 1`, non-negative, sums to 1

---

## Extra Inputs via Setters

If the method needs information beyond the Jacobian matrix (e.g. raw loss values, auxiliary
statistics), expose it via a setter rather than adding arguments to `forward`. This keeps the
`forward` signature uniform across all weightings. CONTRIBUTING.md explicitly anticipates this
pattern.

```python
def set_losses(self, losses: Tensor) -> None:
    """Must be called before each forward with the current per-task losses."""
    self._losses = losses.detach()
```

The same principle applies to any other extra input (e.g. task-specific learning rates,
reference gradients). Document clearly in the docstring that the setter must be called before
each training step.

---

## `__repr__`

Override `__repr__` to show all hyperparameters. Do not override `__str__` — `Aggregator`
already defines `__str__` to return just the class name, and `Weighting` inherits
`nn.Module.__repr__` as its `__str__`. Concrete example:

```python
def __repr__(self) -> str:
    return (
        f"{self.__class__.__name__}("
        f"alpha={self.alpha!r}, "
        f"rho={self.rho!r})"
    )
```

---

## Docstring Usage Examples

If the method has **non-standard usage** (e.g. requires double-sampling, a `set_losses` call
before each forward, or manual state management), add a `.. testcode::` example in the
docstring showing a realistic training step. Methods with standard usage (construct, call with
a Jacobian matrix, done) should **not** add an example — the class hierarchy and parameter
descriptions are sufficient.

Non-standard examples include:
- Cross-batch `A = J_1 @ J_2.T` input (MoDo, SDMGrad) — show both `autojac.jac` calls and
  how to pass `A`.
- `set_losses` requirement (e.g. GradNorm) — show the `weighting.set_losses(losses)` call
  immediately before the aggregation step.

---

## Attribution

If any code is adapted from an external implementation, add:
1. A comment at the top of `_foo.py`:
   ```python
   # Partly adapted from https://github.com/... — MIT License, Copyright (c) ...
   # See NOTICES for the full license text.
   ```
2. An entry in the `NOTICES` file following the existing template.

---

## Files to Create or Modify

| File | Action |
|---|---|
| `src/torchjd/aggregation/_foo.py` | Create — the implementation |
| `src/torchjd/aggregation/__init__.py` | Add import + `__all__` entry (alphabetical) |
| `tests/unit/aggregation/test_foo.py` | Create — mirror `test_modo.py` or `test_excess_mtl.py` |
| `docs/source/docs/aggregation/foo.rst` | Create — mirror `modo.rst` |
| `docs/source/docs/aggregation/index.rst` | Add `foo.rst` to toctree (alphabetical) |
| `CHANGELOG.md` | Add entry under `[Unreleased] → Added` |
| `NOTICES` | Add entry if code is adapted from an external source |

---

## Tests

Mirror `tests/unit/aggregation/test_modo.py` (for matrix-weighting) or
`tests/unit/aggregation/test_excess_mtl.py` (for stateful + set_losses). At minimum cover:

- `test_representations` — verify `repr(...)` string exactly
- `test_expected_structure_*` — use `assert_expected_structure` from `_asserts.py`,
  parametrize over `typical_matrices + scaled_matrices` from `_inputs.py`
- `test_reset_restores_first_step_behavior` — call → call → `reset()` → call; third == first
- setter tests — `*_accepts_valid` and `*_rejects_*` for every hyperparameter
- `test_output_lies_on_simplex` — returned weights sum to 1 and are ≥ 0
- `test_update_recurrence` — manually verify the formula for one step
- `test_two_consecutive_steps` — verify warm-start carry-over if stateful
- `test_changing_m_auto_resets` — state resets when number of tasks changes
- `test_non_differentiable` — weights have no grad if `_NonDifferentiable`
- `test_zero_columns` — `(m, 0)` input → output shape `(0,)`

**Always use `utils.tensors` partials** (`randn_`, `tensor_`, `ones_`, etc.) — never raw
`torch.*`. This ensures tests run on CUDA/float64 via environment variables.

---

## Examples by Pattern

### Simple stateful wrapper (CR-MOGM)
Wraps any `Weighting[_T]` generically, applies EMA to the output weights.
`_CRMOGMWeighting(Weighting[_T], Stateful)` — generic over `_T`, no convenience aggregator.
State: `_lambda: Tensor | None`, `_initial_weights: Tensor | None`.

### Cross-Gramian / double-sampling (MoDo, SDMGrad)
Receives `A = J_1 @ J_2^T` (not PSD → use `_MatrixWeighting`, NOT `_GramianWeighting`).
Users compute `A` via `autojac.jac` on two independent mini-batches.
State: `_w: Tensor | None` (warm-started weights).
Returns `w + λ·w̃` (SDMGrad) or `w` (MoDo) normalized to sum = 1.

### Stateful with Jacobian-sized state (ExcessMTL)
`_ExcessMTLWeighting(_MatrixWeighting, Stateful, _NonDifferentiable)`.
State key: `(m, n, dtype, device)` — because `_grad_sum` has shape `[m, n]`.
Memory warning: `_grad_sum` is Jacobian-sized, held persistently. Document this.
Uses `set_losses` if loss values are needed (ExcessMTL does not, but GradNorm would).
