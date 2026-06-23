# Reference: implementing a `Scalarizer`

A `Scalarizer` reduces a tensor of values of any shape into a single scalar — the baseline that
combines *losses* directly (a plain `loss.backward()` then gives the gradient), as opposed to an
`Aggregator` which combines per-loss *gradients*. Base class: `Scalarizer` in
`src/torchjd/scalarization/_scalarizer_base.py`.

**Don't work from this file alone — read the closest existing class end-to-end (its `_*.py` + `.rst`
+ `test_*.py`) and mirror it.** This reference is the map and the non-obvious rules, not a template.

## Contract for the subclass

- Subclasses `Scalarizer` (an `nn.Module`); `forward(self, values: Tensor, /) -> Tensor` returns a
  **0-dim** scalar.
- The parameter is named **`values`** (positional-only), not `losses` — `Scalarizer` is generic
  (maintainer decision). Accepts **any shape** and reduces over all elements (flatten if needed).

## Files to create / edit (new scalarizer `Foo`)

1. `src/torchjd/scalarization/_foo.py` — the class.
2. `src/torchjd/scalarization/__init__.py` — add the import + the `__all__` entry.
3. `docs/source/docs/scalarization/foo.rst` — doc page (mirror `geometric_mean.rst`).
4. `docs/source/docs/scalarization/index.rst` — add `foo.rst` to the `.. toctree::`.
5. `tests/unit/scalarization/test_foo.py` — tests.
6. `CHANGELOG.md` — entry under `[Unreleased] > ### Added`.
7. *(Only if you adapt third-party code)* license header in `_foo.py` + an entry in `NOTICES`.

## Pick the pattern and mirror it

| Pattern | Mirror | File |
|---|---|---|
| Stateless one-liner | `GeometricMean`, `Mean`, `Sum` | `_geometric_mean.py`, `_mean.py`, `_sum.py` |
| Stateless + preference/reference vector | `STCH`, `COSMOS`, `PBI` | `_stch.py`, `_cosmos.py`, `_pbi.py` |
| Stateful, trainable parameter | `UW`, `IMTL-L` | `_uw.py`, `_imtl_l.py` |
| Stateful, non-trainable history buffer | `DWA` | `_dwa.py` |
| Internal optimizer + multi-call protocol | `FAMO` | `_famo.py` |

### Pattern-specific rules (the things not obvious from one file)

- **Trainable** (`UW`/`IMTL-L`): also subclass `Stateful` (`from torchjd._mixins import Stateful`)
  and implement `reset()`. State is an `nn.Parameter`, init to a neutral default (usually `0`), with
  a `shape: int | Sequence[int]` arg (`Foo(3)` → `(3,)`). Validate `values.shape` at call time
  (`ValueError`). The params are in `.parameters()`, so the user passes them to the optimizer — show
  this in a doctest. A trained per-position param makes it **not** permutation-invariant; don't
  assert it. Add a `shape`-aware `__repr__`.
- **History buffer** (`DWA`): **no** `nn.Parameter` (`list(Foo().parameters())` must be empty); hold
  state in a `register_buffer` (moves with `.to()`, can be created lazily from the first input
  shape). Provide an explicit update method (e.g. scheduler-like `step()`); `forward` **detaches**
  weights derived from the state; `reset()` clears the buffer.
- **Internal optimizer / multi-call** (`FAMO`): private `nn.Parameter` (`_w`) with `.grad` cleared
  after each step; a lazily-created internal `torch.optim.Adam`; an `update(new_losses)` method;
  `forward` detaches the weights. Read `_famo.py` before copying.
- **Preference / reference vector** (`STCH`/`COSMOS`/`PBI`): validate shapes at call time
  (`ValueError`, like `Constant`); flatten `weights`/`values`/`reference` in `forward`. `reference`
  (z*) usually defaults to `0`; `weights` is required or uniform per the paper. Watch `nan`-gradient
  footguns — `‖x‖` has a `0/0` grad at `0` (use `sqrt(‖x‖² + eps)`, see `PBI`); cosine needs an
  eps-clamped denominator (use `torch.nn.functional.cosine_similarity`, see `COSMOS`). Lock with a
  test.

## Docstring conventions

- Use a **raw** `r"""` docstring **only** if it contains LaTeX (`:math:` / `.. math::`) so
  backslashes stay single; plain `"""` otherwise.
- Start with the `:class:` cross-ref(s) (`:class:`~torchjd.scalarization.Scalarizer``, plus
  `:class:`~torchjd.Stateful`` if stateful); link the paper by full title + URL.
- Multi-symbol math → a `.. math::` block + a bullet list defining each symbol (not one dense inline
  paragraph; see `STCH`). Document every `:param:`. Add a usage doctest (for stateful methods show
  the optimizer / `step()` / `update()` cadence). Note preconditions in `.. note::` and decide
  whether to enforce (`ValueError`) or let `nan`/`inf` propagate.

## Tests

Mirror `test_geometric_mean.py` (stateless) or `test_uw.py` (stateful). Shared infra in
`tests/unit/scalarization/`: `_inputs.py` (`shapes = [[], [5], [3, 4], [2, 3, 4]]`, `all_inputs`);
`_asserts.py` (`assert_returns_scalar`, `assert_grad_flow`, `assert_permutation_invariant`);
`utils.tensors` helpers (`tensor_`, `rand_`, `randn_`, `ones_`, `zeros_`, `randperm_` — they respect
`PYTEST_TORCH_DEVICE`/`PYTEST_TORCH_DTYPE`; for stateful instances make a `_foo(shape)` helper that
`.to(device=DEVICE, dtype=DTYPE)`, see `test_uw.py`). Cover: `test_value` (hand-checked),
`test_expected_structure` + `test_grad_flow` (parametrized over shapes), `test_permutation_invariant`
**only if** invariant, the documented edge cases/contracts (e.g. assert `nan` propagates on a bad
input so a future clamp can't slip in; a `does_not_raise()`/`raises(ValueError)` shape table; `reset`
clears state; params train / buffer rolls), and `test_representations`.

## CHANGELOG

`- Added `Foo` from [Paper Title](url) (Venue Year), a `Scalarizer` that <one-line description>.`

## Third-party attribution (only if adapting code, e.g. `FAMO`)

Header comment in `_foo.py`: `# Partly adapted from <url> — <License>, Copyright (c) <year>
<author>. # See NOTICES for the full license text.` plus the full license text in `NOTICES`.

## Verify (from repo root)

```bash
uv run pytest tests/unit/scalarization -W error -v          # new tests
uv run pytest tests/unit -W error                           # full unit regression
uv run ruff check && uv run ruff format --check             # lint + format
uv run make doctest -C docs && uv run make clean -C docs && uv run make html -C docs
uv run pre-commit run --all-files
PYTEST_TORCH_DEVICE=cuda:0 uv run pytest tests/unit -W error # GPU (needs CUDA)
```

- If `uv run` re-syncs unexpectedly, prefix with `UV_NO_SYNC=1`. Docs build is strict (`-W -n`), so
  an `.rst` title underline must match its title length.
- `test_dualproj.py::test_permutation_invariant` and `test_upgrad.py::test_permutation_invariant`
  are known flaky off-Linux (~1 float32 ULP, quadprog), pre-existing and unrelated. CI (Linux) is
  the source of truth.
