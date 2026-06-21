# Scalarization

This package implements the `Scalarizer`s: objects that reduce a tensor of values (typically a
vector of losses) into a single scalar optimizable with a standard `loss.backward()`.

This file is for contributors working on scalarizers. For the list of available scalarizers and their
full API, see [torchjd.org](https://torchjd.org/latest/docs/scalarization/).

## The abstraction

A scalarizer captures a single decision: **how to collapse a vector of values into one scalar to
minimize**. It operates purely on those values: it has no notion of the losses, tasks, or model they
come from, which is why its input is named `values` and not `losses`. It is the value-level
counterpart of an aggregator, which makes the same decision at the gradient level. Everything after
it (backpropagation, the optimizer step) is standard PyTorch.

Concretely, it subclasses `Scalarizer` (in [`_scalarizer_base.py`](_scalarizer_base.py)) and
implements one method:

```python
def forward(self, values: Tensor, /) -> Tensor:
    ...
```

- **Any shape in, scalar out:** it reduces over *all* elements of `values` (scalar, vector, matrix,
  higher-dim) into a single scalar.
- **Pure and differentiable:** the output depends only on `values` and the configured parameters, so
  that `scalarizer(values).backward()` produces the gradient.

## Adding one

A new scalarizer is a class plus the files that register it. Mirror an existing scalarizer of the
same kind:

- `_<name>.py`: the class.
- `__init__.py`: the import and an `__all__` entry.
- `docs/source/docs/scalarization/<name>.rst`: the docs page, added to the `index.rst` toctree.
- `tests/unit/scalarization/test_<name>.py`: the tests.
- `CHANGELOG.md`: an entry under `[Unreleased]`.

## State

Most scalarizers are stateless. Keep yours stateless unless the method genuinely needs state (learned
weights, a loss history). When it does:

- **Subclass `Stateful`** (`from torchjd._mixins import Stateful`) and implement `reset()` to restore
  the initial state.
- **Keep `forward` self-contained.** Do not hide cross-call state or side effects inside it. When the
  method must carry information between calls, expose it through an explicit, named method and
  document the protocol (e.g. a per-epoch `step()`, or an `update()` after the optimizer step).
- **`nn.Parameter` vs buffer:** trainable state is an `nn.Parameter`; non-trained tensors that must
  move with `.to()` are registered with `register_buffer`.

Randomness is not state: a scalarizer may draw fresh randomness on each call (like the random
baseline) without being `Stateful`. There is no stochastic mixin; it just uses the global torch RNG,
so document the behavior and let users seed it with `torch.manual_seed`.

## Things to be careful about

- **Determinism and side effects:** the output should depend only on `values`, the configured
  parameters, and (if the method is intentionally random) the global RNG. Any state change must be
  deliberate, explicit, and undone by `reset()`.
- **Numerical stability:** keep the reduction finite on the edges of its domain (log-sum-exp
  centering, an eps under a norm or in a denominator, etc.), and explain any value shift in a comment
  and a `.. note::`.
- **Hyperparameters:** when a coefficient has no single good value across problems, make it required
  rather than guessing a default, and validate it in `__init__`.
- **Shape validation:** check parameter shapes against `values` at call time and raise `ValueError`.
- **Preconditions:** if the method is undefined on some inputs, document it in a `.. note::` and lock
  it with a test (e.g. assert `nan` propagates rather than being silently clamped).
