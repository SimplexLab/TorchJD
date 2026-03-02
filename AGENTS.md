# AGENTS.md - TorchJD Development Guide

This file contains guidelines for agentic coding agents working in this repository.

## Running Commands

### Using uv
We use [uv](https://docs.astral.sh/uv/) for all Python operations. Always use `uv run` prefix:
```bash
uv run python ...        # Run Python code
uv run pytest ...        # Run tests
uv run ty check ...      # Type checking
uv run ruff check ...    # Linting
uv run ruff format ...   # Formatting
```

### Running Tests

Run all unit tests:
```bash
uv run pytest tests/unit
```

Run a single test file:
```bash
uv run pytest tests/unit/aggregation/test_upgrad.py -W error
```

Run a single test function:
```bash
uv run pytest tests/unit/aggregation/test_upgrad.py::test_function_name -W error
```

Run tests with slow tests included:
```bash
uv run pytest tests/unit --runslow
```

Run doc tests (examples in docstrings and .rst files):
```bash
uv run pytest tests/doc
```

Run tests on CUDA (requires GPU):
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTEST_TORCH_DEVICE=cuda:0 uv run pytest tests/unit
```

Run tests with coverage:
```bash
uv run pytest tests/unit tests/doc --cov=src
```

### Linting and Type Checking

Run type checking:
```bash
uv run ty check
```

Run linting:
```bash
uv run ruff check
```

Run formatting:
```bash
uv run ruff format
```

Run all checks together:
```bash
uv run ty check && uv run ruff check && uv run ruff format
```

### Building Documentation

From the `docs` folder:
```bash
uv run make html
```

Clean build:
```bash
uv run make clean
```

## Code Style Guidelines

### Docstrings

- Only generate docstrings for public functions or functions with more than 4 lines of code
- Use Sphinx style (`:param my_param: Does something`) - never use `:returns:` key
- Include usage examples in docstrings for public classes
- Always update corresponding doc tests in `tests/doc/` when modifying examples

### Type Hints

- Always include type hints in function prototypes (including `-> None`)
- Do not include type hints when initializing local variables
- Use `ty` for type checking

### Imports

- Use `combine-as-imports = true` (configured in pyproject.toml)
- Order imports: standard library, third-party, local
- Use absolute imports (e.g., `from torchjd.aggregation import UPGrad`)

### Formatting

- Line length: 100 characters (enforced by ruff)
- Quote style: double quotes
- Run `uv run ruff format` before committing

### Naming Conventions

- Classes: `PascalCase` (e.g., `class UPGrad`)
- Functions/methods: `snake_case` (e.g., `def jac_to_grad`)
- Constants: `UPPER_SNAKE_CASE`
- Private functions: prefix with `_` (e.g., `_internal_func`)
- Private modules: prefix with `_` (e.g., `_utils`)

### Error Handling

- Use appropriate exception types from Python standard library or PyTorch
- Provide clear error messages with context
- Validate inputs at function boundaries
- Use assertions for internal invariants

### Testing

- Use partial tensor functions from `tests/utils/tensors.py` (e.g., `ones_()`, `randn_()`)
- This ensures tensors are on correct device and dtype automatically
- Manually move models/rng to DEVICE (defined in `settings.py`)
- Use `ModuleFactory` for creating modules on correct device
- Mark slow tests with `@pytest.mark.slow`
- Run affected tests after changes: `uv run pytest tests/unit/<module> -W error`

### Project Structure

```
src/torchjd/
├── __init__.py           # Public API exports
├── autojac/              # Jacobian computation engine
│   ├── _backward.py
│   ├── _jac.py
│   ├── _jac_to_grad.py
│   ├── _mtl_backward.py
│   └── _transform/      # Internal transform implementations
├── autogram/            # Gramian-based engine
│   ├── _engine.py
│   └── ...
├── aggregation/         # Aggregators and weightings
│   ├── _upgrad.py
│   ├── _mean.py
│   └── _utils/          # Internal utilities
└── _linalg/             # Linear algebra utilities

tests/
├── unit/                # Unit tests (mirrors src structure)
├── doc/                 # Docstring and rst example tests
└── utils/               # Test utilities (tensors.py, settings.py)
```

### Adding New Aggregators

- Subclass `Aggregator` or `Weighting` base classes
- Aggregators must be **immutable** (no stateful changes)
- Implement the mathematical mapping as documented
- Add corresponding weighting if applicable
- Update `__init__.py` to export the new class

### Deprecation

- Raise `DeprecationWarning` for deprecated public functionality
- Add test in `tests/unit/test_deprecations.py` to verify warning

### Code Quality

- Follow SOLID principles
- Keep code simple and readable
- Prefer explicit over implicit
- Add type hints to all public interfaces
- Document complex algorithms with comments
- Run all checks before submitting: `uv run ty check && uv run ruff check && uv run ruff format`
