# Contributing to TorchJD

This document explains how to contribute to TorchJD. Please use issues or discussions to communicate
with maintainers before implementing major changes.

## Installation

To work with TorchJD, we suggest you to use [uv](https://docs.astral.sh/uv/). While this is not
mandatory, we only provide installation steps with this tool. You can install it by following their
[installation documentation](https://docs.astral.sh/uv/getting-started/installation/). We also
suggest to use VSCode with the `Python`, `ty` and `ruff` extensions (without `Pylance`).

1) Pre-requisites: Use `uv` to install a Python version compatible with TorchJD and to pin it to the
  `torchjd` folder. From the root of the `torchjd` repo, run:
   ```bash
   uv python install 3.14.0
   uv python pin 3.14.0
   ```

2) Create a virtual environment and install the project in it. From the root of `torchjd`, run:
   ```bash
   uv venv
   CC=gcc uv pip install --python-version=3.14 -e '.[full]' --group check --group doc --group test --group plot
   ```
   If you want to install PyTorch with a different CUDA version (this could be required depending on
   your GPU), you'll need to specify and extra index. For instance, for CUDA 12.6, run:
      ```bash
   uv venv
   CC=gcc uv pip install --python-version=3.14 -e '.[full]' --group check --group doc --group test --group plot --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu126
   ```

   We also advise using `UV_NO_SYNC=1` to prevent `uv` from syncing all the time. This is because by
   default, it tries to resolve libraries compatible with the whole range of Python versions
   supported by TorchJD, but in reality, we just need an installation compatible with the currently
   used Python version. That's also why we specify `--python-version=3.14` when running
   `uv pip install`. To follow that recommendation, add the following line to your `.bashrc`:
   ```bash
   export UV_NO_SYNC=1
   ```
   and start a new terminal. The alternative is to use the `--no-sync` flag whenever you run a pip
   command that would normally sync (like `uv run`).

3) Install pre-commit:
   ```bash
   uv run pre-commit install
   ```

> [!TIP]
> If you're running into issues when `uv` tries to compile `ecos`, make sure that `gcc` is
> installed. Alternatively, you can try to install `clang` or try to use some older Python version
> (3.12) for which `ecos` has provided compiled packages (the list is accessible
> [here](https://pypi.org/project/ecos/#files)).

> [!TIP]
> The Python version that you should specify in your IDE is `<path-to-torchjd>/.venv/bin/python`.

> [!TIP]
> In the following commands, you can get rid of the `uv run` prefix if you activate the `venv`
> created by `uv`, using `source .venv/bin/activate` from the root of `torchjd`. This will, however,
> only work in the current terminal until it is closed.


## Clean reinstallation

If you want to update all dependencies or just reinstall from scratch, run the following command
from the root of `torchjd`:
```bash
rm -rf .venv
rm uv.lock
uv venv
CC=gcc uv pip install --python-version=3.14 -e '.[full]' --group check --group doc --group test --group plot
uv run pre-commit install
```

## Running tests
   - To verify that your installation was successful, and that unit tests pass, run:
     ```bash
     uv run pytest tests/unit
     ```

   - To also run the unit tests that are marked as slow, add the `--runslow` flag:
    ```bash
    uv run pytest tests/unit --runslow
    ```

   - If you have access to a cuda-enabled GPU, you should also check that the unit tests pass on it:
     ```bash
     CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTEST_TORCH_DEVICE=cuda:0 uv run pytest tests/unit
     ```

   - To check that the usage examples from docstrings and `.rst` files are correct, we test their
   behavior in `tests/doc`. To run these tests, do:
     ```bash
     uv run pytest tests/doc
     ```

  - To compute the code coverage locally, you should run the unit tests and the doc tests together,
  with the `--cov` flag:
    ```bash
    uv run pytest tests/unit tests/doc --cov=src
    ```

## Building the documentation locally
   - From the `docs` folder, run:
     ```bash
     uv run make html
     ```
   - You can then open `docs/build/html/index.html` with a web browser.
   - Sometimes, you need to manually delete the built documentation before generating it. To do
   this, from the `docs` folder, run:
     ```bash
     uv run make clean
     ```

## Type checking

We use [ty](https://docs.astral.sh/ty/) for type-checking. If you're on VSCode, we recommend using
the `ty` extension. You can also run it from the root of the repo with:
```bash
uv run ty check
```

## Development guidelines

The following guidelines should help preserve a good code quality in TorchJD. Contributions that do
not respect these guidelines will still be greatly appreciated but will require more work from
maintainers to be merged.

### Documentation

Most source Python files in TorchJD have a corresponding `.rst` in `docs/source`. Please make sure
to add such a documentation entry whenever you add a new public module. In most cases, public
classes should contain a usage example in their docstring. We also ask contributors to add an entry
in the `[Unreleased]` section of the changelog whenever they make a change that may affect users (we
do not report internal changes). If this section does not exist yet (right after a release), you
should create it.

### Testing

We ask contributors to implement the unit tests necessary to check the correctness of their
implementations. Besides, whenever usage examples are provided, we require the example's code to be
tested in `tests/doc`. We require a very high code coverage for newly introduced sources (~95-100%).
To ensure that the tensors generated during the tests are on the right device, you have to use the
partial functions defined in `tests/utils/tensors.py` to instantiate tensors. For instance, instead
of
```python
import torch
a = torch.ones(3, 4)
```
use
```python
from utils.tensors import ones_
a = ones_(3, 4)
```

This will automatically call `torch.ones` with `device=DEVICE`.
If the function you need does not exist yet as a partial function in `tensors.py`, add it.
Lastly, when you create a model or a random generator, you have to move them manually to the right
device (the `DEVICE` defined in `device.py`).
```python
import torch
from torch.nn import Linear
from device import DEVICE

model = Linear(3, 4).to(device=DEVICE)
rng = torch.Generator(device=DEVICE)
```
You may also use a `ModuleFactory` to make the modules on `DEVICE` automatically.

### Coding

We try to keep the quality of the codebase as high as possible. Even if this slows down development
in the short term, it helps a lot in the long term. To make the code easy to understand and to
maintain, we try to keep it simple, and to stick as much as possible to the
[SOLID principles](https://en.wikipedia.org/wiki/SOLID). Try to preserve the existing coding style
of the library when adding new sources. Also, please make sure that new modules are imported by the
`__init__.py` file of the package they are located into. This makes them easier to import for the
user.

## Adding a new aggregator

Mathematically, an aggregator is a mapping $\mathcal A: \mathbb R^{m \times n} \to \mathbb R^n$. In
the context of Jacobian descent, it is used to reduce a Jacobian matrix into a vector that can be
used to update the parameters. In TorchJD, an `Aggregator` subclass should be a faithful
implementation of a mathematical aggregator.

> [!WARNING]
> Currently, we only accept aggregators that have the same interface as the `Aggregator` base class.
> We do not support stateful aggregators yet, so the proposed aggregators **must be immutable**.

> [!NOTE]
> Before working on the implementation of a new aggregator, please contact us via an issue or a
> discussion: in many cases, we have already thought about it, or even started an implementation.

## Deprecation

To deprecate some public functionality, make it raise a `DeprecationWarning`. A test should also be
added in `tests/units/test_deprecations.py`, ensuring that this warning is issued.

## Release

*This section is addressed to maintainers.*

To release a new `torchjd` version, you have to:
- Make sure that all tests, including those on cuda, pass (for this, you need access to a machine
  that has a cuda-enabled GPU).
- Make sure that all important changes since the last release have been reported in the
  `[Unreleased]`
  section at the top of the changelog.
- Add a `[X.Y.Z] - yyyy-mm-dd` header in the changelog just below the `[Unreleased]` header.
- Change the version in `pyproject.toml`.
- Make a pull request with those changes and merge it.
- Make a draft of the release on GitHub (click on `Releases`, then `Draft a new release`, then fill
  the details).
- Publish the release (click on  `Publish release`). This should trigger the deployment of the new
  version on PyPI and the building and deployment of the documentation on github-pages.
- Check that the new version is correctly deployed to PyPI, that it is installable and that it
  works.
- Check that the documentation has been correctly deployed.
