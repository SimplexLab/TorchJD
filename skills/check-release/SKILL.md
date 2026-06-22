---
name: check-release
description: Verifies that a TorchJD release was published correctly by checking the docs site, installing from PyPI, and smoke-testing newly added classes. Use after a release has been merged and published.
---

# Check TorchJD Release

This skill verifies that a release is live and correct after it has been published.

**For agents:** invoke as `/check-release X.Y.Z` (e.g. `/check-release 0.16.0`).
If no version is provided, read the current version from `pyproject.toml`.

---

## Instructions

### Step 1: Determine the version

Read `pyproject.toml` to find the `version` field under `[project]`. Use the version provided as
an argument, or the one from `pyproject.toml` if none is given.

### Step 2: Identify newly added classes

Read `CHANGELOG.md` and find the `## [X.Y.Z]` section. Extract the names of any newly added
public classes, functions, or methods listed under `### Added`. You will use these in later steps.

### Step 3: Check the docs site

Fetch `https://torchjd.org`.

- Verify that the versions dropdown (or switcher) includes `vX.Y.Z` as an entry.
- Verify that the `stable` entry is present.

If the version entry is missing, report it and stop — the rest of the checks depend on the docs
being live.

### Step 4: Verify the new-version docs contain the newly added classes

For each newly added class or function identified in Step 2, fetch its expected docs page under
`https://torchjd.org/vX.Y.Z/`. Use the URL patterns from similar existing classes found in
`README.md` or by browsing the stable docs (`https://torchjd.org/stable/`) to infer the correct
path (e.g. `https://torchjd.org/vX.Y.Z/docs/aggregation`,
`https://torchjd.org/vX.Y.Z/docs/scalarization`, etc.).

Confirm that each new class/function name appears on the fetched page.

### Step 5: Verify the stable docs also reflect the new version

Fetch the same doc pages under `https://torchjd.org/stable/` and confirm the newly added
classes/functions appear there too (i.e. `stable` points to the new release).

### Step 6: Install torchjd from PyPI in a temp environment

Run the following commands to create an isolated install:

```bash
cd /tmp && mkdir -p test_torchjd_install && cd test_torchjd_install
uv venv && uv pip install torchjd
```

Verify the installed version matches X.Y.Z:

```bash
cd /tmp/test_torchjd_install && uv pip show torchjd
```

If the version is wrong, report it and stop.

### Step 7: Smoke-test the newly added classes

Write a minimal Python script `/tmp/test_torchjd_install/smoke_test.py` that:

- Imports each newly added class or function by its fully-qualified name from `torchjd`.
- Instantiates or calls each one with a minimal valid input (e.g. a small `torch.Tensor`, a dummy
  preference vector, or no arguments if the class takes none).
- Does NOT assert correctness of values — only that the code runs without raising an exception.

Use the existing test suite under `tests/` or the docs pages fetched in Step 4 as a reference for
correct import paths and minimal usage patterns.

Run the script:

```bash
cd /tmp/test_torchjd_install && uv run python smoke_test.py
```

Report the result. If it crashes, show the traceback.

### Step 8: Clean up

```bash
rm -rf /tmp/test_torchjd_install
```

### Step 9: Report

Summarize what was verified:
- Docs site: version dropdown ✓/✗, new-version page ✓/✗, stable page ✓/✗
- PyPI install: version matches ✓/✗
- Smoke test: each newly added class ✓/✗ (list them)

If everything passes, the release is confirmed good. If anything failed, describe what needs
attention.
