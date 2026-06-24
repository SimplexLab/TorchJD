---
name: prepare-release
description: Prepares a torchjd package release by verifying the changelog, README, bumping the version in pyproject.toml, and opening a release PR. Use when a maintainer asks to prepare a release or to release.
---

# Prepare TorchJD Release

This skill covers the pre-merge steps of the release process.

**For agents:** invoke as `/prepare-release X.Y.Z` (e.g. `/prepare-release 0.16.0`).
If no version is provided, read the current version from `pyproject.toml` and ask the user
what the new version should be before proceeding.

**For humans:** follow the numbered steps below in your terminal.
Replace `X.Y.Z` with the version you are releasing and `yyyy-mm-dd` with today's date.

---

## Instructions

### Step 1: Determine the version

Read `pyproject.toml` to find the current `version` field under `[project]`. The new release
version is either provided as an argument or must be confirmed with the user before continuing.

Verify it follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Step 2: Verify CHANGELOG

Read `CHANGELOG.md` and compare it against the commits done since the last version release. Verify that no user-facing change was forgotten, and that all were correctly added under the [Unreleased] section.

### Step 3: Check README for interface changes

Read `CHANGELOG.md` and identify any entries in the `[Unreleased]` section that affect the public
interface (new or removed classes, functions, or arguments; changed signatures or behavior).

If such entries exist, read `README.md` and verify it accurately describes the current interface.

If a table listing methods exists in `README.md`, add to it any newly added method of the corresponding type.

If anything else seems missing, report that to the user.

### Step 4: Run the unit tests

CRITICAL: Do not skip this step, even partially.

Run the unit tests and confirm they all pass both on CPU and GPU (you need a CUDA-enabled GPU for this):

```bash
uv run pytest tests/unit
PYTEST_TORCH_DEVICE=cuda:0 uv run pytest tests/unit
```

If any tests fail, stop and report the failures. Do not proceed to the next steps.

### Step 5: Add the version header to the changelog

In `CHANGELOG.md`, insert a new `## [X.Y.Z] - yyyy-mm-dd` heading immediately after the blank
line that follows `## [Unreleased]`, before the existing subsections:

```diff
 ## [Unreleased]

+## [X.Y.Z] - yyyy-mm-dd
+
 ### Added
```

The `[Unreleased]` section stays in place and will accumulate entries for the next release.
The newly-inserted heading claims the existing content as its own.

### Step 6: Bump the version in `pyproject.toml`

In `pyproject.toml`, update the `version` field under `[project]`:

```diff
-version = "A.B.C"
+version = "X.Y.Z"
```

### Step 7: Open the release PR

Checkout a new branch release-vX.Y.Z, stage changes, commit, then open a pull request targeting `main`.
Return the PR URL when done.

### Step 8: Create a draft release on GitHub

CRITICAL: The release you'll create should ALWAYS be a draft. Never even suggest to make the real release. A maintainer will manually release the draft if it seems ready.

The command should be:
```
gh release create vX.Y.Z --draft --title vX.Y.Z --notes "<insert notes here>"
```

To write the actual release notes, look at what is done in recent releases and suggest the new notes. Make it short. Also, prompt the user for a good-looking emoji (propose a list) to use in the main section of the release notes.

---
