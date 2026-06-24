---
name: implement-method
description: Implements a new method (scalarizer or aggregator) in TorchJD, starting from the research produced by the research-method skill and following the established file-by-file conventions. Use when a contributor wants to add the actual implementation of a scalarizer or aggregator that has already been investigated and listed in the tracking issues.
---

# Implement new method

This skill implements a new method by recovering its research, comparing the paper against the
existing implementations, settling the non-standard parts of its interface, and producing the full
set of TorchJD files (class, docs, tests, changelog) that match the established conventions.

It is the companion of the `research-method` skill: that one investigates a method and records a row
in a tracking issue; this one turns that row into a merged implementation.

**For agents:** invoke as `/implement-method method-name (paper-name)` (e.g.
`/implement-method stch (Smooth Tchebycheff Scalarization for Multi-objective Optimization)`).
If no method name is provided, ask the user for the name of the method and the title of the paper.

**For humans:** follow the numbered steps below to guide your development.

---

## Instructions

### Step 1: Recover the research context

Determine whether the method should be a **scalarizer** or an **aggregator**, then read everything
the `research-method` skill already found about it:

- Scalarizers are tracked in https://github.com/SimplexLab/TorchJD/issues/667, aggregators in
  https://github.com/SimplexLab/TorchJD/issues/665. Fetch the relevant issue and find the row for
  this method. Read every column:
  - **Ref** — the paper (open it; you will need the exact equations / algorithm).
  - **Stateful** — whether and how the method holds state.
  - **Existing implementations** — links to the official repo (if any) and the best-known
    third-party ones (LibMTL, libmoon, pymoo, ...), ideally with the exact file(s) and line(s).
  - **Special Remarks** — may link to a full research write-up (e.g. a `claude.ai` share produced by
    `research-method`). Read it if present.
- The most valuable inputs are the **non-standard interface aspects** uncovered during research:
  statefulness, trainable parameters, randomness, warm-up / history buffers, statistics beyond the
  `forward` values (e.g. per-task losses for an aggregator), and preconditions. If these are not
  fully captured in the issue, **ask the user to share the `research-method` findings** before
  continuing. Do not guess them.

If the method is not in the tracking issue yet, run `research-method` first.

### Step 2: Load the implementation reference for this method type

Read only the reference matching the method type, to keep context focused:

- **Scalarizer** → read `references/scalarizers.md`.
- **Aggregator** → read `references/aggregators.md`.

Each reference lists the exact files to create/edit and the TorchJD-specific conventions, with the
closest existing methods to mirror.

### Step 3: Compare the paper with the existing implementations

Always do this — it is the step we invariably end up needing. Read the relevant equations / the
algorithm box in the paper, then read the official and best-known third-party implementations at the
exact files/lines from the tracking row.

Reconcile any discrepancies between them. The ones that most often bite:

- **Minimization vs maximization.** TorchJD minimizes losses; much MOO/evolutionary work is written
  for maximization, with the minimization form buried in a footnote. Find it, and check the sign of
  every reference / ideal-point subtraction.
- **Normalization.** A direction or weight vector may be normalized (`w / ‖w‖`) in the code but not
  the paper, or vice versa.
- **Dead arguments.** An impl may accept a parameter (e.g. a reference point) yet silently ignore it.
- **Droppable terms.** An `abs` / `clamp` / `max(0, ·)` in the paper may be unnecessary under the
  method's preconditions (e.g. non-negative weights); drop it only with a justification.
- **Other:** an extra factor, an init value, a stabilization / epsilon trick.

Decide which to follow, note **why**, and surface the disagreement to the user — the implementation
should be faithful to a clearly-stated source, not an unexplained blend.

### Step 4: Settle the interface and design decisions

Using the research findings (Step 1) and the comparison (Step 3), map each non-standard aspect onto
the closest existing pattern from the reference loaded in Step 2 (statefulness, trainable parameters,
an internal optimizer, a preference/reference vector, ...). Then settle, for any method type:

- **Preconditions** (e.g. positivity): enforce them (raise `ValueError`) or only document them, and
  how `nan`/`inf` should propagate.
- Which constructor arguments are **required vs optional**, and their **defaults**.

List the non-standard parts and your proposed handling, and **confirm the design with the user
before writing code.** This is where most of the maintainer review happens, so settle it up front.

### Step 5: Implement the method

Follow the file-by-file checklist in the reference loaded at Step 2. Match the style, naming, and
conventions of the closest existing method. If you adapt code from a third-party repository, add the
license header to the source file and an entry to `NOTICES` (see the reference).

### Step 6: Verify

Run the checks listed in the reference (unit tests with `-W error`, lint, and the docs
build/doctest). GPU tests require a CUDA device; if you cannot run them, provide the exact commands
for the user to run on their GPU and report back the results.

### Step 7: Self-review the code you produced

Before opening anything, re-read your own diff against the requirements and improve what can be
improved. Check that:

- The class follows the closest existing method's conventions (the reference's checklist): correct
  base class(es), `forward(self, values, /)` returning a 0-dim scalar, shape validation, `reset()`
  for stateful methods, a correct `__repr__`, and the docstring conventions (`r"""` only with LaTeX,
  `:class:` cross-ref, `.. math::` + bullet list, a usage doctest, `:param:` for each argument).
- The design decisions settled in Step 4 are actually reflected in the code, and any discrepancy
  between the paper and the existing implementations (Step 3) is resolved deliberately, with a
  comment or docstring note where it is non-obvious.
- The tests cover the documented edge cases and contracts, not just the happy path.
- All six files are present and consistent (class, `__init__.py`, `.rst`, toctree, test,
  `CHANGELOG.md`), plus `NOTICES` + a license header if you adapted code.

Apply the fixes you find, then re-run the relevant checks from Step 6.

### Step 8: Open a draft PR

Create a new branch, commit, and open a **draft** pull request targeting `main`, following the
repository's PR conventions (a `CHANGELOG.md` entry under `[Unreleased] > ### Added`; when asked for
a PR description, output raw GitHub-flavored markdown in a fenced code block, with GitHub math syntax
`$...$` / `$$...$$` and no em dashes). Keep it a draft so the contributor can read the code
themselves before requesting maintainer review. Return the PR URL when done.

---
