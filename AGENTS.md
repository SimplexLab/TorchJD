- Only generate docstrings for public functions or functions that contain more than 4 lines of code.
- Use the Sphinx style for Python docstrings (e.g. :param my_param: Does something) and never
  include a :returns: key.
- The code you generate should always contain type hints in the function prototypes (including
  return type hints of None), but type hints are not needed when initializing variables.
- We use uv for everything (e.g. we do `uv run python ...` to run some python code, and
  `uv run pytest tests/unit` to run unit tests). Please prefer `uv run python -c ...` over
  `python3 -c ...`
- When you create or modify a code example in a public docstring, always update the corresponding
  doc test in the appropriate file of `tests/doc`. This also applies to any change in an example of
  a `.rst` file, that must be updated in the corresponding test in `tests/doc/test_rst.py`.
- After generating code, please run `uv run ty check`, `uv run ruff check` and `uv run ruff format`.
  Fix any error.
- After changing anything in `src` or in `tests/unit` or `tests/doc`, please identify the affected
  test files in `tests/` and run them with e.g.
  `uv run pytest tests/unit/aggregation/test_upgrad.py -W error`. Fix any error, either in the
  changes you've made or by adapting the tests to the new specifications.
