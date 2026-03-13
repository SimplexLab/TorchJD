#!/bin/bash
# Post-edit hook for Python files. Runs type checking and linting/formatting.
# Usage: post-edit-python.sh <file_path>
# Intended to be called by AI coding agents after editing a python file.
FILE_PATH="$1"

[[ "$FILE_PATH" != *.py ]] && exit 0

cd "$(git rev-parse --show-toplevel)"
uv run ty check
uv run ruff check --fix "$FILE_PATH"
uv run ruff format "$FILE_PATH"
