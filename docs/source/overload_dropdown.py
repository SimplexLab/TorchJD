"""Sphinx extension that shows ``@overload`` stubs in a collapsible dropdown.

For every function or method whose docstring is processed by autodoc, this
extension calls :func:`typing.get_overloads` to retrieve the registered
overload stubs and prepends a ``sphinx_design`` ``.. dropdown::`` block to
the docstring so readers can inspect each overload signature without cluttering
the main documentation page.

Requires Python 3.11+ (for :func:`typing.get_overloads`) and the
``sphinx-design`` Sphinx extension.
"""

import inspect
import sys
import types
import typing
from collections.abc import Sequence

from sphinx.application import Sphinx


def _format_annotation(ann: object) -> str:
    """Format a type annotation using short (unqualified) class names."""
    if ann is None or ann is type(None):
        return "None"

    # X | Y union types (Python 3.10+ syntax)
    if isinstance(ann, types.UnionType):
        return " | ".join(_format_annotation(a) for a in ann.__args__)

    origin = getattr(ann, "__origin__", None)
    if origin is not None:
        args: Sequence[object] = getattr(ann, "__args__", None) or ()

        if origin is typing.Union:
            return " | ".join(_format_annotation(a) for a in args)

        origin_name: str = (
            getattr(origin, "__name__", None) or getattr(origin, "_name", None) or repr(origin)
        )
        if args:
            args_str = ", ".join(_format_annotation(a) for a in args)
            return f"{origin_name}[{args_str}]"
        return origin_name

    name: str | None = getattr(ann, "__name__", None)
    if name:
        return name

    return str(ann)


def _format_param(param: inspect.Parameter) -> str:
    """Format a single parameter without separators (``/`` or ``*``)."""
    if param.kind == inspect.Parameter.VAR_POSITIONAL:
        name_part = f"*{param.name}"
    elif param.kind == inspect.Parameter.VAR_KEYWORD:
        name_part = f"**{param.name}"
    else:
        name_part = param.name

    if param.annotation is not inspect.Parameter.empty:
        name_part = f"{name_part}: {_format_annotation(param.annotation)}"

    if param.default is not inspect.Parameter.empty:
        name_part = f"{name_part} = {param.default!r}"

    return name_part


def _build_param_strs(sig: inspect.Signature) -> tuple[list[str], str]:
    """Return ``(param_strs, return_str)`` for *sig* with short type names.

    *param_strs* includes the bare ``/`` and ``*`` separators as individual
    entries so that callers can join them however they like.
    """
    items = [(p.kind, _format_param(p)) for p in sig.parameters.values()]

    param_strs: list[str] = []
    for i, (kind, s) in enumerate(items):
        prev_kind = items[i - 1][0] if i > 0 else None

        # Insert '/' after the last positional-only parameter.
        if (
            prev_kind == inspect.Parameter.POSITIONAL_ONLY
            and kind != inspect.Parameter.POSITIONAL_ONLY
        ):
            param_strs.append("/")

        # Insert bare '*' before the first keyword-only parameter when there
        # is no preceding *args parameter.
        if kind == inspect.Parameter.KEYWORD_ONLY and prev_kind not in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            param_strs.append("*")

        param_strs.append(s)

    # Trailing '/' when every parameter is positional-only.
    if items and items[-1][0] == inspect.Parameter.POSITIONAL_ONLY:
        param_strs.append("/")

    ret = sig.return_annotation
    return_str = f" -> {_format_annotation(ret)}" if ret is not inspect.Signature.empty else ""

    return param_strs, return_str


def _overload_code_lines(func_name: str, overload_func: object, indent: str) -> list[str]:
    """Return source-style lines for one overload, indented by *indent*."""
    try:
        sig = inspect.signature(overload_func)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return [f"{indent}@overload", f"{indent}def {func_name}(...): ...", ""]

    param_strs, return_str = _build_param_strs(sig)

    lines = [
        f"{indent}@overload",
        f"{indent}def {func_name}(",
        *[f"{indent}    {p}," for p in param_strs],
        f"{indent}){return_str}: ...",
    ]
    lines.append("")
    return lines


def _process_docstring(
    _app: Sphinx,
    what: str,
    name: str,
    obj: object,
    _options: object,
    lines: list[str],
) -> None:
    """Prepend an *Overloads* dropdown to docstrings of overloaded functions."""
    if what not in ("function", "method"):
        return
    if sys.version_info < (3, 11):
        return

    try:
        overloads = typing.get_overloads(obj)
    except Exception:
        return

    if not overloads:
        return

    func_name = name.split(".")[-1]

    # A plain container div is used instead of a sphinx_design dropdown so
    # that the toggle lives in the function-signature <dt> (added by JS) rather
    # than as a separate collapsible card below it.
    # Indentation: container content → 3 spaces; code-block content → 6 spaces.
    dropdown: list[str] = [".. container:: overloads-block", "", "   .. code-block:: python", ""]
    for overload_func in overloads:
        dropdown.extend(_overload_code_lines(func_name, overload_func, indent="      "))
    dropdown.append("")

    for i, line in enumerate(dropdown):
        lines.insert(i, line)


def setup(app: Sphinx) -> dict[str, object]:
    app.connect("autodoc-process-docstring", _process_docstring)
    return {"version": "0.1", "parallel_read_safe": True}
