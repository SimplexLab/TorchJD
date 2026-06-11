import os
import re
import sys


def generate_redirects(build_dir: str) -> None:
    """
    Generates HTML redirect pages from the old ``docs/`` paths to the new ``reference/`` paths.

    For each page found under ``<build_dir>/reference/``, this creates a corresponding redirect
    HTML file under ``<build_dir>/docs/`` that redirects browser visits to the new location.

    Supports both the ``dirhtml`` and ``html`` Sphinx builders.

    :param build_dir: Path to the build output directory (e.g. ``build/dirhtml`` or ``build/html``).
    """

    reference_dir = os.path.join(build_dir, "reference")
    if not os.path.isdir(reference_dir):
        print(f"No reference directory found at {reference_dir}, skipping redirects.")
        return

    if _is_dirhtml_builder(reference_dir):
        _generate_redirects_dirhtml(build_dir, reference_dir)
    else:
        _generate_redirects_html(build_dir, reference_dir)


def verify_redirects(build_dir: str) -> bool:
    """
    Verifies that all redirect files are correct.

    Checks:
    - Every ``reference/`` page has a corresponding redirect in ``docs/``.
    - Every redirect target resolves to an existing file on disk.
    - No orphaned redirects (redirect files with no matching ``reference/`` page).

    :param build_dir: Path to the build output directory.
    :returns: ``True`` if all checks pass, ``False`` otherwise.
    """

    reference_dir = os.path.join(build_dir, "reference")
    if not os.path.isdir(reference_dir):
        print(f"Error: no reference directory found at {reference_dir}")
        return False

    if _is_dirhtml_builder(reference_dir):
        return _verify_redirects_dirhtml(build_dir, reference_dir)
    return _verify_redirects_html(build_dir, reference_dir)


def _is_dirhtml_builder(reference_dir: str) -> bool:
    upgrad_path = os.path.join(reference_dir, "aggregation", "upgrad")
    return os.path.isdir(upgrad_path)


# ── generation ────────────────────────────────────────────────────────────────────


def _generate_redirects_dirhtml(build_dir: str, reference_dir: str) -> None:
    for root, _dirs, _files in os.walk(reference_dir):
        rel_path = os.path.relpath(root, reference_dir)
        if rel_path == ".":
            rel_path = ""

        depth = rel_path.count(os.sep) + int(rel_path != "")
        relative_prefix = os.sep.join([".."] * (depth + 1))
        if rel_path:
            redirect_target = f"{relative_prefix}/reference/{rel_path}/"
        else:
            redirect_target = f"{relative_prefix}/reference/"

        dest_dir = os.path.join(build_dir, "docs", rel_path)
        os.makedirs(dest_dir, exist_ok=True)
        _write_redirect_file(os.path.join(dest_dir, "index.html"), redirect_target)
        print(f"Redirect: docs/{rel_path}/ -> reference/{rel_path}/")


def _generate_redirects_html(build_dir: str, reference_dir: str) -> None:
    for root, _dirs, files in os.walk(reference_dir):
        rel_dir = os.path.relpath(root, reference_dir)
        if rel_dir == ".":
            rel_dir = ""

        for file in files:
            if not file.endswith(".html"):
                continue
            rel_file = os.path.join(rel_dir, file) if rel_dir else file

            depth = rel_file.count(os.sep) + 1
            relative_prefix = os.sep.join([".."] * depth)
            redirect_target = f"{relative_prefix}/reference/{rel_file}"

            dest_file = os.path.join(build_dir, "docs", rel_file)
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            _write_redirect_file(dest_file, redirect_target)
            print(f"Redirect: docs/{rel_file} -> reference/{rel_file}")


def _write_redirect_file(filepath: str, target: str) -> None:
    canonical_path = target
    while canonical_path.startswith("../"):
        canonical_path = canonical_path[3:]
    canonical_url = f"https://torchjd.org/stable/{canonical_path}"
    content = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        f'    <meta http-equiv="refresh" content="0; url={target}">\n'
        f'    <link rel="canonical" href="{canonical_url}">\n'
        "</head>\n"
        "<body>\n"
        f'    <p>This page has moved to <a href="{target}">{target}</a>.</p>\n'
        "</body>\n"
        "</html>\n"
    )
    with open(filepath, "w") as f:
        f.write(content)


# ── verification ───────────────────────────────────────────────────────────────────

_META_REFRESH_RE = re.compile(r'<meta\s+http-equiv="refresh"\s+content="0;\s*url=([^"]+)"')


def _verify_redirects_dirhtml(build_dir: str, reference_dir: str) -> bool:
    ok = True
    docs_dir = os.path.join(build_dir, "docs")

    # Collect all reference pages (every directory in reference/ is a page).
    ref_pages = set()
    for root, _dirs, _files in os.walk(reference_dir):
        rel_path = os.path.relpath(root, reference_dir)
        if rel_path == ".":
            rel_path = ""
        ref_pages.add(rel_path)

    # Collect all redirect pages.
    redirect_pages = set()
    for root, _dirs, files in os.walk(docs_dir):
        if "index.html" in files:
            rel_path = os.path.relpath(root, docs_dir)
            if rel_path == ".":
                rel_path = ""
            redirect_pages.add(rel_path)

    # Check every reference page has a redirect.
    for page in sorted(ref_pages):
        if page not in redirect_pages:
            print(f"Missing redirect: docs/{page}/")
            ok = False

    # Check no orphaned redirects.
    for page in sorted(redirect_pages):
        if page not in ref_pages:
            print(f"Orphaned redirect: docs/{page}/ (no matching reference page)")
            ok = False

    # Check redirect targets exist.
    for page in sorted(ref_pages & redirect_pages):
        redirect_file = os.path.join(docs_dir, page, "index.html")
        target = _extract_redirect_target(redirect_file)
        if target is None:
            print(f"Broken redirect (no meta refresh): docs/{page}/index.html")
            ok = False
            continue
        resolved = os.path.normpath(os.path.join(os.path.dirname(redirect_file), target))
        if not os.path.isdir(resolved):
            print(f"Broken redirect (target directory not found): docs/{page}/ -> {target}")
            ok = False

    return ok


def _verify_redirects_html(build_dir: str, reference_dir: str) -> bool:
    ok = True
    docs_dir = os.path.join(build_dir, "docs")

    # Collect all reference pages (.html files).
    ref_pages = set()
    for root, _dirs, files in os.walk(reference_dir):
        rel_dir = os.path.relpath(root, reference_dir)
        if rel_dir == ".":
            rel_dir = ""
        for file in files:
            if file.endswith(".html"):
                rel_file = os.path.join(rel_dir, file) if rel_dir else file
                ref_pages.add(rel_file)

    # Collect all redirect pages (.html files).
    redirect_pages = set()
    for root, _dirs, files in os.walk(docs_dir):
        rel_dir = os.path.relpath(root, docs_dir)
        if rel_dir == ".":
            rel_dir = ""
        for file in files:
            if file.endswith(".html"):
                rel_file = os.path.join(rel_dir, file) if rel_dir else file
                redirect_pages.add(rel_file)

    # Check every reference page has a redirect.
    for page in sorted(ref_pages):
        if page not in redirect_pages:
            print(f"Missing redirect: docs/{page}")
            ok = False

    # Check no orphaned redirects.
    for page in sorted(redirect_pages):
        if page not in ref_pages:
            print(f"Orphaned redirect: docs/{page} (no matching reference page)")
            ok = False

    # Check redirect targets exist.
    for page in sorted(ref_pages & redirect_pages):
        redirect_file = os.path.join(docs_dir, page)
        target = _extract_redirect_target(redirect_file)
        if target is None:
            print(f"Broken redirect (no meta refresh): docs/{page}")
            ok = False
            continue
        resolved = os.path.normpath(os.path.join(os.path.dirname(redirect_file), target))
        if not os.path.isfile(resolved):
            print(f"Broken redirect (target not found): docs/{page} -> {target}")
            ok = False

    return ok


def _extract_redirect_target(filepath: str) -> str | None:
    try:
        with open(filepath) as f:
            content = f.read()
    except OSError:
        return None
    match = _META_REFRESH_RE.search(content)
    if match is None:
        return None
    return match.group(1)


# ── cli ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.argv.remove("--check")
        if len(sys.argv) > 1:
            build_dir = sys.argv[1]
        elif os.path.isdir(os.path.join("build", "dirhtml")):
            build_dir = os.path.join("build", "dirhtml")
        else:
            build_dir = os.path.join("build", "html")
        ok = verify_redirects(build_dir)
        sys.exit(0 if ok else 1)

    if len(sys.argv) > 1:
        build_dir = sys.argv[1]
    elif os.path.isdir(os.path.join("build", "dirhtml")):
        build_dir = os.path.join("build", "dirhtml")
    else:
        build_dir = os.path.join("build", "html")
    generate_redirects(build_dir)
