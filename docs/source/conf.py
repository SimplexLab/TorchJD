# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import os
import sys

import tomli

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

_PATH_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_ROOT = os.path.realpath(os.path.join(_PATH_HERE, "..", ".."))
_PATH_PYPROJECT = os.path.join(_PATH_ROOT, "pyproject.toml")

sys.path.insert(0, _PATH_HERE)

# Read all metadata from pyproject.toml, so that we don't duplicate it.

with open(_PATH_PYPROJECT, mode="rb") as fp:
    pyproject_config = tomli.load(fp)

authors = pyproject_config["project"]["authors"]
author_names = [author["name"] for author in authors]
author_name_emails = [f"{author['name']} <{author['email']}>" for author in authors]

project = pyproject_config["project"]["name"]
copyright = ", ".join(author_names)
author = ", ".join(author_name_emails)
release = pyproject_config["project"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "myst_parser",  # Enables markdown support
    "sphinx_design",  # Enables side to side cards
    "overload_dropdown",  # Shows @overload stubs in a collapsible dropdown
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
templates_path = ["_templates"]

autodoc_member_order = "bysource"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
}

html_favicon = "icons/favicon.ico"
html_theme_options = {
    "light_logo": "logo-light-mode.png",
    "dark_logo": "logo-dark-mode.png",
    "dark_css_variables": {
        "color-problematic": "#eeeeee",  # class names
    },
    "light_css_variables": {
        "color-problematic": "#000000",  # class names
    },
    "sidebar_hide_name": True,
}

html_css_files = ["overload_dropdown.css"]
html_js_files = [
    ("https://stats.torchjd.org/js/script.js", {"data-domain": "torchjd.org", "defer": "defer"}),
    "overload_dropdown.js",
]

html_title = "TorchJD"


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """
    Returns an optional link to the source code of an object defined by its domain and info.

    See https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html#confval-linkcode_resolve
    for more information.
    """

    if domain != "py" or not info["module"]:
        return None

    obj = _get_obj(info)
    file_name = _get_file_name(obj)

    if not file_name:
        return None

    line_str = _get_line_str(obj)
    version_str = _get_version_str()

    link = f"https://github.com/SimplexLab/TorchJD/blob/{version_str}/{file_name}{line_str}"
    return link


def _get_obj(_info: dict[str, str]) -> object:
    module_name = _info["module"]
    full_name = _info["fullname"]
    sub_module = sys.modules.get(module_name)
    obj = sub_module
    for part in full_name.split("."):
        obj = getattr(obj, part)
    # strip decorators, which would resolve to the source of the decorator
    obj = inspect.unwrap(obj)  # type: ignore[arg-type]
    return obj


def _get_file_name(obj: object) -> str | None:
    try:
        file_name = inspect.getsourcefile(obj)  # type: ignore[arg-type]
        if file_name is None:
            return None
        return os.path.relpath(file_name, start=_PATH_ROOT)
    except TypeError:  # This seems to happen when obj is a property
        return None


def _get_line_str(obj: object) -> str:
    source, start = inspect.getsourcelines(obj)  # type: ignore[arg-type]
    end = start + len(source) - 1
    line_str = f"#L{start}-L{end}"
    return line_str


def _get_version_str() -> str:
    try:
        version_str = os.environ["TORCHJD_VERSION"]
    except KeyError:
        version_str = "main"
    return version_str
