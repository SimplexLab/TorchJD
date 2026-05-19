from importlib.util import find_spec
from typing import Any


class _WithOptionalDeps:
    """
    Mixin that raises :class:`ImportError` at instantiation time if required optional dependencies
    are not installed.

    Subclasses must define :attr:`_REQUIRED_DEPS` (list of package names to check via
    :func:`importlib.util.find_spec`) and :attr:`_INSTALL_HINT` (appended to the error message).

    .. warning::
        This mixin must appear **first** in the inheritance list so that its :meth:`__init__`
        runs before any base class that uses the optional dependencies.
    """

    _REQUIRED_DEPS: list[str]
    _INSTALL_HINT: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        missing = [name for name in self._REQUIRED_DEPS if find_spec(name) is None]
        if len(missing) != 0:
            raise ImportError(
                f"{self.__class__.__name__} requires {missing} to be installed. "
                f"{self._INSTALL_HINT}"
            )
        super().__init__(*args, **kwargs)
