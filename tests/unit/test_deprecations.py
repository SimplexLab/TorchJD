import pytest


# deprecated since 2025-08-18
def test_deprecate_imports_from_torchjd() -> None:
    with pytest.deprecated_call():
        from torchjd import backward  # noqa: F401

    with pytest.deprecated_call():
        from torchjd import mtl_backward  # noqa: F401

    with pytest.raises(ImportError):
        from torchjd import something_that_does_not_exist  # noqa: F401
