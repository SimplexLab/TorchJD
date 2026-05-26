from pytest import raises
from torch import Tensor

from torchjd.scalarization import Scalarizer


def test_cannot_instantiate_abstract_base() -> None:
    with raises(TypeError):
        Scalarizer()  # type: ignore[abstract]


class _Identity(Scalarizer):
    def forward(self, losses: Tensor, /) -> Tensor:
        return losses.sum()


def test_default_representations() -> None:
    s = _Identity()
    assert repr(s) == "_Identity()"
    assert str(s) == "_Identity"
