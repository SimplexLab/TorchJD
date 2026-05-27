import torch
from pytest import mark
from torch import Tensor
from utils.contexts import fork_rng
from utils.tensors import ones_, tensor_

from torchjd.scalarization import Random

from ._asserts import assert_grad_flow, assert_returns_scalar
from ._inputs import typical_inputs


@mark.parametrize("losses", typical_inputs)
def test_expected_structure(losses: Tensor) -> None:
    assert_returns_scalar(Random(), losses)


@mark.parametrize("losses", typical_inputs)
def test_grad_flow(losses: Tensor) -> None:
    assert_grad_flow(Random(), losses)


def test_deterministic_under_seed() -> None:
    losses = tensor_([1.0, 2.0, 3.0, 4.0])
    scalarizer = Random()
    with fork_rng(seed=0):
        a = scalarizer(losses)
    with fork_rng(seed=0):
        b = scalarizer(losses)
    torch.testing.assert_close(a, b)


def test_weights_sum_to_one() -> None:
    # If all losses equal 1, then sum(weights * losses) == 1 when weights sum to 1.
    losses = ones_((5,))
    torch.testing.assert_close(Random()(losses), tensor_(1.0))


def test_representations() -> None:
    s = Random()
    assert repr(s) == "Random()"
    assert str(s) == "Random"
