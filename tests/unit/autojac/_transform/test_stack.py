from collections.abc import Iterable

import torch
from torch import Tensor
from utils.dict_assertions import assert_tensor_dicts_are_close
from utils.tensors import ones_, tensor_, zeros_

from torchjd.autojac._transform import Stack, Transform
from torchjd.autojac._transform._base import TensorDict


class FakeGradientsTransform(Transform):
    """Transform that produces gradients filled with ones, for testing purposes."""

    def __init__(self, keys: Iterable[Tensor]) -> None:
        self.keys = set(keys)

    def __call__(self, _input: TensorDict, /) -> TensorDict:
        return {key: torch.ones_like(key) for key in self.keys}

    def check_keys(self, _input_keys: set[Tensor], /) -> set[Tensor]:
        return self.keys


def test_single_key() -> None:
    """
    Tests that the Stack transform correctly stacks gradients into a jacobian, in a very simple
    example with 2 transforms sharing the same key.
    """

    key = zeros_([3, 4])
    input = {}

    transform = FakeGradientsTransform([key])
    stack = Stack([transform, transform])

    output = stack(input)
    expected_output = {key: ones_([2, 3, 4])}

    assert_tensor_dicts_are_close(output, expected_output)


def test_disjoint_key_sets() -> None:
    """
    Tests that the Stack transform correctly stacks gradients into a jacobian, in an example where
    the output key sets of all of its transforms are disjoint. The missing values should be replaced
    by zeros.
    """

    key1 = zeros_([1, 2])
    key2 = zeros_([3])
    input = {}

    transform1 = FakeGradientsTransform([key1])
    transform2 = FakeGradientsTransform([key2])
    stack = Stack([transform1, transform2])

    output = stack(input)
    expected_output = {
        key1: tensor_([[[1.0, 1.0]], [[0.0, 0.0]]]),
        key2: tensor_([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    }

    assert_tensor_dicts_are_close(output, expected_output)


def test_overlapping_key_sets() -> None:
    """
    Tests that the Stack transform correctly stacks gradients into a jacobian, in an example where
    the output key sets all of its transforms are overlapping (non-empty intersection, but not
    equal). The missing values should be replaced by zeros.
    """

    key1 = zeros_([1, 2])
    key2 = zeros_([3])
    key3 = zeros_([4])
    input = {}

    transform12 = FakeGradientsTransform([key1, key2])
    transform23 = FakeGradientsTransform([key2, key3])
    stack = Stack([transform12, transform23])

    output = stack(input)
    expected_output = {
        key1: tensor_([[[1.0, 1.0]], [[0.0, 0.0]]]),
        key2: tensor_([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        key3: tensor_([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
    }

    assert_tensor_dicts_are_close(output, expected_output)


def test_empty() -> None:
    """Tests that the Stack transform correctly handles an empty list of transforms."""

    stack = Stack([])
    input = {}
    output = stack(input)
    expected_output = {}

    assert_tensor_dicts_are_close(output, expected_output)
