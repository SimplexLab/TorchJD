from pytest import raises
from utils.dict_assertions import assert_tensor_dicts_are_close
from utils.tensors import tensor_

from torchjd.autojac._transform import Init, RequirementError


def test_single_input() -> None:
    """
    Tests that when there is a single key to initialize, the Init transform creates a TensorDict
    whose value is a tensor full of ones, of the same shape as its key.
    """

    key = tensor_([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input = {}

    init = Init({key})

    output = init(input)
    expected_output = {key: tensor_([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])}

    assert_tensor_dicts_are_close(output, expected_output)


def test_multiple_inputs() -> None:
    """
    Tests that when there are several keys to initialize, the Init transform creates a TensorDict
    whose values are tensors full of ones, of the same shape as their corresponding keys.
    """

    key1 = tensor_([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    key2 = tensor_([1.0, 3.0, 5.0])
    input = {}

    init = Init({key1, key2})

    output = init(input)
    expected = {
        key1: tensor_([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        key2: tensor_([1.0, 1.0, 1.0]),
    }
    assert_tensor_dicts_are_close(output, expected)


def test_conjunction_of_inits_is_init() -> None:
    """
    Tests that the conjunction of 2 Init transforms is equivalent to a single Init transform with
    multiple keys.
    """

    x1 = tensor_(5.0)
    x2 = tensor_(6.0)
    input = {}

    init1 = Init({x1})
    init2 = Init({x2})
    conjunction_of_inits = init1 | init2
    init = Init({x1, x2})

    output = conjunction_of_inits(input)
    expected_output = init(input)

    assert_tensor_dicts_are_close(output, expected_output)


def test_check_keys() -> None:
    """Tests that the `check_keys` method works correctly: the input_keys should be empty."""

    key = tensor_([1.0])
    init = Init({key})

    output_keys = init.check_keys(set())
    assert output_keys == {key}

    with raises(RequirementError):
        init.check_keys({key})
