import torch
from pytest import raises
from utils.dict_assertions import assert_tensor_dicts_are_close
from utils.tensors import tensor_

from torchjd.autojac._transform import RequirementError, Select


def test_partition() -> None:
    """
    Tests that the Select transform works correctly by applying 2 different Selects to a TensorDict,
    whose keys form a partition of the keys of the TensorDict.
    """

    key1 = tensor_([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    key2 = tensor_([1.0, 3.0, 5.0])
    key3 = tensor_(2.0)
    value1 = torch.ones_like(key1)
    value2 = torch.ones_like(key2)
    value3 = torch.ones_like(key3)
    input = {key1: value1, key2: value2, key3: value3}

    select1 = Select({key1, key2})
    select2 = Select({key3})

    output1 = select1(input)
    expected_output1 = {key1: value1, key2: value2}

    assert_tensor_dicts_are_close(output1, expected_output1)

    output2 = select2(input)
    expected_output2 = {key3: value3}

    assert_tensor_dicts_are_close(output2, expected_output2)


def test_conjunction_of_selects_is_select() -> None:
    """
    Tests that the conjunction of 2 Select transforms is equivalent to directly using a Select with
    the union of the keys of the 2 Selects.
    """

    x1 = tensor_(5.0)
    x2 = tensor_(6.0)
    x3 = tensor_(7.0)
    input = {x1: torch.ones_like(x1), x2: torch.ones_like(x2), x3: torch.ones_like(x3)}

    select1 = Select({x1})
    select2 = Select({x2})
    conjunction_of_selects = select1 | select2
    select = Select({x1, x2})

    output = conjunction_of_selects(input)
    expected_output = select(input)

    assert_tensor_dicts_are_close(output, expected_output)


def test_check_keys() -> None:
    """
    Tests that the `check_keys` method works correctly: the set of keys to select should be a subset
    of the set of required_keys.
    """

    key1 = tensor_([1.0])
    key2 = tensor_([2.0])
    key3 = tensor_([3.0])

    output_keys = Select({key1, key2}).check_keys({key1, key2, key3})
    assert output_keys == {key1, key2}

    with raises(RequirementError):
        Select({key1, key2}).check_keys({key1})
