from pytest import mark, raises
from utils.asserts import assert_grad_close, assert_jac_close
from utils.dict_assertions import assert_tensor_dicts_are_close
from utils.tensors import ones_, tensor_, zeros_

from torchjd.autojac._transform import AccumulateGrad, AccumulateJac


def test_single_grad_accumulation():
    """
    Tests that the AccumulateGrad transform correctly accumulates gradients in .grad fields when run
    once.
    """

    shapes = [[], [1], [2, 3]]
    keys = [zeros_(shape, requires_grad=True) for shape in shapes]
    values = [ones_(shape) for shape in shapes]
    input = dict(zip(keys, values, strict=True))

    accumulate = AccumulateGrad()

    output = accumulate(input)
    assert_tensor_dicts_are_close(output, {})

    for key, value in zip(keys, values, strict=True):
        assert_grad_close(key, value)


@mark.parametrize("iterations", [1, 2, 4, 10, 13])
def test_multiple_grad_accumulations(iterations: int):
    """
    Tests that the AccumulateGrad transform correctly accumulates gradients in .grad fields when run
    `iterations` times.
    """

    shapes = [[], [1], [2, 3]]
    keys = [zeros_(shape, requires_grad=True) for shape in shapes]
    values = [ones_(shape) for shape in shapes]
    accumulate = AccumulateGrad()

    for _ in range(iterations):
        # Clone values to ensure that we accumulate values that are not ever used afterwards
        input = {key: value.clone() for key, value in zip(keys, values, strict=True)}
        accumulate(input)

    for key, value in zip(keys, values, strict=True):
        assert_grad_close(key, iterations * value)


def test_accumulate_grad_fails_when_no_requires_grad():
    """
    Tests that the AccumulateGrad transform raises an error when it tries to populate a .grad of a
    tensor that does not require grad.
    """

    key = zeros_([1], requires_grad=False)
    value = ones_([1])
    input = {key: value}

    accumulate = AccumulateGrad()

    with raises(ValueError):
        accumulate(input)


def test_accumulate_grad_fails_when_no_leaf_and_no_retains_grad():
    """
    Tests that the AccumulateGrad transform raises an error when it tries to populate a .grad of a
    tensor that is not a leaf and that does not retain grad.
    """

    key = tensor_([1.0], requires_grad=True) * 2
    value = ones_([1])
    input = {key: value}

    accumulate = AccumulateGrad()

    with raises(ValueError):
        accumulate(input)


def test_accumulate_grad_check_keys():
    """Tests that the `check_keys` method works correctly for AccumulateGrad."""

    key = tensor_([1.0], requires_grad=True)
    accumulate = AccumulateGrad()

    output_keys = accumulate.check_keys({key})
    assert output_keys == set()


def test_single_jac_accumulation():
    """
    Tests that the AccumulateJac transform correctly accumulates jacobians in .jac fields when run
    once.
    """

    shapes = [[], [1], [2, 3]]
    keys = [zeros_(shape, requires_grad=True) for shape in shapes]
    values = [ones_([4, *shape]) for shape in shapes]
    input = dict(zip(keys, values, strict=True))

    accumulate = AccumulateJac()

    output = accumulate(input)
    assert_tensor_dicts_are_close(output, {})

    for key, value in zip(keys, values, strict=True):
        assert_jac_close(key, value)


@mark.parametrize("iterations", [1, 2, 4, 10, 13])
def test_multiple_jac_accumulations(iterations: int):
    """
    Tests that the AccumulateJac transform correctly accumulates jacobians in .jac fields when run
    `iterations` times.
    """

    shapes = [[], [1], [2, 3]]
    keys = [zeros_(shape, requires_grad=True) for shape in shapes]
    values = [ones_([4, *shape]) for shape in shapes]

    accumulate = AccumulateJac()

    for _ in range(iterations):
        # Clone values to ensure that we accumulate values that are not ever used afterwards
        input = {key: value.clone() for key, value in zip(keys, values, strict=True)}
        accumulate(input)

    for key, value in zip(keys, values, strict=True):
        assert_jac_close(key, iterations * value)


def test_accumulate_jac_fails_when_no_requires_grad():
    """
    Tests that the AccumulateJac transform raises an error when it tries to populate a .jac of a
    tensor that does not require grad.
    """

    key = zeros_([1], requires_grad=False)
    value = ones_([4, 1])
    input = {key: value}

    accumulate = AccumulateJac()

    with raises(ValueError):
        accumulate(input)


def test_accumulate_jac_fails_when_no_leaf_and_no_retains_grad():
    """
    Tests that the AccumulateJac transform raises an error when it tries to populate a .jac of a
    tensor that is not a leaf and that does not retain grad.
    """

    key = tensor_([1.0], requires_grad=True) * 2
    value = ones_([4, 1])
    input = {key: value}

    accumulate = AccumulateJac()

    with raises(ValueError):
        accumulate(input)


def test_accumulate_jac_fails_when_shape_mismatch():
    """
    Tests that the AccumulateJac transform raises an error when the jacobian shape does not match
    the parameter shape (ignoring the first dimension).
    """

    key = zeros_([2, 3], requires_grad=True)
    value = ones_([4, 3, 2])  # Wrong shape: should be [4, 2, 3], not [4, 3, 2]
    input = {key: value}

    accumulate = AccumulateJac()

    with raises(RuntimeError):
        accumulate(input)


def test_accumulate_jac_check_keys():
    """Tests that the `check_keys` method works correctly for AccumulateJac."""

    key = tensor_([1.0], requires_grad=True)
    accumulate = AccumulateJac()

    output_keys = accumulate.check_keys({key})
    assert output_keys == set()
