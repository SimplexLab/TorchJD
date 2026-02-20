import torch
from pytest import mark, raises
from utils.asserts import assert_has_jac, assert_has_no_jac, assert_jac_close
from utils.tensors import eye_, randn_, tensor_

from torchjd.autojac import backward
from torchjd.autojac._backward import _create_jac_tensors_dict, _create_transform
from torchjd.autojac._transform import OrderedSet


@mark.parametrize("default_jac_tensors", [True, False])
def test_check_create_transform(default_jac_tensors: bool) -> None:
    """Tests that _create_transform creates a valid Transform."""

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    optional_jac_tensors = (
        None if default_jac_tensors else [tensor_([1.0, 0.0]), tensor_([0.0, 1.0])]
    )

    jac_tensors = _create_jac_tensors_dict(
        tensors=OrderedSet([y1, y2]),
        opt_jac_tensors=optional_jac_tensors,
    )
    transform = _create_transform(
        tensors=OrderedSet([y1, y2]),
        inputs=OrderedSet([a1, a2]),
        retain_graph=False,
        parallel_chunk_size=None,
    )

    output_keys = transform.check_keys(set(jac_tensors.keys()))
    assert output_keys == set()


def test_jac_is_populated() -> None:
    """Tests that backward correctly fills the .jac field."""

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    backward([y1, y2])

    for a in [a1, a2]:
        assert_has_jac(a)


@mark.parametrize("shape", [(1, 3), (2, 3), (2, 6), (5, 8), (20, 55)])
@mark.parametrize("manually_specify_inputs", [True, False])
@mark.parametrize("chunk_size", [1, 2, None])
def test_value_is_correct(
    shape: tuple[int, int],
    manually_specify_inputs: bool,
    chunk_size: int | None,
) -> None:
    """
    Tests that the .jac value filled by backward is correct in a simple example of matrix-vector
    product.
    """

    J = randn_(shape)
    input = randn_([shape[1]], requires_grad=True)
    output = J @ input  # Note that the Jacobian of output w.r.t. input is J.

    inputs = [input] if manually_specify_inputs else None

    backward(
        [output],
        inputs=inputs,
        parallel_chunk_size=chunk_size,
    )

    assert_jac_close(input, J)


@mark.parametrize("rows", [1, 2, 5])
def test_jac_tensors_value_is_correct(rows: int) -> None:
    """
    Tests that backward correctly computes the product of jac_tensors and the Jacobian.
    result = jac_tensors @ Jacobian(tensors, inputs).
    """
    input_size = 4
    output_size = 3

    J_model = randn_((output_size, input_size))

    input = randn_([input_size], requires_grad=True)
    tensor = J_model @ input

    J_init = randn_((rows, output_size))

    backward(
        tensor,
        jac_tensors=J_init,
        inputs=[input],
    )

    expected_jac = J_init @ J_model
    assert_jac_close(input, expected_jac)


@mark.parametrize("rows", [1, 3])
def test_jac_tensors_multiple_components(rows: int) -> None:
    """
    Tests that jac_tensors works correctly when tensors is a list of multiple tensors. The
    jac_tensors must match the structure of tensors.
    """
    input_len = 2
    input = randn_([input_len], requires_grad=True)

    y1 = input * 2
    y2 = torch.cat([input, input[:1]])

    J1 = randn_((rows, 2))
    J2 = randn_((rows, 3))

    backward([y1, y2], jac_tensors=[J1, J2], inputs=[input])

    jac_y1 = eye_(2) * 2

    jac_y2 = tensor_([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    expected = J1 @ jac_y1 + J2 @ jac_y2

    assert_jac_close(input, expected)


def test_jac_tensors_length_mismatch() -> None:
    """Tests that backward raises a ValueError early if len(jac_tensors) != len(tensors)."""
    x = tensor_([1.0, 2.0], requires_grad=True)
    y1 = x * 2
    y2 = x * 3

    J1 = randn_((2, 2))

    with raises(
        ValueError,
        match=r"`jac_tensors` should have the same length as `tensors`\. \(got 1 and 2\)",
    ):
        backward([y1, y2], jac_tensors=[J1], inputs=[x])


def test_jac_tensors_shape_mismatch() -> None:
    """
    Tests that backward raises a ValueError early if the shape of a tensor in jac_tensors is
    incompatible with the corresponding tensor.
    """
    x = tensor_([1.0, 2.0], requires_grad=True)
    y = x * 2

    J_bad = randn_((3, 5))

    with raises(
        ValueError,
        match=r"Shape mismatch: `jac_tensors\[0\]` has shape .* but `tensors\[0\]` has shape .*\.",
    ):
        backward(y, jac_tensors=J_bad, inputs=[x])


@mark.parametrize(
    "rows_y1, rows_y2",
    [
        (3, 5),
        (1, 2),
    ],
)
def test_jac_tensors_inconsistent_first_dimension(rows_y1: int, rows_y2: int) -> None:
    """
    Tests that backward raises a ValueError early when the provided jac_tensors have inconsistent
    first dimensions.
    """
    x = tensor_([1.0, 2.0], requires_grad=True)

    y1 = x * 2
    y2 = x.sum()

    j1 = randn_((rows_y1, 2))
    j2 = randn_((rows_y2,))

    with raises(
        ValueError, match=r"All Jacobians in `jac_tensors` should have the same number of rows\."
    ):
        backward([y1, y2], jac_tensors=[j1, j2], inputs=[x])


def test_empty_inputs() -> None:
    """Tests that backward does not fill the .jac values if no input is specified."""

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    backward([y1, y2], inputs=[])

    for a in [a1, a2]:
        assert_has_no_jac(a)


def test_partial_inputs() -> None:
    """
    Tests that backward fills the right .jac values when only a subset of the actual inputs are
    specified as inputs.
    """

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    backward([y1, y2], inputs=[a1])

    assert_has_jac(a1)
    assert_has_no_jac(a2)


def test_empty_tensors_fails() -> None:
    """Tests that backward raises an error when called with an empty list of tensors."""

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    with raises(ValueError):
        backward([], inputs=[a1, a2])


def test_multiple_tensors() -> None:
    """
    Tests that giving multiple tensors to backward is equivalent to giving a single tensor
    containing all the values of the original tensors.
    """

    J1 = tensor_([[-1.0, 1.0], [2.0, 4.0]])
    J2 = tensor_([[1.0, 1.0], [0.6, 0.8]])

    # First computation graph: multiple tensors
    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    backward([y1, y2])

    assert_jac_close(a1, J1)
    assert_jac_close(a2, J2)

    # Second computation graph: single concatenated tensor
    b1 = tensor_([1.0, 2.0], requires_grad=True)
    b2 = tensor_([3.0, 4.0], requires_grad=True)

    z1 = tensor_([-1.0, 1.0]) @ b1 + b2.sum()
    z2 = (b1**2).sum() + b2.norm()

    backward(torch.cat([z1.reshape(-1), z2.reshape(-1)]))

    assert_jac_close(b1, J1)
    assert_jac_close(b2, J2)


@mark.parametrize("chunk_size", [None, 1, 2, 4])
def test_various_valid_chunk_sizes(chunk_size: int | None) -> None:
    """Tests that backward works for various valid values of parallel_chunk_size."""

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    backward([y1, y2], parallel_chunk_size=chunk_size)

    for a in [a1, a2]:
        assert_has_jac(a)


@mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_fails(chunk_size: int) -> None:
    """Tests that backward raises an error when using invalid chunk sizes."""

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    with raises(ValueError):
        backward([y1, y2], parallel_chunk_size=chunk_size)


def test_input_retaining_grad_fails() -> None:
    """
    Tests that backward raises an error when some input in the computation graph of the ``tensors``
    parameter retains grad and vmap has to be used.
    """

    a = tensor_([1.0, 2.0], requires_grad=True)
    b = 2 * a
    b.retain_grad()
    y = 3 * b

    # backward itself doesn't raise the error, but it fills b.grad with a BatchedTensor
    # (and it also fills b.jac with the correct Jacobian)
    backward(y, inputs=[b])

    with raises(RuntimeError):
        # Using such a BatchedTensor should result in an error
        _ = -b.grad  # type: ignore[unsupported-operator]


def test_non_input_retaining_grad_fails() -> None:
    """
    Tests that backward fails to fill a valid `.grad` when some tensor in the computation graph of
    the ``tensors`` parameter retains grad and vmap has to be used.
    """

    a = tensor_([1.0, 2.0], requires_grad=True)
    b = 2 * a
    b.retain_grad()
    y = 3 * b

    # backward itself doesn't raise the error, but it fills b.grad with a BatchedTensor
    backward(y, inputs=[a])

    with raises(RuntimeError):
        # Using such a BatchedTensor should result in an error
        _ = -b.grad  # type: ignore[unsupported-operator]


@mark.parametrize("chunk_size", [1, 3, None])
def test_tensor_used_multiple_times(chunk_size: int | None) -> None:
    """
    Tests that backward works correctly when one of the inputs is used multiple times. In this
    setup, the autograd graph is still acyclic, but the graph of tensors used becomes cyclic.
    """

    a = tensor_(3.0, requires_grad=True)
    b = 2.0 * a
    c = a * b
    d = a * c
    e = a * d

    backward([d, e], parallel_chunk_size=chunk_size)

    J = tensor_([2.0 * 3.0 * (a**2).item(), 2.0 * 4.0 * (a**3).item()])

    assert_jac_close(a, J)


def test_repeated_tensors() -> None:
    """
    Tests that backward does not allow repeating tensors.

    This behavior is different from torch.autograd.backward which would sum the gradients of the
    repeated tensors, but it simplifies a lot the implementation of autojac and there are
    alternative ways of producing Jacobians with repeated rows anyway.
    """

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + (a2**2).sum()

    with raises(ValueError):
        backward([y1, y1, y2])


def test_repeated_inputs() -> None:
    """
    Tests that backward correctly works when some inputs are repeated. In this case, since
    torch.autograd.backward ignores the repetition of the inputs, it is natural for autojac to
    ignore that as well.
    """

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + (a2**2).sum()

    J1 = tensor_([[-1.0, 1.0], [2.0, 4.0]])
    J2 = tensor_([[1.0, 1.0], [6.0, 8.0]])

    backward([y1, y2], inputs=[a1, a1, a2])

    assert_jac_close(a1, J1)
    assert_jac_close(a2, J2)
