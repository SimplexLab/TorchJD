from pytest import mark
from torch.testing import assert_close
from utils.asserts import assert_is_psd_matrix
from utils.tensors import randn_, tensor_

from torchjd._linalg import compute_gramian, is_matrix, normalize, regularize


@mark.parametrize(
    "shape",
    [
        [1],
        [12],
        [4, 3],
        [3, 1],
        [4, 4],
        [4, 3],
        [6, 7],
        [5, 0],
        [6, 7, 9],
    ],
)
def test_gramian_is_psd(shape: list[int]):
    matrix = randn_(shape)
    gramian = compute_gramian(matrix)
    assert_is_psd_matrix(gramian)


def test_compute_gramian_scalar_input_0():
    t = tensor_(5.0)
    gramian = compute_gramian(t, contracted_dims=0)
    expected = tensor_(25.0)

    assert_close(gramian, expected)


def test_compute_gramian_vector_input_0():
    t = tensor_([2.0, 3.0])
    gramian = compute_gramian(t, contracted_dims=0)
    expected = tensor_([[4.0, 6.0], [6.0, 9.0]])

    assert_close(gramian, expected)


def test_compute_gramian_vector_input_1():
    t = tensor_([2.0, 3.0])
    gramian = compute_gramian(t, contracted_dims=1)
    expected = tensor_(13.0)

    assert_close(gramian, expected)


def test_compute_gramian_matrix_input_0():
    t = tensor_([[1.0, 2.0], [3.0, 4.0]])
    gramian = compute_gramian(t, contracted_dims=0)
    expected = tensor_(
        [
            [[[1.0, 3.0], [2.0, 4.0]], [[2.0, 6.0], [4.0, 8.0]]],
            [[[3.0, 9.0], [6.0, 12.0]], [[4.0, 12.0], [8.0, 16.0]]],
        ],
    )

    assert_close(gramian, expected)


def test_compute_gramian_matrix_input_1():
    t = tensor_([[1.0, 2.0], [3.0, 4.0]])
    gramian = compute_gramian(t, contracted_dims=1)
    expected = tensor_([[5.0, 11.0], [11.0, 25.0]])

    assert_close(gramian, expected)


def test_compute_gramian_matrix_input_2():
    t = tensor_([[1.0, 2.0], [3.0, 4.0]])
    gramian = compute_gramian(t, contracted_dims=2)
    expected = tensor_(30.0)

    assert_close(gramian, expected)


@mark.parametrize(
    "shape",
    [
        [3, 1],
        [4, 4],
        [4, 3],
        [6, 7],
        [5, 0],
    ],
)
def test_normalize_yields_psd(shape: list[int]):
    matrix = randn_(shape)
    assert is_matrix(matrix)
    gramian = compute_gramian(matrix)
    normalized_gramian = normalize(gramian, 1e-05)
    assert_is_psd_matrix(normalized_gramian)


@mark.parametrize(
    "shape",
    [
        [3, 1],
        [4, 4],
        [4, 3],
        [6, 7],
        [5, 0],
    ],
)
def test_regularize_yields_psd(shape: list[int]):
    matrix = randn_(shape)
    assert is_matrix(matrix)
    gramian = compute_gramian(matrix)
    normalized_gramian = regularize(gramian, 1e-05)
    assert_is_psd_matrix(normalized_gramian)
