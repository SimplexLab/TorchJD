from pytest import mark
from torch.testing import assert_close
from utils.asserts import assert_is_psd_matrix, assert_is_psd_tensor
from utils.tensors import randn_

from torchjd._linalg import compute_gramian, is_psd_matrix
from torchjd.autogram._gramian_utils import flatten, movedim, reshape


@mark.parametrize(
    ["original_shape", "target_shape"],
    [
        ([], []),
        ([], [1, 1]),
        ([1], []),
        ([12], [2, 3, 2]),
        ([12], [4, 3]),
        ([12], [12]),
        ([4, 3], [12]),
        ([4, 3], [2, 3, 2]),
        ([4, 3], [3, 4]),
        ([4, 3], [4, 3]),
        ([6, 7, 9], [378]),
        ([6, 7, 9], [9, 42]),
        ([6, 7, 9], [2, 7, 27]),
        ([6, 7, 9], [6, 7, 9]),
    ],
)
def test_reshape_equivarience(original_shape: list[int], target_shape: list[int]) -> None:
    """Tests that reshape_gramian is such that compute_gramian is equivariant to a reshape."""

    original_matrix = randn_([*original_shape, 2])
    target_matrix = original_matrix.reshape([*target_shape, 2])

    original_gramian = compute_gramian(original_matrix, 1)
    target_gramian = compute_gramian(target_matrix, 1)

    reshaped_gramian = reshape(original_gramian, target_shape)

    assert_close(reshaped_gramian, target_gramian)


@mark.parametrize(
    ["original_shape", "target_shape"],
    [
        ([], []),
        ([], [1, 1]),
        ([1], []),
        ([12], [2, 3, 2]),
        ([12], [4, 3]),
        ([12], [12]),
        ([4, 3], [12]),
        ([4, 3], [4, 3]),
        ([6, 7, 9], [378]),
        ([6, 7, 9], [6, 7, 9]),
    ],
)
def test_reshape_yields_psd(original_shape: list[int], target_shape: list[int]) -> None:
    matrix = randn_([*original_shape, 2])
    gramian = compute_gramian(matrix, 1)
    reshaped_gramian = reshape(gramian, target_shape)
    assert_is_psd_tensor(reshaped_gramian, atol=1e-04, rtol=0.0)


@mark.parametrize(
    "shape",
    [
        [],
        [1],
        [12],
        [4, 3],
        [6, 7, 9],
    ],
)
def test_flatten_yields_matrix(shape: list[int]) -> None:
    matrix = randn_([*shape, 2])
    gramian = compute_gramian(matrix, 1)
    flattened_gramian = flatten(gramian)
    assert is_psd_matrix(flattened_gramian)


@mark.parametrize(
    "shape",
    [
        [],
        [1],
        [12],
        [4, 3],
        [6, 7, 9],
    ],
)
def test_flatten_yields_psd(shape: list[int]) -> None:
    matrix = randn_([*shape, 2])
    gramian = compute_gramian(matrix, 1)
    flattened_gramian = flatten(gramian)
    assert_is_psd_matrix(flattened_gramian, atol=1e-04, rtol=0.0)


@mark.parametrize(
    ["shape", "source", "destination"],
    [
        ([], [], []),
        ([1], [0], [0]),
        ([1], [], []),
        ([1, 1], [], []),
        ([1, 1], [1], [0]),
        ([6, 7], [1], [0]),
        ([3, 1], [0, 1], [1, 0]),
        ([1, 1, 1], [], []),
        ([3, 2, 5], [], []),
        ([1, 1, 1], [2], [0]),
        ([3, 2, 5], [1], [2]),
        ([2, 2, 3], [0, 2], [1, 0]),
        ([2, 2, 3], [0, 2, 1], [1, 0, 2]),
    ],
)
def test_movedim_equivariance(shape: list[int], source: list[int], destination: list[int]) -> None:
    """Tests that movedim_gramian is such that compute_gramian is equivariant to a movedim."""

    original_matrix = randn_([*shape, 2])
    target_matrix = original_matrix.movedim(source, destination)

    original_gramian = compute_gramian(original_matrix, 1)
    target_gramian = compute_gramian(target_matrix, 1)

    moveddim_gramian = movedim(original_gramian, source, destination)

    assert_close(moveddim_gramian, target_gramian)


@mark.parametrize(
    ["shape", "source", "destination"],
    [
        ([], [], []),
        ([1], [0], [0]),
        ([1], [], []),
        ([1, 1], [], []),
        ([1, 1], [1], [0]),
        ([6, 7], [1], [0]),
        ([3, 1], [0, 1], [1, 0]),
        ([1, 1, 1], [], []),
        ([3, 2, 5], [], []),
        ([1, 1, 1], [2], [0]),
        ([3, 2, 5], [1], [2]),
        ([2, 2, 3], [0, 2], [1, 0]),
        ([2, 2, 3], [0, 2, 1], [1, 0, 2]),
    ],
)
def test_movedim_yields_psd(shape: list[int], source: list[int], destination: list[int]) -> None:
    matrix = randn_([*shape, 2])
    gramian = compute_gramian(matrix, 1)
    moveddim_gramian = movedim(gramian, source, destination)
    assert_is_psd_tensor(moveddim_gramian)
