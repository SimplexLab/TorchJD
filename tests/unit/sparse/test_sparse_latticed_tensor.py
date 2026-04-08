import torch
from pytest import mark
from torch import Tensor, tensor
from torch.ops import aten  # type: ignore
from torch.testing import assert_close
from utils.tensors import randn_, tensor_, zeros_

from torchjd.sparse._aten_function_overrides.einsum import einsum
from torchjd.sparse._aten_function_overrides.pointwise import (
    _IN_PLACE_POINTWISE_FUNCTIONS,
    _POINTWISE_FUNCTIONS,
)
from torchjd.sparse._coalesce import fix_zero_basis_vectors
from torchjd.sparse._sparse_latticed_tensor import (
    SparseLatticedTensor,
    fix_ungrouped_dims,
    get_full_source,
    get_groupings,
)


def test_to_dense():
    n = 2
    m = 3
    a = randn_([n, m])
    b = SparseLatticedTensor(a, tensor([[1, 0], [0, 1], [0, 1], [1, 0]]), offset=None, shape=None)
    c = b.to_dense()

    for i in range(n):
        for j in range(m):
            assert c[i, j, j, i] == a[i, j]


def test_to_dense2():
    a = tensor_([1.0, 2.0, 3.0])
    b = SparseLatticedTensor(a, tensor([[4]]), offset=None, shape=None)
    c = b.to_dense()
    expected = tensor_([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])
    assert torch.all(torch.eq(c, expected))


@mark.parametrize(
    ["a_pshape", "a_basis", "b_pshape", "b_basis", "a_indices", "b_indices", "output_indices"],
    [
        (
            [4, 5],
            tensor([[1, 0], [1, 0], [0, 1]]),
            [4, 5],
            tensor([[1, 0], [0, 1], [0, 1]]),
            [0, 1, 2],
            [0, 2, 3],
            [0, 1, 3],
        ),
        (
            [2, 3, 5],
            tensor([[3, 1, 0], [1, 0, 2]]),
            [10, 3],
            tensor([[1, 0], [0, 1]]),
            [0, 1],
            [1, 2],
            [0, 2],
        ),
        (
            [6, 2, 3],
            tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            [2, 3],
            tensor([[3, 1], [1, 0], [0, 1]]),
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ),
    ],
)
def test_einsum(
    a_pshape: list[int],
    a_basis: Tensor,
    b_pshape: list[int],
    b_basis: Tensor,
    a_indices: list[int],
    b_indices: list[int],
    output_indices: list[int],
):
    a = SparseLatticedTensor(randn_(a_pshape), a_basis, offset=None, shape=None)
    b = SparseLatticedTensor(randn_(b_pshape), b_basis, offset=None, shape=None)

    res = einsum((a, a_indices), (b, b_indices), output=output_indices)

    expected = torch.einsum(a.to_dense(), a_indices, b.to_dense(), b_indices, output_indices)

    assert isinstance(res, SparseLatticedTensor)
    assert_close(res.to_dense(), expected)


@mark.parametrize(
    "shape",
    [
        [],
        [2],
        [2, 3],
        [2, 3, 4],
    ],
)
def test_sparse_latticed_tensor_scalar(shape: list[int]):
    a = randn_(shape)
    b = SparseLatticedTensor(a, torch.eye(len(shape), dtype=torch.int64), offset=None, shape=None)

    assert_close(a, b.to_dense())


@mark.parametrize("dim", [2, 3, 4, 5, 10])
def test_diag_equivalence(dim: int):
    a = randn_([dim])
    b = SparseLatticedTensor(a, tensor([[1], [1]]), offset=None, shape=None)

    diag_a = torch.diag(a)

    assert_close(b.to_dense(), diag_a)


def test_three_virtual_single_physical():
    dim = 10
    a = randn_([dim])
    b = SparseLatticedTensor(a, tensor([[1], [1], [1]]), offset=None, shape=None)

    expected = zeros_([dim, dim, dim])
    for i in range(dim):
        expected[i, i, i] = a[i]

    assert_close(b.to_dense(), expected)


@mark.parametrize("func", _POINTWISE_FUNCTIONS)
def test_pointwise(func):
    dim = 10
    a = randn_([dim])
    b = SparseLatticedTensor(a, tensor([[1], [1]]), offset=None, shape=None)
    c = b.to_dense()
    res = func(b)
    assert isinstance(res, SparseLatticedTensor)

    assert_close(res.to_dense(), func(c), equal_nan=True)


@mark.parametrize("func", _IN_PLACE_POINTWISE_FUNCTIONS)
def test_inplace_pointwise(func):
    dim = 10
    a = randn_([dim])
    b = SparseLatticedTensor(a, tensor([[1], [1]]), offset=None, shape=None)
    c = b.to_dense()
    func(b)
    assert isinstance(b, SparseLatticedTensor)

    assert_close(b.to_dense(), func(c), equal_nan=True)


@mark.parametrize("func", [torch.mean, torch.sum])
def test_unary(func):
    dim = 10
    a = randn_([dim])
    b = SparseLatticedTensor(a, tensor([[1], [1]]), offset=None, shape=None)
    c = b.to_dense()

    res = func(b)
    assert_close(res.to_dense(), func(c))


@mark.parametrize(
    ["physical_shape", "basis", "target_shape", "expected_physical_shape", "expected_basis"],
    [
        (
            [2, 3],
            tensor([[1, 0], [1, 0], [0, 1]]),
            [2, 2, 3],
            [2, 3],
            tensor([[1, 0], [1, 0], [0, 1]]),
        ),  # no change of shape
        (
            [2, 3],
            tensor([[1, 0], [3, 1]]),
            [2, 6],
            [2, 3],
            tensor([[1, 0], [3, 1]]),
        ),  # no change of shape
        (
            [2, 3],
            tensor([[1, 0], [1, 0], [0, 1]]),
            [2, 6],
            [2, 3],
            tensor([[1, 0], [3, 1]]),
        ),  # squashing 2 dims
        (
            [2, 3],
            tensor([[1, 0], [3, 1]]),
            [2, 2, 3],
            [2, 3],
            tensor([[1, 0], [1, 0], [0, 1]]),
        ),  # unsquashing into 2 dims
        (
            [2, 3],
            tensor([[9, 1]]),
            [2, 6],
            [2, 3],
            tensor([[1, 0], [3, 1]]),
        ),  # unsquashing into 2 dims
        (
            [2, 3],
            tensor([[1, 0], [1, 0], [0, 1]]),
            [12],
            [2, 3],
            tensor([[9, 1]]),
        ),  # squashing 3 dims
        (
            [2, 3],
            tensor([[9, 1]]),
            [2, 2, 3],
            [2, 3],
            tensor([[1, 0], [1, 0], [0, 1]]),
        ),  # unsquashing into 3 dims
        (
            [4],
            tensor([[1], [1]]),
            [2, 2, 4],
            [2, 2],
            tensor([[1, 0], [0, 1], [2, 1]]),
        ),  # unsquashing physical dim
        (
            [4],
            tensor([[1], [1]]),
            [4, 2, 2],
            [2, 2],
            tensor([[2, 1], [1, 0], [0, 1]]),
        ),  # unsquashing physical dim
        (
            [2, 3, 4],
            tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            [4, 12],
            [2, 12],
            tensor([[3, 0], [0, 1]]),
        ),  # world boss
        (
            [2, 12],
            tensor([[3, 0], [0, 1]]),
            [2, 2, 3, 4],
            [2, 3, 4],
            tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ),  # world boss
    ],
)
def test_view(
    physical_shape: list[int],
    basis: Tensor,
    target_shape: list[int],
    expected_physical_shape: list[int],
    expected_basis: Tensor,
):
    a = randn_(tuple(physical_shape))
    t = SparseLatticedTensor(a, basis, offset=None, shape=None)

    result = aten.view.default(t, target_shape)
    expected = t.to_dense().reshape(target_shape)

    assert isinstance(result, SparseLatticedTensor)
    assert list(result.physical.shape) == expected_physical_shape
    assert torch.equal(result.basis, expected_basis)
    assert torch.all(torch.eq(result.to_dense(), expected))


@mark.parametrize(
    ["pshape", "basis", "expected"],
    [
        (
            [[32, 2, 3, 4, 5]],
            torch.tensor([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 60, 20, 5, 1]]),
            [[0], [1, 2, 3, 4]],
        )
    ],
)
def test_get_groupings(pshape: list[int], basis: torch.Tensor, expected: list[list[int]]):
    result = get_groupings(pshape, basis)
    assert result == expected


@mark.parametrize(
    ["physical_shape", "basis", "expected_physical_shape", "expected_basis"],
    [
        (
            [3, 4, 5],
            tensor([[20, 5, 1], [4, 1, 12], [0, 0, 1]]),
            [12, 5],
            tensor([[5, 1], [1, 12], [0, 1]]),
        ),
        (
            [32, 20, 8],
            tensor([[1, 0, 0], [1, 32, 0], [0, 0, 1]]),
            [32, 20, 8],
            tensor([[1, 0, 0], [1, 32, 0], [0, 0, 1]]),
        ),
        ([3, 3, 4], tensor([[3, 1, 0], [0, 4, 1]]), [3, 3, 4], tensor([[3, 1, 0], [0, 4, 1]])),
    ],
)
def test_fix_ungrouped_dims(
    physical_shape: list[int],
    basis: Tensor,
    expected_physical_shape: list[int],
    expected_basis: Tensor,
):
    physical = randn_(physical_shape)
    fixed_physical, fixed_basis = fix_ungrouped_dims(physical, basis)

    assert list(fixed_physical.shape) == expected_physical_shape
    assert torch.equal(fixed_basis, expected_basis)


@mark.parametrize(
    [
        "source",
        "destination",
        "ndim",
    ],
    [
        ([2, 4], [0, 3], 5),
        ([5, 3, 6], [2, 0, 5], 8),
    ],
)
def test_get_column_indices(source: list[int], destination: list[int], ndim: int):
    # TODO: this test should be improved / removed. It creates quite big tensors for nothing.

    t = randn_(list(torch.randint(3, 8, size=(ndim,))))
    full_destination = list(range(ndim))
    full_source = get_full_source(source, destination, ndim)
    assert torch.equal(t.movedim(full_source, full_destination), t.movedim(source, destination))


@mark.parametrize(
    ["slt_args", "dim", "expected_densify"],
    [
        ([([3], tensor([[1], [1]])), ([3], tensor([[1], [1]]))], 1, False),
        ([([3], tensor([[2]])), ([4], tensor([[2]]))], 0, True),
        ([([3, 2], tensor([[1, 0], [1, 3]])), ([3, 2], tensor([[1, 0], [1, 3]]))], 1, False),
    ],
)
def test_concatenate(
    slt_args: list[tuple[list[int], Tensor]],
    dim: int,
    expected_densify: bool,
):
    tensors = [
        SparseLatticedTensor(randn_(pshape), basis, offset=None, shape=None)
        for pshape, basis in slt_args
    ]
    res = aten.cat.default(tensors, dim)
    expected = aten.cat.default([t.to_dense() for t in tensors], dim)

    if expected_densify:
        assert not isinstance(res, SparseLatticedTensor)
    else:
        assert isinstance(res, SparseLatticedTensor)

    assert torch.all(torch.eq(res.to_dense(), expected))


@mark.parametrize(
    ["physical", "basis", "expected_physical", "expected_basis"],
    [
        (
            tensor_([[1, 2, 3], [4, 5, 6]]),
            tensor([[1, 0], [1, 0], [2, 0]]),
            tensor_([6, 15]),
            tensor([[1], [1], [2]]),
        ),
        (
            tensor_([[1, 2, 3], [4, 5, 6]]),
            tensor([[1, 1], [1, 0], [2, 0]]),
            tensor_([[1, 2, 3], [4, 5, 6]]),
            tensor([[1, 1], [1, 0], [2, 0]]),
        ),
        (
            tensor_([[3, 2, 1], [6, 5, 4]]),
            tensor([[0, 0], [0, 0], [0, 0]]),
            tensor_(21),
            tensor([[], [], []], dtype=torch.int64),
        ),
    ],
)
def test_fix_zero_basis_vectors(
    physical: Tensor,
    basis: Tensor,
    expected_physical: Tensor,
    expected_basis: Tensor,
):
    physical, basis = fix_zero_basis_vectors(physical, basis)
    assert torch.equal(physical, expected_physical)
    assert torch.equal(basis, expected_basis)
