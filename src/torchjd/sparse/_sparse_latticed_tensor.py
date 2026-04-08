import itertools
from collections.abc import Callable
from functools import wraps
from math import prod

import torch
from torch import Tensor, arange, meshgrid, stack, tensor, tensordot, zeros
from torch.utils._pytree import tree_map


class SparseLatticedTensor(Tensor):
    _HANDLED_FUNCTIONS = dict[Callable, Callable]()

    @staticmethod
    def __new__(
        cls,
        physical: Tensor,
        basis: Tensor,
        offset: Tensor | None = None,
        shape: list[int] | tuple[int, ...] | torch.Size | Tensor | None = None,
    ):
        assert basis.dtype == torch.int64

        # Note [Passing requires_grad=true tensors to subclasses]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calling _make_subclass directly in an autograd context is
        # never the right thing to do, as this will detach you from
        # the autograd graph.  You must create an autograd function
        # representing the "constructor" (NegativeView, in this case)
        # and call that instead.  This assert helps prevent direct usage
        # (which is bad!)
        assert not physical.requires_grad or not torch.is_grad_enabled()

        if shape is None:
            pshape = tensor(physical.shape, dtype=torch.int64)
            shape = basis @ (pshape - 1) + 1

        return Tensor._make_wrapper_subclass(
            cls, list(shape), dtype=physical.dtype, device=physical.device
        )

    def __init__(
        self,
        physical: Tensor,
        basis: Tensor,
        offset: Tensor | None,
        shape: list[int] | tuple[int, ...] | torch.Size | Tensor | None,
    ):
        """
        This constructor is made for specifying physical and basis exactly. It should not modify
        it.

        For this reason, another constructor will be made to either modify the physical / basis to
        simplify the result, or to create a dense tensor directly if it's already dense.

        :param physical: The dense tensor holding the actual data.
        :param basis: Integer (int64) tensor of shape [virtual_ndim, physical_ndim], representing
            the linear transformation between an index in the physical tensor and the corresponding
            index in the virtual tensor, i.e. v_index = basis @ p_index + offset.
        :param offset: Offset for the virtual index, i.e. v_index = basis @ p_index + offset.
        :param shape: Size of the sparse tensor. If not provided, the size will be inferred as the
            minimum size big enough to hold all non-zero elements.

        # TODO: make a nicer interface where it's possible to provide lists or sizes instead of
            always having to provide int tensors
        """

        if offset is None:
            offset = torch.zeros(len(self.shape), dtype=torch.int64)

        if any(s == 1 for s in physical.shape):
            raise ValueError(
                "physical must not contain any dimension of size 1. Found physical.shape="
                f"{physical.shape}."
            )
        if basis.dtype is not torch.int64:
            raise ValueError(f"basis should be of int64 dtype. Found basis.dtype={basis.dtype}.")
        if not (basis >= 0).all():
            raise ValueError(f"All basis vectors must be non-negative. Found basis={basis}.")
        if basis.shape[1] != physical.ndim:
            raise ValueError(
                f"basis should have 1 column per physical dimension. Found basis={basis} and "
                f"physical.shape={physical.shape}."
            )
        if (basis.sum(dim=0) == 0).any():
            raise ValueError(
                f"basis should not have any column full of zeros. Found basis={basis}."
            )
        groups = get_groupings(list(physical.shape), basis)
        if any(len(group) != 1 for group in groups):
            raise ValueError(
                f"Dimensions must be maximally grouped. Found basis={basis} and " f"groups={groups}"
            )

        self.physical = physical
        self.basis = basis
        self.offset = offset

        if shape is None:
            pshape = tensor(physical.shape, dtype=torch.int64)
            shape = basis @ (pshape - 1) + 1
        if isinstance(shape, torch.Tensor):
            self.shape_t = shape
        else:
            self.shape_t = tensor(shape, dtype=torch.int64)

        self.pshape_t = tensor(physical.shape, dtype=torch.int64)

    def to_dense(
        self, dtype: torch.dtype | None = None, *, masked_grad: bool | None = None
    ) -> Tensor:
        assert dtype is None  # We may add support for this later
        assert masked_grad is None  # We may add support for this later

        if self.physical.ndim == 0:
            return self.physical

        p_index_ranges = [arange(s) for s in self.physical.shape]
        p_indices_grid = stack(meshgrid(*p_index_ranges, indexing="ij"))

        # addmm_cuda not implemented for Long tensors => gotta have these tensors on cpu
        reshaped_offset = self.offset.reshape([-1] + [1] * self.physical.ndim)
        v_indices_grid = tensordot(self.basis, p_indices_grid, dims=1) + reshaped_offset
        # v_indices_grid is of shape [n_virtual_dims] + physical_shape
        res = zeros(self.shape, device=self.device, dtype=self.dtype)
        res[tuple(v_indices_grid)] = self.physical
        return res

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func in cls._HANDLED_FUNCTIONS:
            return cls._HANDLED_FUNCTIONS[func](*args, **kwargs)

        print_fallback(func, args, kwargs)
        unwrapped_args = tree_map(unwrap_to_dense, args)
        unwrapped_kwargs = tree_map(unwrap_to_dense, kwargs)
        return func(*unwrapped_args, **unwrapped_kwargs)

    def __repr__(self, *, tensor_contents=None) -> str:
        return f"SparseLatticedTensor(physical={self.physical}, basis={self.basis}, offset={self.offset}, size={self.shape_t})"

    @classmethod
    def implements(cls, torch_function):
        """Register a torch function override for ScalarTensor"""

        @wraps(torch_function)
        def decorator(func):
            cls._HANDLED_FUNCTIONS[torch_function] = func
            return func

        return decorator

    @property
    def start_margin(self) -> Tensor:
        """
        Returns the margin at the start of each virtual dimension. Can be negative.

        The result is an int tensor of shape [virtual_ndim].
        """

        return self.offset

    @property
    def end_margin(self) -> Tensor:
        """
        Returns the margin at the end of each virtual dimension. Can be negative.

        The result is an int tensor of shape [virtual_ndim].
        """

        return self.shape_t - self.physical_image_size - self.offset

    @property
    def margin(self) -> Tensor:
        """
         Returns the margin at the start and end of each virtual dimension. Can be negative.

        The result is an int tensor of shape [virtual_ndim, 2].
        """

        return torch.stack([self.start_margin, self.end_margin], dim=1)

    @property
    def min_natural_virtual_indices(self) -> Tensor:
        # Basis where each positive element is replaced by 0
        non_positive_basis = torch.min(self.basis, torch.zeros_like(self.basis))
        max_physical_index = self.pshape_t - 1
        return (non_positive_basis * max_physical_index.unsqueeze(0)).sum(dim=1)

    @property
    def max_natural_virtual_indices(self) -> Tensor:
        # Basis where each negative element is replaced by 0
        non_negative = torch.max(self.basis, torch.zeros_like(self.basis))
        max_physical_index = self.pshape_t - 1
        return (non_negative * max_physical_index.unsqueeze(0)).sum(dim=1)

    @property
    def physical_image_size(self) -> Tensor:
        """
        Returns the shape of the image of the physical through the basis transform.

        The result is an int tensor of shape [virtual_ndim].
        """

        one = torch.ones(self.ndim, dtype=torch.int64)
        return self.max_natural_virtual_indices - self.min_natural_virtual_indices + one


impl = SparseLatticedTensor.implements


def print_fallback(func, args, kwargs) -> None:
    def tensor_to_str(t: Tensor) -> str:
        result = f"{t.__class__.__name__} - shape: {t.shape}"
        if isinstance(t, SparseLatticedTensor):
            result += f" - pshape: {t.physical.shape} - basis: {t.basis} - offset: {t.offset}"

        return result

    print(f"Falling back to dense for {func.__name__}")
    if len(args) > 0:
        print("* args:")
        for arg in args:
            if isinstance(arg, Tensor):
                print(f"  > {tensor_to_str(arg)}")
            elif isinstance(arg, list) and len(arg) > 0 and isinstance(arg[0], Tensor):
                list_content = "\n     ".join([tensor_to_str(t) for t in arg])
                print(f"  > [{list_content}]")
            else:
                print(f"  > {arg}")
    if len(kwargs) > 0:
        print("* kwargs:")
        for k, v in kwargs.items():
            print(f"  > {k}: {v}")
    print()


def get_groupings(pshape: list[int], basis: Tensor) -> list[list[int]]:
    basis_time_pshape = basis * tensor(pshape, dtype=torch.int64)
    groups = {i: {i} for i, column in enumerate(basis.T)}
    group_ids = [i for i in range(len(basis.T))]
    for i1, i2 in itertools.combinations(range(basis.shape[1]), 2):
        if torch.equal(basis[:, i1], basis_time_pshape[:, i2]):
            groups[group_ids[i1]].update(groups[group_ids[i2]])
            group_ids[i2] = group_ids[i1]

    new_columns = [sorted(groups[group_id]) for group_id in sorted(set(group_ids))]

    if len(new_columns) != len(pshape):
        print(f"Combined pshape with the following new columns: {new_columns}.")

    return new_columns


def to_sparse_latticed_tensor(t: Tensor) -> SparseLatticedTensor:
    if isinstance(t, SparseLatticedTensor):
        return t
    else:
        return make_slt(t, torch.eye(t.ndim, dtype=torch.int64), None, None)


def to_most_efficient_tensor(
    physical: Tensor,
    basis: Tensor,
    offset: Tensor | None,
    size: list[int] | tuple[int, ...] | torch.Size | Tensor | None,
) -> Tensor:
    physical, basis = fix_dim_of_size_1(physical, basis)
    physical, basis = fix_ungrouped_dims(physical, basis)

    if (basis.sum(dim=0) == 1).all():
        print("Turning supposedly dense SLT into Tensor. This can be bugged and slow.")
        # TODO: this condition is broken if basis is allowed to have negative values. It also only
        #  works when size is the default and offset is 0.
        # TODO: this can be done more efficiently (without even creating the SLT)
        return SparseLatticedTensor(physical, basis, offset, size).to_dense()
    else:
        return SparseLatticedTensor(physical, basis, offset, size)


def unwrap_to_dense(t: Tensor):
    if isinstance(t, SparseLatticedTensor):
        return t.to_dense()
    else:
        return t


def get_full_source(source: list[int], destination: list[int], ndim: int) -> list[int]:
    """
    Doing a movedim with source and destination is always equivalent to doing a movedim with
    [0, 1, ..., ndim-1] (aka "full_destination") as destination, and the "full_source" as source.

    This function computes the full_source based on a source and destination.

    Example:
    source=[2, 4]
    destination=[0, 3]
    ndim=5

    full_source = [2, 0, 1, 4, 3]
    full_destination = [0, 1, 2, 3, 4]
    """

    idx = torch.full((ndim,), -1, dtype=torch.int64)
    idx[destination] = tensor(source, dtype=torch.int64)
    source_set = set(source)
    idx[idx.eq(-1)] = tensor([i for i in range(ndim) if i not in source_set], dtype=torch.int64)

    return idx.tolist()


def fix_dim_of_size_1(physical: Tensor, basis: Tensor) -> tuple[Tensor, Tensor]:
    """
    Removes physical dimensions of size one and returns the corresponding new physical and new basis
    """

    is_of_size_1 = tensor([s == 1 for s in physical.shape], dtype=torch.bool)
    return physical.squeeze(), basis[:, ~is_of_size_1]


def fix_ungrouped_dims(physical: Tensor, basis: Tensor) -> tuple[Tensor, Tensor]:
    """Squash together physical dimensions that can be squashed."""

    groups = get_groupings(list(physical.shape), basis)
    nphysical = physical.reshape([prod([physical.shape[dim] for dim in group]) for group in groups])
    basis_mapping = torch.zeros(physical.ndim, nphysical.ndim, dtype=torch.int64)
    for j, group in enumerate(groups):
        basis_mapping[group[-1], j] = 1

    new_basis = basis @ basis_mapping
    return nphysical, new_basis


def make_slt(
    physical: Tensor,
    basis: Tensor,
    offset: Tensor | None,
    size: list[int] | tuple[int, ...] | torch.Size | Tensor | None,
) -> SparseLatticedTensor:
    """Fix physical and basis and create a SparseLatticedTensor with them."""

    physical, basis = fix_dim_of_size_1(physical, basis)
    physical, basis = fix_ungrouped_dims(physical, basis)
    return SparseLatticedTensor(physical, basis, offset, size)
