from torch import Tensor
from torch.ops import aten  # type: ignore

from torchjd.sparse._sparse_latticed_tensor import SparseLatticedTensor, impl


@impl(aten.threshold_backward.default)
def threshold_backward_default(
    grad_output: SparseLatticedTensor, self: Tensor, threshold
) -> SparseLatticedTensor:
    new_physical = aten.threshold_backward.default(grad_output.physical, self, threshold)

    return SparseLatticedTensor(
        new_physical, grad_output.basis, grad_output.offset, grad_output.shape_t
    )


@impl(aten.hardtanh_backward.default)
def hardtanh_backward_default(
    grad_output: SparseLatticedTensor,
    self: Tensor,
    min_val: Tensor | int | float,
    max_val: Tensor | int | float,
) -> SparseLatticedTensor:
    if isinstance(self, SparseLatticedTensor):
        raise NotImplementedError()

    new_physical = aten.hardtanh_backward.default(grad_output.physical, self, min_val, max_val)
    return SparseLatticedTensor(
        new_physical, grad_output.basis, grad_output.offset, grad_output.shape_t
    )


@impl(aten.hardswish_backward.default)
def hardswish_backward_default(grad_output: SparseLatticedTensor, self: Tensor):
    if isinstance(self, SparseLatticedTensor):
        raise NotImplementedError()

    new_physical = aten.hardswish_backward.default(grad_output.physical, self)
    return SparseLatticedTensor(
        new_physical, grad_output.basis, grad_output.offset, grad_output.shape_t
    )
