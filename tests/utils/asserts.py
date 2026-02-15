import torch
from torch import Tensor
from torch.testing import assert_close

from torchjd._linalg import is_psd_matrix, is_psd_tensor
from torchjd.autogram._gramian_utils import flatten
from torchjd.autojac._accumulation import is_tensor_with_jac


def assert_has_jac(t: Tensor) -> None:
    assert is_tensor_with_jac(t)
    assert t.jac is not None and t.jac.shape[1:] == t.shape


def assert_has_no_jac(t: Tensor) -> None:
    assert not is_tensor_with_jac(t)


def assert_jac_close(t: Tensor, expected_jac: Tensor, **kwargs) -> None:
    assert is_tensor_with_jac(t)
    assert_close(t.jac, expected_jac, **kwargs)


def assert_has_grad(t: Tensor) -> None:
    assert (t.grad is not None) and (t.shape == t.grad.shape)


def assert_has_no_grad(t: Tensor) -> None:
    assert t.grad is None


def assert_grad_close(t: Tensor, expected_grad: Tensor, **kwargs) -> None:
    assert t.grad is not None
    assert_close(t.grad, expected_grad, **kwargs)


def assert_is_psd_matrix(matrix: Tensor, **kwargs) -> None:
    assert is_psd_matrix(matrix)
    assert_close(matrix, matrix.mH, **kwargs)

    eig_vals = torch.linalg.eigvalsh(matrix)
    expected_eig_vals = eig_vals.clamp(min=0.0)

    assert_close(eig_vals, expected_eig_vals, **kwargs)


def assert_is_psd_tensor(t: Tensor, **kwargs) -> None:
    assert is_psd_tensor(t)
    matrix = flatten(t)
    assert_is_psd_matrix(matrix, **kwargs)
