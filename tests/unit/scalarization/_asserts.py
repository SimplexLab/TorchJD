import torch
from torch import Tensor
from utils.tensors import randperm_

from torchjd.scalarization import Scalarizer


def assert_returns_scalar(scalarizer: Scalarizer, losses: Tensor) -> None:
    out = scalarizer(losses)
    assert out.dim() == 0
    assert out.isfinite()


def assert_grad_flow(scalarizer: Scalarizer, losses: Tensor) -> None:
    leaf = losses.detach().requires_grad_()
    out = scalarizer(leaf)
    out.backward()
    assert leaf.grad is not None
    assert leaf.grad.isfinite().all()


def assert_permutation_invariant(scalarizer: Scalarizer, losses: Tensor) -> None:
    out = scalarizer(losses)
    flat = losses.flatten()
    permuted = flat[randperm_(flat.numel())].reshape(losses.shape)
    out_permuted = scalarizer(permuted)
    torch.testing.assert_close(out, out_permuted)
