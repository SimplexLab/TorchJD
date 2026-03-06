from functools import partial

import torch
from settings import DEVICE, DTYPE
from torch import nn
from torch.utils._pytree import PyTree, tree_map

from utils.architectures import get_in_out_shapes
from utils.contexts import fork_rng

# Curried calls to torch functions that require a device so that we automatically fix the device
# for code written in the tests, while not affecting code written in src (what
# torch.set_default_device or what a too large `with torch.device(DEVICE)` context would have done).

# Default device is most likely int.
arange_ = partial(torch.arange, device=DEVICE)
randint_ = partial(torch.randint, device=DEVICE)
randperm_ = partial(torch.randperm, device=DEVICE)

# Default device is most likely float. Set it to the right kind of float.
empty_ = partial(torch.empty, device=DEVICE, dtype=DTYPE)
eye_ = partial(torch.eye, device=DEVICE, dtype=DTYPE)
ones_ = partial(torch.ones, device=DEVICE, dtype=DTYPE)
rand_ = partial(torch.rand, device=DEVICE, dtype=DTYPE)
randn_ = partial(torch.randn, device=DEVICE, dtype=DTYPE)
tensor_ = partial(torch.tensor, device=DEVICE, dtype=DTYPE)
zeros_ = partial(torch.zeros, device=DEVICE, dtype=DTYPE)


def make_inputs_and_targets(model: nn.Module, batch_size: int) -> tuple[PyTree, PyTree]:
    input_shapes, output_shapes = get_in_out_shapes(model)
    with fork_rng(seed=0):
        inputs = _make_tensors(batch_size, input_shapes)
        targets = _make_tensors(batch_size, output_shapes)

    return inputs, targets


def _make_tensors(batch_size: int, tensor_shapes: PyTree) -> PyTree:
    def is_leaf(s: PyTree) -> bool:
        return isinstance(s, tuple) and all(isinstance(e, int) for e in s)

    return tree_map(lambda s: randn_((batch_size, *s)), tensor_shapes, is_leaf=is_leaf)
