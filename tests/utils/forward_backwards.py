from collections.abc import Callable

import torch
from torch import Tensor, nn, vmap
from torch.nn.functional import mse_loss
from torch.utils._pytree import PyTree, tree_flatten, tree_map
from torch.utils.hooks import RemovableHandle

from torchjd._linalg import PSDTensor
from torchjd.aggregation import Aggregator, Weighting
from torchjd.autogram import Engine
from torchjd.autojac import backward
from torchjd.autojac._jac_to_grad import jac_to_grad
from utils.architectures import get_in_out_shapes
from utils.contexts import fork_rng


def autograd_forward_backward(
    model: nn.Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], list[Tensor]],
) -> None:
    losses = forward_pass(model, inputs, loss_fn, reduce_to_vector)
    losses.sum().backward()


def autojac_forward_backward(
    model: nn.Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], list[Tensor]],
    aggregator: Aggregator,
) -> None:
    losses = forward_pass(model, inputs, loss_fn, reduce_to_vector)
    backward(losses)
    jac_to_grad(list(model.parameters()), aggregator)


def autograd_gramian_forward_backward(
    model: nn.Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], list[Tensor]],
    weighting: Weighting,
) -> None:
    losses = forward_pass(model, inputs, loss_fn, reduce_to_vector)
    gramian = compute_gramian_with_autograd(losses, list(model.parameters()), retain_graph=True)
    losses.backward(weighting(gramian))


def autogram_forward_backward(
    model: nn.Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], list[Tensor]],
    engine: Engine,
    weighting: Weighting,
) -> None:
    losses = forward_pass(model, inputs, loss_fn, reduce_to_vector)
    gramian = engine.compute_gramian(losses)
    losses.backward(weighting(gramian))


def forward_pass(
    model: nn.Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], list[Tensor]],
    reduction: Callable[[list[Tensor]], Tensor],
) -> PyTree:
    with fork_rng(seed=0):
        output = model(inputs)

    _, expected_output_shapes = get_in_out_shapes(model)
    assert tree_map(lambda t: t.shape[1:], output) == expected_output_shapes

    loss_tensors = loss_fn(output)
    losses = reduction(loss_tensors)
    return losses


def make_mse_loss_fn(targets: PyTree) -> Callable[[PyTree], list[Tensor]]:
    def mse_loss_fn(outputs: PyTree) -> list[Tensor]:
        flat_outputs, _ = tree_flatten(outputs)
        flat_targets, _ = tree_flatten(targets)

        loss_tensors = [
            mse_loss(output, target, reduction="none")
            for output, target in zip(flat_outputs, flat_targets, strict=True)
        ]

        return loss_tensors

    return mse_loss_fn


def reduce_to_first_tensor(loss_tensors: list[Tensor]) -> Tensor:
    return loss_tensors[0]


def reduce_to_matrix(loss_tensors: list[Tensor]) -> Tensor:
    return torch.concat([reshape_raw_losses(t) for t in loss_tensors], dim=1)


def reduce_to_vector(loss_tensors: list[Tensor]) -> Tensor:
    return reduce_to_matrix(loss_tensors).mean(dim=1)


def reduce_to_scalar(loss_tensors: list[Tensor]) -> Tensor:
    return reduce_to_matrix(loss_tensors).mean()


def reshape_raw_losses(raw_losses: Tensor) -> Tensor:
    assert raw_losses.ndim > 0

    if raw_losses.ndim == 1:
        return raw_losses.unsqueeze(1)
    return raw_losses.flatten(start_dim=1)


def compute_gramian_with_autograd(
    output: Tensor,
    params: list[nn.Parameter],
    retain_graph: bool = False,
) -> PSDTensor:
    """
    Computes the Gramian of the Jacobian of the outputs with respect to the params using vmapped
    calls to the autograd engine.
    """

    rg_params = [p for p in params if p.requires_grad]

    def get_vjp(grad_outputs: Tensor) -> list[Tensor]:
        grads = torch.autograd.grad(
            output,
            rg_params,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        return [grad for grad in grads if grad is not None]

    jacobians = vmap(get_vjp)(torch.diag(torch.ones_like(output)))
    jacobian_matrices = [jacobian.reshape([jacobian.shape[0], -1]) for jacobian in jacobians]
    gramian = sum([jacobian @ jacobian.T for jacobian in jacobian_matrices])

    return gramian


class CloneParams:
    """
    ContextManager enabling the computation of per-usage gradients.

    For each submodule with direct trainable parameters, registers:
    - A pre-hook that clones the params before using them, so that gradients will be computed with
      respect to the cloned params.
    - A post-hook that restores the original params.

    The list of clones is returned so that we know where to find the .grad values corresponding to
    each individual usage of a parameter.

    Exiting this context manager takes care of removing hooks and restoring the original params (in
    case an exception occurred before the post-hook could do it).

    Note that this does not work for intra-module parameter reuse, which would require a node-based
    algorithm rather than a module-based algorithm.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.clones = list[nn.Parameter]()
        self._module_to_original_params = dict[nn.Module, dict[str, nn.Parameter]]()
        self._handles: list[RemovableHandle] = []

    def __enter__(self) -> list[nn.Parameter]:
        """Register hooks and return list of (orig_param_id, clone_param)."""

        def pre_hook(module: nn.Module, _) -> None:
            self._module_to_original_params[module] = {}
            for name, param in module.named_parameters():
                if param is None or not param.requires_grad:
                    continue
                self._module_to_original_params[module][name] = param
                clone = nn.Parameter(param.detach().clone().requires_grad_())
                self._set_module_param(module, name, clone)
                self.clones.append(clone)

        def post_hook(module: nn.Module, _, __) -> None:
            self._restore_original_params(module)

        # Register hooks on all modules with direct trainable params
        for mod in self.model.modules():
            if any(p.requires_grad for p in mod.parameters(recurse=False)):
                self._handles.append(mod.register_forward_pre_hook(pre_hook))
                self._handles.append(mod.register_forward_hook(post_hook))

        return self.clones

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove hooks and restore parameters."""
        for handle in self._handles:
            handle.remove()
        for module in self.model.modules():
            self._restore_original_params(module)

        return False  # don't suppress exceptions

    def _restore_original_params(self, module: nn.Module):
        original_params = self._module_to_original_params.pop(module, {})
        for name, param in original_params.items():
            self._set_module_param(module, name, param)

    @staticmethod
    def _set_module_param(module: nn.Module, name: str, param: nn.Parameter) -> None:
        name_parts = name.split(".")
        for module_name in name_parts[:-1]:
            module = module.get_submodule(module_name)
        param_name = name_parts[-1]
        setattr(module, param_name, param)
