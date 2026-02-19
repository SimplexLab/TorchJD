import weakref
from typing import cast

import torch
from torch import Tensor, nn
from torch.autograd.graph import get_gradient_edge
from torch.utils._pytree import PyTree, tree_flatten, tree_unflatten
from torch.utils.hooks import RemovableHandle as TorchRemovableHandle

from ._edge_registry import EdgeRegistry
from ._gramian_accumulator import GramianAccumulator
from ._gramian_computer import GramianComputer

# Note about import from protected _pytree module:
# PyTorch maintainers plan to make pytree public (see
# https://github.com/pytorch/pytorch/issues/65761, https://github.com/pytorch/pytorch/pull/137400).
# It should also come with better speed, because the current implementation is slow, according to
# https://github.com/pytorch/pytorch/issues/65761#issue-1010116111.
# When pytree becomes public, this import will have to be changed with a conditional import (to
# still support older versions of PyTorch where pytree is protected).


class ModuleHookManager:
    """
    Class responsible for handling hooks and Nodes that computes the Gramian reverse accumulation.

    :param target_edges: Registry for tracking gradient edges that serve as targets for the first
        differentiation.
    :param gramian_accumulator: Accumulator for collecting the Jacobians into a Gramian.
    """

    def __init__(
        self,
        target_edges: EdgeRegistry,
        gramian_accumulator: GramianAccumulator,
    ) -> None:
        self._target_edges = target_edges
        self._gramian_accumulator = gramian_accumulator
        self.gramian_accumulation_phase = BoolRef(False)
        self._handles: list[TorchRemovableHandle] = []

        # When the ModuleHookManager is not referenced anymore, there is no reason to keep the hooks
        # alive. In fact, keeping the hooks alive would also keep the target edges alive, which
        # would keep the graph or part of the graph alive. Since the graph contains nodes that store
        # the module in their context, which themselves reference their hooks, the hooks will be
        # caught in a reference cycle and will not be freed by the garbage collector. It is thus
        # important to remove the hooks whenever we're sure we won't need them anymore.
        # We could have used a __del__ method here, with the same effects, but weakref.finalize
        # seems to be a better practice (and it only works if the function to call is static).
        self._finalizer = weakref.finalize(self, ModuleHookManager.remove_hooks, self._handles)

    def hook_module(self, module: nn.Module, gramian_computer: GramianComputer) -> None:
        """
        Add a module hook used to insert Jacobian accumulation nodes into the backward graph.

        The hook injects a AutogramNode function into the computation graph after the module,
        enabling Gramian computation.
        """

        hook = Hook(
            self.gramian_accumulation_phase,
            self._target_edges,
            self._gramian_accumulator,
            gramian_computer,
        )
        self._handles.append(module.register_forward_hook(hook, with_kwargs=True))

    @staticmethod
    def remove_hooks(handles: list[TorchRemovableHandle]) -> None:
        """
        Remove all registered hooks. This method is deliberately static so that it can be called by
        weakref.finalize.
        """

        for handle in handles:
            handle.remove()


class BoolRef:
    """Class wrapping a boolean value, acting as a reference to this boolean value."""

    def __init__(self, value: bool) -> None:
        self.value = value

    def __bool__(self) -> bool:
        return self.value


class Hook:
    def __init__(
        self,
        gramian_accumulation_phase: BoolRef,
        target_edges: EdgeRegistry,
        gramian_accumulator: GramianAccumulator,
        gramian_computer: GramianComputer,
    ) -> None:
        self.gramian_accumulation_phase = gramian_accumulation_phase
        self.target_edges = target_edges
        self.gramian_accumulator = gramian_accumulator
        self.gramian_computer = gramian_computer

    def __call__(
        self,
        _module: nn.Module,
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
        outputs: PyTree,
    ) -> PyTree:
        if self.gramian_accumulation_phase:
            return outputs

        flat_outputs, output_spec = tree_flatten(outputs)

        rg_outputs = list[Tensor]()
        rg_output_indices = list[int]()
        for idx, output in enumerate(flat_outputs):
            if isinstance(output, Tensor) and output.requires_grad:
                rg_outputs.append(output)
                rg_output_indices.append(idx)

        if len(rg_outputs) == 0:
            # This can happen only if a module has a trainable param but outputs no tensor that
            # require grad
            return outputs

        self.gramian_computer.track_forward_call()

        # We only care about running the AutogramNode, so we need one of its child
        # edges (the edges of the original outputs of the model) as target. For memory
        # efficiency, we select the smallest one (that requires grad).
        preference = torch.tensor([t.numel() for t in rg_outputs])
        index = cast(int, preference.argmin().item())
        self.target_edges.register(get_gradient_edge(rg_outputs[index]))

        autograd_fn_rg_outputs = AutogramNode.apply(
            self.gramian_accumulation_phase,
            self.gramian_computer,
            args,
            kwargs,
            self.gramian_accumulator,
            *rg_outputs,
        )

        for idx, output in zip(rg_output_indices, autograd_fn_rg_outputs, strict=True):
            flat_outputs[idx] = output

        return tree_unflatten(flat_outputs, output_spec)


class AutogramNode(torch.autograd.Function):
    """
    Autograd function that is identity on forward and that launches the computation and accumulation
    of the gramian on backward.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(
        _gramian_accumulation_phase: BoolRef,
        _gramian_computer: GramianComputer,
        _args: tuple[PyTree, ...],
        _kwargs: dict[str, PyTree],
        _gramian_accumulator: GramianAccumulator,
        *rg_tensors: Tensor,
    ) -> tuple[Tensor, ...]:
        return tuple(t.detach() for t in rg_tensors)

    # For Python version > 3.10, the type of `inputs` should become
    # tuple[BoolRef, GramianComputer, tuple[PyTree, ...], dict[str, PyTree], GramianAccumulator, *tuple[Tensor, ...]]
    @staticmethod
    def setup_context(
        ctx,
        inputs: tuple,
        _,
    ) -> None:  # type: ignore[reportIncompatibleMethodOverride]
        ctx.gramian_accumulation_phase = inputs[0]
        ctx.gramian_computer = inputs[1]
        ctx.args = inputs[2]
        ctx.kwargs = inputs[3]
        ctx.gramian_accumulator = inputs[4]
        ctx.rg_outputs = inputs[5:]

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> tuple:
        # For python > 3.10: -> tuple[None, None, None, None, None, *tuple[Tensor, ...]]

        if ctx.gramian_accumulation_phase:
            optional_gramian = ctx.gramian_computer(
                ctx.rg_outputs,
                grad_outputs,
                ctx.args,
                ctx.kwargs,
            )
            if optional_gramian is not None:
                ctx.gramian_accumulator.accumulate_gramian(optional_gramian)

        return None, None, None, None, None, *grad_outputs
