from typing import cast

import torch
from torch import Tensor, nn, vmap
from torch.autograd.graph import get_gradient_edge

from torchjd.sparse import make_slt

from ._edge_registry import EdgeRegistry
from ._gramian_accumulator import GramianAccumulator
from ._gramian_computer import GramianComputer, JacobianBasedGramianComputerWithCrossTerms
from ._jacobian_computer import AutogradJacobianComputer
from ._module_hook_manager import ModuleHookManager


class Engine:
    """
    Engine to compute the Gramian of the Jacobian of some tensor with respect to the direct
    parameters of all provided modules. It is based on Algorithm 3 of `Jacobian Descent For
    Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_ but goes even further:

    * It works for any computation graph (not just sequential models).
    * It is highly optimized for batched computations but also supports non-batched computations.
    * It supports any shape of tensor to differentiate (not just a vector of losses). For more
      details about this, look at :meth:`Engine.compute_gramian`.

    As explained in Section 6 of `Jacobian Descent For Multi-Objective Optimization
    <https://arxiv.org/pdf/2406.16232>`_, most :class:`Aggregators
    <torchjd.aggregation._aggregator_bases.Aggregator>` combine the rows of the Jacobian using some
    weights that depend only on the Gramian of the Jacobian. Because of that, the typical usage of
    the autogram engine is to directly compute this Gramian, extract weights from it (with a
    :class:`~torchjd.aggregation._weighting_bases.Weighting`), and use those weights to
    backpropagate the losses. This is equivalent to doing a step of standard Jacobian descent using
    :func:`torchjd.autojac.backward`.

    :param modules: The modules whose parameters will contribute to the Gramian of the Jacobian.
        Several modules can be provided, but it's important that none of them is a child module of
        another of them.

    .. admonition::
        Example

        Train a model using Gramian-based Jacobian descent.

        .. code-block:: python
            :emphasize-lines: 5-6, 15-16, 18-19, 26-28

            import torch
            from torch.nn import Linear, MSELoss, ReLU, Sequential
            from torch.optim import SGD

            from torchjd.aggregation import UPGradWeighting
            from torchjd.autogram import Engine

            # Generate data (8 batches of 16 examples of dim 5) for the sake of the example
            inputs = torch.randn(8, 16, 5)
            targets = torch.randn(8, 16)

            model = Sequential(Linear(5, 4), ReLU(), Linear(4, 1))
            optimizer = SGD(model.parameters())

            criterion = MSELoss(reduction="none")  # Important to use reduction="none"
            weighting = UPGradWeighting()

            # Create the engine before the backward pass, and only once.
            engine = Engine(model)

            for input, target in zip(inputs, targets):
                output = model(input).squeeze(dim=1)  # shape: [16]
                losses = criterion(output, target)  # shape: [16]

                optimizer.zero_grad()
                gramian = engine.compute_gramian(losses)  # shape: [16, 16]
                weights = weighting(gramian)  # shape: [16]
                losses.backward(weights)
                optimizer.step()

        This is equivalent to just calling ``torchjd.autojac.backward(losses, UPGrad())``. However,
        since the Jacobian never has to be entirely in memory, it is often much more
        memory-efficient, and thus typically faster, to use the Gramian-based approach.

    .. warning:: For autogram to be fast and low-memory, it is very important to use only batched
        modules (i.e. modules that treat each element of the batch independently). For instance,
        BatchNorm is not a batched module because it computes some statistics over the batch.

    .. warning::
        `RNNs <https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html>`_ may not be
        supported on cuda because vmap is not implemented for RNN on that device.

    .. warning::
        Parent modules should call their child modules directly rather than using their child
        modules' parameters themselves. For instance, the following model is not supported:

        >>> class Model(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.linear = nn.Linear(2, 3)  # Child module
        >>>
        >>>     def forward(self, input: Tensor) -> Tensor:
        >>>         # Incorrect: Use the child module's parameters directly without calling it.
        >>>         return input @ self.linear.weight.T + self.linear.bias
        >>>         # Correct alternative: return self.linear(input)

    .. note::
          For maximum efficiency, modules should ideally not contain both direct trainable
          parameters and child modules, especially if those direct trainable parameters are used
          before the child modules. You can always wrap those direct trainable parameters into
          another child module to avoid the slow-down.
    """

    def __init__(self, *modules: nn.Module):
        self._gramian_accumulator = GramianAccumulator()
        self._target_edges = EdgeRegistry()
        self._module_hook_manager = ModuleHookManager(self._target_edges, self._gramian_accumulator)
        self._gramian_computers = dict[nn.Module, GramianComputer]()

        for module in modules:
            self._hook_module_recursively(module)

    def _hook_module_recursively(self, module: nn.Module) -> None:
        if any(p.requires_grad for p in module.parameters(recurse=False)):
            gramian_computer = self._make_gramian_computer(module)
            self._gramian_computers[module] = gramian_computer
            self._module_hook_manager.hook_module(module, gramian_computer)
        else:
            for child in module.children():
                self._hook_module_recursively(child)

    def _make_gramian_computer(self, module: nn.Module) -> GramianComputer:
        jacobian_computer = AutogradJacobianComputer(module)
        gramian_computer = JacobianBasedGramianComputerWithCrossTerms(jacobian_computer)

        return gramian_computer

    def compute_gramian(self, output: Tensor) -> Tensor:
        r"""
        Computes the Gramian of the Jacobian of ``output`` with respect to the direct parameters of
        all ``modules``.

        :param output: The tensor of arbitrary shape to differentiate. The shape of the returned
            Gramian depends on the shape of this output.

        .. note::
            This function doesn't require ``output`` to be a vector. For example, if ``output`` is
            a matrix of shape :math:`[m_1, m_2]`, its Jacobian :math:`J` with respect to the
            parameters will be of shape :math:`[m_1, m_2, n]`, where :math:`n` is the number of
            parameters in the model. This is what we call a `generalized Jacobian`. The
            corresponding Gramian :math:`G = J J^\top` will be of shape
            :math:`[m_1, m_2, m_2, m_1]`. This is what we call a `generalized Gramian`. The number
            of dimensions of the returned generalized Gramian will always be twice that of the
            ``output``.

            A few examples:
                - 0D (scalar) ``output``: 0D Gramian (this can be used to efficiently compute the
                  squared norm of the gradient of ``output``).
                - 1D (vector) ``output``: 2D Gramian (this is the standard setting of Jacobian
                  descent).
                - 2D (matrix) ``output``: 4D Gramian (this can be used for :doc:`Instance-Wise
                  Multi-Task Learning (IWMTL) <../../examples/iwmtl>`, as each sample in the batch
                  has one loss per task).
                - etc.
        """

        self._module_hook_manager.gramian_accumulation_phase.value = True

        try:
            leaf_targets = list(self._target_edges.get_leaf_edges({get_gradient_edge(output)}))

            def differentiation(_grad_output: Tensor) -> tuple[Tensor, ...]:
                return torch.autograd.grad(
                    outputs=output,
                    inputs=leaf_targets,
                    grad_outputs=_grad_output,
                    retain_graph=True,
                )

            output_dims = list(range(output.ndim))
            identity = torch.eye(output.ndim, dtype=torch.int64)
            basis = torch.concatenate([identity, identity], dim=0)
            jac_output = make_slt(torch.ones_like(output), basis, None, None)

            vmapped_diff = differentiation
            for _ in output_dims:
                vmapped_diff = vmap(vmapped_diff)

            _ = vmapped_diff(jac_output)

            # If the gramian were None, then leaf_targets would be empty, so autograd.grad would
            # have failed. So gramian is necessarily a valid Tensor here.
            gramian = cast(Tensor, self._gramian_accumulator.gramian)
        finally:
            # Reset everything that has a state, even if the previous call raised an exception
            self._module_hook_manager.gramian_accumulation_phase.value = False
            self._gramian_accumulator.reset()
            self._target_edges.reset()
            for gramian_computer in self._gramian_computers.values():
                gramian_computer.reset()

        return gramian
