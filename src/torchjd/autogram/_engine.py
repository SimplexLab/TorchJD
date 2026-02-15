from typing import cast

import torch
from torch import Tensor, nn, vmap
from torch.autograd.graph import get_gradient_edge

from torchjd._linalg import PSDMatrix

from ._edge_registry import EdgeRegistry
from ._gramian_accumulator import GramianAccumulator
from ._gramian_computer import GramianComputer, JacobianBasedGramianComputerWithCrossTerms
from ._gramian_utils import movedim, reshape
from ._jacobian_computer import (
    AutogradJacobianComputer,
    FunctionalJacobianComputer,
    JacobianComputer,
)
from ._module_hook_manager import ModuleHookManager

_MODULES_INCOMPATIBLE_WITH_BATCHED = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LazyBatchNorm1d,
    nn.LazyBatchNorm2d,
    nn.LazyBatchNorm3d,
    nn.SyncBatchNorm,
    nn.RNNBase,
)

_TRACK_RUNNING_STATS_MODULE_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LazyBatchNorm1d,
    nn.LazyBatchNorm2d,
    nn.LazyBatchNorm3d,
    nn.SyncBatchNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LazyInstanceNorm1d,
    nn.LazyInstanceNorm2d,
    nn.LazyInstanceNorm3d,
)


class Engine:
    """
    Engine to compute the Gramian of the Jacobian of some tensor with respect to the direct
    parameters of all provided modules. It is based on Algorithm 3 of `Jacobian Descent For
    Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_ but goes even further:

    * It works for any computation graph (not just sequential models).
    * It is optimized for batched computations (as long as ``batch_dim`` is specified).
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
    :param batch_dim: If the modules work with batches and process each batch element independently,
        then many intermediary Jacobians are sparse (block-diagonal), which allows for a substantial
        memory optimization by backpropagating a squashed Jacobian instead. This parameter indicates
        the batch dimension of the output tensor, if any.

    .. admonition::
        Example

        Train a model using Gramian-based Jacobian descent.

        .. code-block:: python
            :emphasize-lines: 5-6, 15-16, 18-19, 26-29

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
            engine = Engine(model, batch_dim=0)

            for input, target in zip(inputs, targets):
                output = model(input).squeeze(dim=1)  # shape: [16]
                losses = criterion(output, target)  # shape: [16]

                gramian = engine.compute_gramian(losses)  # shape: [16, 16]
                weights = weighting(gramian)  # shape: [16]
                losses.backward(weights)
                optimizer.step()
                optimizer.zero_grad()

        This is equivalent to just calling ``torchjd.autojac.backward(losses, UPGrad())``. However,
        since the Jacobian never has to be entirely in memory, it is often much more
        memory-efficient, and thus typically faster, to use the Gramian-based approach.

    .. warning::
        When providing a non-None ``batch_dim``, all provided modules must respect a few conditions:

        * They should treat the elements of the batch independently. Most common layers respect
          this, but for example `BatchNorm
          <https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>`_ does not (it
          computes some average and standard deviation over the elements of the batch).
        * Their inputs and outputs can be anything, but each input tensor and each output tensor
          must be batched on its first dimension. When available (e.g. in `Transformers
          <https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`_,
          `MultiheadAttention
          <https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>`_,
          etc.), the ``batch_first`` parameter has to be set to ``True``. Also, this makes `RNNs
          <https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html>`_ not supported yet
          because their hidden state is batched on dimension 1 even if ``batch_first`` is ``True``.
        * They should not perform in-place operations on tensors (for instance you should not use
          ``track_running_stats=True`` in normalization layers).
        * They should not have side effects during the forward pass (since their forward pass will
          be called twice, the side effects could be different from what's expected).
        * If they have some randomness during the forward pass, they should not have direct
          trainable parameters. For this reason,
          `Transformers
          <https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`_, which use a
          dropout function (rather than a `Dropout
          <https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html>`_ layer) in a
          module with some trainable parameters, has to be used with
          ``dropout=0.0``. Note that a `Dropout
          <https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html>`_ layers are
          entirely supported and should be preferred. It is also perfectly fine for random modules
          to have child modules that have trainable parameters, so if you have a random module with
          some direct parameters, a simple fix is to wrap these parameters into a child module.

        If you're building your own architecture, respecting those criteria should be quite easy.
        However, if you're using an existing architecture, you may have to modify it to make it
        compatible with the autogram engine. For instance, you may want to replace `BatchNorm2d
        <https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>`_ layers by
        `GroupNorm <https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html>`_ or
        `InstanceNorm2d
        <https://docs.pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html>`_ layers.

        The alternative is to use ``batch_dim=None``, but it's not recommended since it will
        increase memory usage by a lot and thus typically slow down computation.

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

    def __init__(
        self,
        *modules: nn.Module,
        batch_dim: int | None,
    ):
        self._gramian_accumulator = GramianAccumulator()
        self._target_edges = EdgeRegistry()
        self._batch_dim = batch_dim
        self._module_hook_manager = ModuleHookManager(self._target_edges, self._gramian_accumulator)
        self._gramian_computers = dict[nn.Module, GramianComputer]()

        for module in modules:
            self._hook_module_recursively(module)

    def _hook_module_recursively(self, module: nn.Module) -> None:
        if any(p.requires_grad for p in module.parameters(recurse=False)):
            self._check_module_is_compatible(module)
            gramian_computer = self._make_gramian_computer(module)
            self._gramian_computers[module] = gramian_computer
            self._module_hook_manager.hook_module(module, gramian_computer)
        else:
            for child in module.children():
                self._hook_module_recursively(child)

    def _make_gramian_computer(self, module: nn.Module) -> GramianComputer:
        jacobian_computer: JacobianComputer
        if self._batch_dim is not None:
            jacobian_computer = FunctionalJacobianComputer(module)
        else:
            jacobian_computer = AutogradJacobianComputer(module)
        gramian_computer = JacobianBasedGramianComputerWithCrossTerms(jacobian_computer)

        return gramian_computer

    def _check_module_is_compatible(self, module: nn.Module) -> None:
        if self._batch_dim is not None:
            if isinstance(module, _MODULES_INCOMPATIBLE_WITH_BATCHED):
                raise ValueError(
                    f"Found a module of type {type(module)}, which is incompatible with the "
                    f"autogram engine when `batch_dim` is not `None`. The incompatible module types"
                    f" are {_MODULES_INCOMPATIBLE_WITH_BATCHED} (and their subclasses). The "
                    f"recommended fix is to replace incompatible layers by something else (e.g. "
                    f"BatchNorm by InstanceNorm). If you really can't and performance is not a "
                    f"priority, you may also just set `batch_dim=None` when creating the engine.",
                )
            if isinstance(module, _TRACK_RUNNING_STATS_MODULE_TYPES) and module.track_running_stats:
                raise ValueError(
                    f"Found a module of type {type(module)}, with `track_running_stats=True`, which"
                    f" is incompatible with the autogram engine when `batch_dim` is not `None`, due"
                    f" to performing in-place operations on tensors and having side-effects during "
                    f"the forward pass. Try setting `track_running_stats` to `False`. If you really"
                    f" can't and performance is not a priority, you may also just set "
                    f"`batch_dim=None` when creating the engine.",
                )

    # Currently, the type PSDMatrix is hidden from users, so Tensor is correct.
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

        if self._batch_dim is not None:
            # move batched dim to the end
            ordered_output = output.movedim(self._batch_dim, -1)
            ordered_shape = list(ordered_output.shape)
            batch_size = ordered_shape[-1]
            has_non_batch_dim = len(ordered_shape) > 1
            target_shape = [batch_size]
        else:
            ordered_output = output
            ordered_shape = list(ordered_output.shape)
            has_non_batch_dim = len(ordered_shape) > 0
            target_shape = []

        if has_non_batch_dim:
            target_shape = [-1, *target_shape]

        reshaped_output = ordered_output.reshape(target_shape)
        # There are four different cases for the shape of reshaped_output:
        # - Not batched and not non-batched: scalar of shape []
        # - Batched only: vector of shape [batch_size]
        # - Non-batched only: vector of shape [dim]
        # - Batched and non-batched: matrix of shape [dim, batch_size]

        self._module_hook_manager.gramian_accumulation_phase.value = True

        try:
            square_gramian = self._compute_square_gramian(reshaped_output, has_non_batch_dim)
        finally:
            # Reset everything that has a state, even if the previous call raised an exception
            self._module_hook_manager.gramian_accumulation_phase.value = False
            self._gramian_accumulator.reset()
            self._target_edges.reset()
            for gramian_computer in self._gramian_computers.values():
                gramian_computer.reset()

        unordered_gramian = reshape(square_gramian, ordered_shape)

        if self._batch_dim is not None:
            gramian = movedim(unordered_gramian, [-1], [self._batch_dim])
        else:
            gramian = unordered_gramian

        return gramian

    def _compute_square_gramian(self, output: Tensor, has_non_batch_dim: bool) -> PSDMatrix:
        leaf_targets = list(self._target_edges.get_leaf_edges({get_gradient_edge(output)}))

        def differentiation(_grad_output: Tensor) -> tuple[Tensor, ...]:
            return torch.autograd.grad(
                outputs=output,
                inputs=leaf_targets,
                grad_outputs=_grad_output,
                retain_graph=True,
            )

        if has_non_batch_dim:
            # There is one non-batched dimension, it is the first one
            non_batch_dim_len = output.shape[0]
            identity_matrix = torch.eye(non_batch_dim_len, device=output.device, dtype=output.dtype)
            ones = torch.ones_like(output[0])
            jac_output = torch.einsum("ij, ... -> ij...", identity_matrix, ones)

            _ = vmap(differentiation)(jac_output)
        else:
            grad_output = torch.ones_like(output)
            _ = differentiation(grad_output)

        # If the gramian were None, then leaf_targets would be empty, so autograd.grad would
        # have failed. So gramian is necessarily a valid Tensor here.
        gramian = cast(PSDMatrix, self._gramian_accumulator.gramian)

        return gramian
