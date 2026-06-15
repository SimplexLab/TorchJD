# Partly adapted from https://github.com/Cranial-XIX/FAMO — MIT License, Copyright (c) 2023 Bo Liu.
# See NOTICES for the full license text.
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn.functional import softmax
from torch.optim import Adam

from torchjd._mixins import Stateful

from ._scalarizer_base import Scalarizer

_EPSILON = 1e-8


class FAMO(Scalarizer, Stateful):
    r"""
    :class:`~torchjd.Stateful`
    :class:`~torchjd.scalarization.Scalarizer` that combines the input tensor of values using Fast
    Adaptive Multitask Optimization (FAMO), proposed in `FAMO: Fast Adaptive Multitask Optimization
    <https://proceedings.neurips.cc/paper_files/paper/2023/file/b2fe1ee8d936ac08dd26f2ff58986c8f-Paper-Conference.pdf>`_.

    FAMO decreases all task losses at an approximately equal rate while using only the loss values,
    so it never needs the per-task gradients. The values are combined as

    .. math::
        c \sum_i z_i \log(\ell_i - b_i + \epsilon), \qquad
        z = \mathrm{softmax}(w), \qquad
        c = \left( \sum_i \frac{z_i}{\ell_i - b_i + \epsilon} \right)^{-1}

    where:

    - :math:`\ell_i` is the :math:`i`-th value (typically the loss of task :math:`i`);
    - :math:`b_i` is the lower bound on the :math:`i`-th loss (the ``min_losses`` parameter,
      ``0`` by default);
    - :math:`w_i` is the task-weighting logit of task :math:`i`, learned internally by FAMO;
    - :math:`z = \mathrm{softmax}(w)` are the task weights;
    - :math:`c` is a normalization constant (treated as a constant in the backward pass) that makes
      the resulting update a convex combination of the task gradients;
    - :math:`\epsilon` is a small positive constant for numerical stability.

    Backpropagating this scalarized loss gives FAMO's balanced update direction for the model.

    The task-weighting logits :math:`w` are not learned through that backward pass. Instead, after
    the model has been updated, call :meth:`update` with the losses recomputed on the same batch. It
    measures how much each loss changed across the step,

    .. math::
        \delta_i = \log(\ell_i^{\text{before}} - b_i + \epsilon)
        - \log(\ell_i^{\text{after}} - b_i + \epsilon),

    and takes an `Adam <https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html>`_ step
    on :math:`w` in that direction. FAMO owns this ``Adam`` internally
    (configured by ``lr`` and ``weight_decay``), so you only call the scalarizer and then
    :meth:`update`; there is no second optimizer to manage.

    :param shape: The shape of the values to scalarize, used to create one task-weighting logit per
        value. An ``int`` ``n`` is interpreted as the shape ``(n,)``.
    :param min_losses: The per-task lower bound :math:`b` subtracted from the values before the
        logarithm. If provided, it must have the shape given by ``shape``. If ``None``, zeros are
        used, in which case the values must be strictly positive.
    :param lr: Learning rate of the internal ``Adam`` that learns the task-weighting logits. Must be
        non-negative. The paper uses ``0.025``.
    :param weight_decay: Weight decay of the internal ``Adam``, i.e. the paper's regularization
        coefficient on the logits. Must be non-negative. Defaults to ``1e-3`` (as in the paper's
        Algorithm 2 and in LibMTL); the official implementation uses ``1e-5``.

    The following example shows how to do one iteration of training of a model with FAMO. The losses
    are recomputed on the same batch after the model step so that :meth:`update` can adjust the
    weights.

        >>> import torch
        >>> from torch.nn import Linear
        >>>
        >>> from torchjd.scalarization import FAMO
        >>>
        >>> model = Linear(3, 2)
        >>> scalarizer = FAMO(2)  # Move to the right device with e.g. FAMO(2).to(device="cuda")
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>>
        >>> features = torch.randn(8, 3)
        >>> losses = model(features).pow(2).mean(dim=0)  # One loss per output dimension.
        >>> loss = scalarizer(losses)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
        >>>
        >>> # Recompute the losses on the same batch, after the model update.
        >>> new_losses = model(features).pow(2).mean(dim=0)
        >>> scalarizer.update(new_losses)  # Updates the task weights internally.

    .. note::
        FAMO takes the logarithm of :math:`\ell_i - b_i`, so each value must stay strictly above its
        lower bound :math:`b_i` (the paper assumes non-negative losses). With the default
        ``min_losses`` of zeros, this means the values must be strictly positive. This precondition
        is not enforced.

    .. note::
        This implementation was adapted from the `official implementation
        <https://github.com/Cranial-XIX/FAMO>`_.
    """

    min_losses: Tensor

    def __init__(
        self,
        shape: int | Sequence[int],
        min_losses: Tensor | None = None,
        lr: float = 0.025,
        weight_decay: float = 1e-3,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Parameter `lr` should be non-negative. Found `lr = {lr}`.")
        if weight_decay < 0.0:
            raise ValueError(
                f"Parameter `weight_decay` should be non-negative. Found `weight_decay = "
                f"{weight_decay}`."
            )

        super().__init__()
        self._w = nn.Parameter(torch.zeros(shape))

        if min_losses is None:
            min_losses = torch.zeros(self._w.shape)
        elif min_losses.shape != self._w.shape:
            raise ValueError(
                f"Parameter `min_losses` should have shape {tuple(self._w.shape)} (matching the "
                f"shape of the logits). Found `min_losses.shape = {tuple(min_losses.shape)}`."
            )
        self.register_buffer("min_losses", min_losses)

        self.lr = lr
        self.weight_decay = weight_decay
        self._optimizer: Adam | None = None
        self._prev_losses: Tensor | None = None

    def forward(self, values: Tensor, /) -> Tensor:
        self._check_shape(values)

        self._prev_losses = values.detach().clone()

        weights = softmax(self._w.flatten(), dim=0).reshape(values.shape).detach()
        shifted = values - self.min_losses + _EPSILON
        normalizer = (weights / shifted).sum().detach()
        return ((weights / normalizer) * torch.log(shifted)).sum()

    def update(self, values: Tensor, /) -> None:
        """
        Updates the task-weighting logits from the change in losses across the model update, by
        taking one step of the internal ``Adam``. Must be called after the scalarizer has been
        called on the batch's losses, with the losses recomputed on the same batch after the model
        step.
        """

        if self._prev_losses is None:
            raise ValueError(
                "`update` must be called after the scalarizer is called on the losses."
            )
        self._check_shape(values)

        before = self._prev_losses - self.min_losses + _EPSILON
        after = values.detach() - self.min_losses + _EPSILON
        delta = torch.log(before) - torch.log(after)

        with torch.enable_grad():
            weights = softmax(self._w.flatten(), dim=0)
            grad = torch.autograd.grad(weights, self._w, grad_outputs=delta.flatten())[0]

        if self._optimizer is None:
            self._optimizer = Adam([self._w], lr=self.lr, weight_decay=self.weight_decay)
        self._w.grad = grad
        self._optimizer.step()
        # Clear the gradient so it cannot leak into a user optimizer that the logits were mistakenly
        # added to: FAMO is the only thing that should step them.
        self._w.grad = None

    def reset(self) -> None:
        with torch.no_grad():
            self._w.zero_()
        self._optimizer = None
        self._prev_losses = None

    def _check_shape(self, values: Tensor) -> None:
        if values.shape != self._w.shape:
            raise ValueError(
                f"Parameter `values` should have shape {tuple(self._w.shape)} (matching the shape "
                f"of the logits). Found `values.shape = {tuple(values.shape)}`."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(shape={tuple(self._w.shape)}, lr={self.lr}, "
            f"weight_decay={self.weight_decay})"
        )
