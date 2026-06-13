from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn.functional import softmax

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
    - :math:`w_i` is the learnable task-weighting logit of task :math:`i`, stored as an
      ``nn.Parameter``;
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

    and sets the gradient of :math:`w` to the matching softmax vector-Jacobian product. Your own
    optimizer then steps :math:`w`. To match the paper, use ``Adam`` with a weight decay equal to
    the paper's regularization coefficient.

    :param shape: The shape of the values to scalarize, used to create one task-weighting logit per
        value. An ``int`` ``n`` is interpreted as the shape ``(n,)``.
    :param min_losses: The per-task lower bound :math:`b` subtracted from the values before the
        logarithm. If provided, it must have the shape given by ``shape``. If ``None``, zeros are
        used, in which case the values must be strictly positive.

    The following example shows how to train a model with FAMO. The model and the task-weighting
    logits use two separate optimizers, and the losses are recomputed on the same batch after the
    model step so that :meth:`update` can adjust the weights.

        >>> import torch
        >>> from torch.nn import Linear
        >>>
        >>> from torchjd.scalarization import FAMO
        >>>
        >>> model = Linear(3, 2)
        >>> scalarizer = FAMO(2)  # Move to the right device with e.g. FAMO(2).to(device="cuda")
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> weight_optimizer = torch.optim.Adam(
        ...     scalarizer.parameters(), lr=0.025, weight_decay=0.001
        ... )
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
        >>> scalarizer.update(new_losses)  # Sets the gradient of the task-weighting logits.
        >>> weight_optimizer.step()

    .. note::
        FAMO takes the logarithm of :math:`\ell_i - b_i`, so each value must stay strictly above its
        lower bound :math:`b_i` (the paper assumes non-negative losses). With the default
        ``min_losses`` of zeros, this means the values must be strictly positive. This precondition
        is not enforced.
    """

    min_losses: Tensor

    def __init__(self, shape: int | Sequence[int], min_losses: Tensor | None = None) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(shape))

        if min_losses is None:
            min_losses = torch.zeros(self.w.shape)
        elif min_losses.shape != self.w.shape:
            raise ValueError(
                f"Parameter `min_losses` should have shape {tuple(self.w.shape)} (matching the "
                f"shape of the logits). Found `min_losses.shape = {tuple(min_losses.shape)}`."
            )
        self.register_buffer("min_losses", min_losses)

        self._prev_losses: Tensor | None = None

    def forward(self, values: Tensor, /) -> Tensor:
        self._check_shape(values)

        self._prev_losses = values.detach().clone()

        weights = softmax(self.w.flatten(), dim=0).reshape(values.shape).detach()
        shifted = values - self.min_losses + _EPSILON
        normalizer = (weights / shifted).sum().detach()
        return ((weights / normalizer) * torch.log(shifted)).sum()

    def update(self, values: Tensor, /) -> None:
        """
        Sets the gradient of the task-weighting logits from the change in losses across the model
        update, overwriting any existing gradient. Must be called after the scalarizer has been
        called on the batch's losses, with the losses recomputed on the same batch after the model
        step. The logits are not stepped here; call your own optimizer's ``step`` afterwards.
        """

        if self._prev_losses is None:
            raise ValueError("`update` must be called after `forward`, which stores the losses.")
        self._check_shape(values)

        before = self._prev_losses - self.min_losses + _EPSILON
        after = values.detach() - self.min_losses + _EPSILON
        delta = torch.log(before) - torch.log(after)

        with torch.enable_grad():
            weights = softmax(self.w.flatten(), dim=0)
            grad = torch.autograd.grad(weights, self.w, grad_outputs=delta.flatten())[0]
        self.w.grad = grad

    def reset(self) -> None:
        with torch.no_grad():
            self.w.zero_()
        self._prev_losses = None

    def _check_shape(self, values: Tensor) -> None:
        if values.shape != self.w.shape:
            raise ValueError(
                f"Parameter `values` should have shape {tuple(self.w.shape)} (matching the shape "
                f"of the logits). Found `values.shape = {tuple(values.shape)}`."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={tuple(self.w.shape)})"
