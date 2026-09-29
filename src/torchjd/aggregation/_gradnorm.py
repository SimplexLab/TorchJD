from math import isfinite

import torch
from torch import Tensor, nn

from torchjd._mixins import Stateful
from torchjd.linalg import PSDMatrix

from ._aggregator_bases import GramianWeightedAggregator
from ._mixins import _NonDifferentiable
from ._weighting_bases import _GramianWeighting


class GradNormWeighting(_GramianWeighting, Stateful, _NonDifferentiable):
    r"""
    :class:`~torchjd.Stateful`
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.PSDMatrix`] from
    `GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks
    <https://proceedings.mlr.press/v80/chen18a.html>`_ (ICML 2018).

    The trainable parameter ``weights`` starts at one for each task. Call :meth:`set_losses`
    before each forward, then minimise :meth:`balancing_loss` with an external optimizer and
    call :meth:`renormalize` after its step. The forward returns detached weights for the
    model's backward pass. It does not update the weights or accumulate their gradients.

    For a Gramian :math:`G = JJ^\top`, the balancing loss is

    .. math::
        L_{\mathrm{grad}} = \sum_i \left| g_i - \operatorname{stopgrad}
        \left(\bar g r_i^\alpha\right) \right|, \qquad
        g_i = |w_i|\sqrt{G_{ii}}, \qquad
        r_i = \frac{L_i/L_i(0)}{\operatorname{mean}_j(L_j/L_j(0))}.

    Here :math:`w_i` is the learned weight, :math:`L_i` the current task loss,
    :math:`L_i(0)` its value at the first call to :meth:`set_losses` and :math:`\bar g`
    the mean weighted gradient norm. Only ``weights`` receives gradients from this loss.

    :param n_tasks: Number of tasks. Must be positive and remains fixed for this instance.
    :param alpha: Non-negative strength of training-rate balancing. With ``0``, GradNorm
        targets equal gradient norms. The paper uses ``1.5`` for its NYUv2 experiments.

    Move the module to the model's device and dtype before creating its optimizer. Initial
    losses must be finite and strictly positive; later losses may be zero. If all current
    losses are zero, the balancing loss and its weight gradients are zero.

    This follows Algorithm 1's direct weight updates and normalization to a sum of
    ``n_tasks``. LibMTL instead learns softmax logits and uses first-epoch losses as its
    baseline. Choose a weight learning rate that keeps the weights non-negative.

    .. testcode::

        import torch
        from torch.nn import Linear
        from torch.optim import Adam, SGD

        from torchjd.aggregation import GradNormWeighting
        from torchjd.autojac import jac

        model = Linear(3, 2)
        weighting = GradNormWeighting(2)
        model_optimizer = SGD(model.parameters(), lr=0.01)
        weight_optimizer = Adam(weighting.parameters(), lr=0.001)

        for features in torch.randn(4, 8, 3):
            model_optimizer.zero_grad()
            weight_optimizer.zero_grad()
            losses = model(features).square().mean(dim=0)
            jacs = jac(losses, list(model.parameters()), retain_graph=True)
            J = torch.cat([j.flatten(1) for j in jacs], dim=1)
            weighting.set_losses(losses)
            weights = weighting(J @ J.T)
            weighting.balancing_loss().backward()
            losses.backward(weights)
            model_optimizer.step()
            weight_optimizer.step()
            weighting.renormalize()

    See :doc:`the GradNorm example <../../examples/gradnorm>` for computing the norms
    using only the last shared layer while weighting all model parameters.
    """

    _initial_losses: Tensor
    _initialized: Tensor
    _losses: Tensor | None
    _norms: Tensor | None

    def __init__(self, n_tasks: int, alpha: float = 1.5) -> None:
        super().__init__()
        if n_tasks < 1:
            raise ValueError(f"Parameter `n_tasks` must be positive. Found n_tasks={n_tasks!r}.")
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(n_tasks))
        self.register_buffer("_initial_losses", torch.zeros(n_tasks))
        self.register_buffer("_initialized", torch.tensor(False))
        self.register_buffer("_losses", None, persistent=False)
        self.register_buffer("_norms", None, persistent=False)

    @property
    def n_tasks(self) -> int:
        return self.weights.numel()

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"Attribute `alpha` must be finite and non-negative. Found {value!r}.")
        self._alpha = value

    def set_losses(self, losses: Tensor) -> None:
        """
        Stores the current unweighted task losses. The first call also records the baseline
        losses. Call this before each forward, keeping the same task order throughout training.
        """
        if losses.shape != self.weights.shape:
            raise ValueError(f"Parameter `losses` must have shape ({self.n_tasks},).")
        if losses.device != self.weights.device or losses.dtype != self.weights.dtype:
            raise ValueError("Parameter `losses` must have the same device and dtype as `weights`.")
        if not torch.isfinite(losses).all() or (losses < 0).any():
            raise ValueError("Parameter `losses` must be finite and non-negative.")
        if not self._initialized:
            if (losses == 0).any():
                raise ValueError("Initial losses must be strictly positive.")
            self._initial_losses.copy_(losses.detach())
            self._initialized.fill_(True)
        self._losses = losses.detach().clone()
        self._norms = None

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        if self._losses is None:
            raise ValueError("Call `set_losses` before the forward pass.")
        if gramian.shape != (self.n_tasks, self.n_tasks):
            raise ValueError(
                f"Parameter `gramian` must have shape ({self.n_tasks}, {self.n_tasks})."
            )
        if gramian.device != self.weights.device or gramian.dtype != self.weights.dtype:
            raise ValueError(
                "Parameter `gramian` must have the same device and dtype as `weights`."
            )
        self._norms = gramian.detach().diagonal().clamp_min(0).sqrt()
        return self.weights.detach().clone()

    def balancing_loss(self) -> Tensor:
        """
        Computes the auxiliary loss using the most recent forward's gradient norms and losses.
        Backpropagate this scalar before stepping the weights' optimizer. Its gradients affect
        only ``weights``, even if the supplied losses or Gramian have an autograd graph.
        """
        if self._losses is None or self._norms is None:
            raise ValueError("Call `set_losses` and the forward pass before `balancing_loss`.")
        if not self._losses.any():
            return (self.weights * 0).sum()
        ratios = self._losses / self._initial_losses
        rates = ratios / ratios.mean()
        norms = self.weights.abs() * self._norms
        targets = (norms.mean() * rates.pow(self.alpha)).detach()
        return (norms - targets).abs().sum()

    def renormalize(self) -> None:
        """
        Rescales the weights in place to sum to ``n_tasks`` after an optimizer step, as in
        Algorithm 1. Raises if an update produced non-finite or negative weights, or a zero
        sum. Reduce the weight learning rate if updates cross this boundary.
        """
        with torch.no_grad():
            total = self.weights.sum()
            if not torch.isfinite(total) or (self.weights < 0).any() or total <= 0:
                raise ValueError("Weights must be finite and non-negative, with a positive sum.")
            self.weights.mul_(self.n_tasks / total)

    def reset(self) -> None:
        """
        Restores unit weights and clears the loss baseline and cached batch statistics.
        Parameter identity is preserved. Reset the external optimizer separately to discard
        its momentum or other state when starting a new experiment.
        """
        with torch.no_grad():
            self.weights.fill_(1)
            self._initial_losses.zero_()
            self._initialized.fill_(False)
        self.weights.grad = None
        self._losses = None
        self._norms = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_tasks={self.n_tasks}, alpha={self.alpha!r})"


class GradNorm(GramianWeightedAggregator, Stateful, _NonDifferentiable):
    r"""
    :class:`~torchjd.Stateful` :class:`~torchjd.aggregation.GramianWeightedAggregator`
    using :class:`~torchjd.aggregation.GradNormWeighting`.

    Gradient norms are computed over all columns of the supplied Jacobian. For norms based on
    a subset of parameters, use :class:`~torchjd.aggregation.GradNormWeighting` directly.
    Pass ``parameters()`` to a separate optimizer. Call :meth:`set_losses` before aggregation,
    backpropagate :meth:`balancing_loss` and call :meth:`renormalize` after the optimizer step.

    :param n_tasks: Fixed positive number of tasks (rows of the Jacobian).
    :param alpha: Non-negative strength of training-rate balancing.
    """

    gramian_weighting: GradNormWeighting

    def __init__(self, n_tasks: int, alpha: float = 1.5) -> None:
        super().__init__(GradNormWeighting(n_tasks, alpha))

    @property
    def n_tasks(self) -> int:
        return self.gramian_weighting.n_tasks

    @property
    def alpha(self) -> float:
        return self.gramian_weighting.alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self.gramian_weighting.alpha = value

    def set_losses(self, losses: Tensor) -> None:
        """Stores the current task losses. See :meth:`GradNormWeighting.set_losses`."""
        self.gramian_weighting.set_losses(losses)

    def balancing_loss(self) -> Tensor:
        """Computes the auxiliary loss. See :meth:`GradNormWeighting.balancing_loss`."""
        return self.gramian_weighting.balancing_loss()

    def renormalize(self) -> None:
        """Rescales the task weights. See :meth:`GradNormWeighting.renormalize`."""
        self.gramian_weighting.renormalize()

    def reset(self) -> None:
        """Resets the task weights and loss baseline. See :meth:`GradNormWeighting.reset`."""
        self.gramian_weighting.reset()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_tasks={self.n_tasks}, alpha={self.alpha!r})"
