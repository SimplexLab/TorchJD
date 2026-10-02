from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from torchjd._mixins import Stateful
from torchjd.linalg import PSDMatrix

from ._aggregator_bases import GramianWeightedAggregator
from ._mixins import _NonDifferentiable
from ._weighting_bases import _GramianWeighting


# Non-differentiable: the weights are obtained from an active-set solver, which branches on the
# values of the Gramian.
class PCDWeighting(_GramianWeighting, Stateful, _NonDifferentiable):
    r"""
    :class:`~torchjd.Stateful`
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.PSDMatrix`]
    giving the weights of :class:`~torchjd.aggregation.PCD`.

    The first row and column of the Gramian correspond to the primary objective.

    :param tau: The fraction :math:`\tau \in [0, 1]` of normalized first-order progress guaranteed
        to each secondary objective. Either a float, shared by all secondary objectives, or a vector
        with one value per secondary objective (i.e. of length :math:`m - 1`).
    :param beta: The decay :math:`\beta \in [0, 1)` of the exponential moving average of the
        squared gradient norms.
    :param eps: The non-negative constant :math:`\epsilon` added to the moving average before
        taking its inverse square root.

    .. note::
        The quadratic program is solved exactly with the dual active-set method of `Goldfarb and
        Idnani (1983) <https://doi.org/10.1007/BF02591962>`_, expressed in terms of the Gramian
        only. The reference implementation instead enumerates the working sets of constraints
        (Appendix B.2 of the paper). Both methods find the same minimizer, but the enumeration is
        exponential in :math:`m`.
    """

    def __init__(self, tau: float | Tensor = 0.02, beta: float = 0.999, eps: float = 1e-8) -> None:
        super().__init__()
        self.tau = tau
        self.beta = beta
        self.eps = eps
        self.register_buffer("_sq_norm_ema", None)
        self.register_buffer("_n_steps", None)
        self._state_key: int | None = None

    @property
    def tau(self) -> float | Tensor:
        return self._tau

    @tau.setter
    def tau(self, value: float | Tensor) -> None:
        if isinstance(value, Tensor):
            if value.ndim != 1:
                raise ValueError(
                    f"Attribute `tau` must be a float or a vector (1D Tensor). Found `tau.ndim = "
                    f"{value.ndim}`.",
                )
            is_valid = bool(((value >= 0.0) & (value <= 1.0)).all())
        else:
            is_valid = 0.0 <= value <= 1.0
        if not is_valid:
            raise ValueError(f"Attribute `tau` must be in [0, 1]. Found tau={value!r}.")
        self._tau = value

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        if not (0.0 <= value < 1.0):
            raise ValueError(f"Attribute `beta` must be in [0, 1). Found beta={value!r}.")
        self._beta = value

    @property
    def eps(self) -> float:
        return self._eps

    @eps.setter
    def eps(self, value: float) -> None:
        if not (value >= 0.0):
            raise ValueError(f"Attribute `eps` must be non-negative. Found eps={value!r}.")
        self._eps = value

    def reset(self) -> None:
        """Clears the moving average of the squared gradient norms."""

        self._sq_norm_ema = None
        self._n_steps = None
        self._state_key = None

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        # The problem only has size m x m, so we solve it on cpu and in float64.
        G = gramian.to(device="cpu", dtype=torch.float64)
        m = G.shape[0]
        weights = torch.zeros(m, dtype=torch.float64)
        if m == 0:
            return weights.to(device=gramian.device, dtype=gramian.dtype)

        taus = self._get_taus(m)
        if not G.isfinite().all():
            # Let nan and inf propagate to the output, without corrupting the moving average.
            return torch.full_like(gramian.diagonal(), torch.nan)

        sq_norms = G.diagonal()
        scales = self._update_scales(sq_norms)

        # Relative precision of the Gramian. Below it, the solver treats gradients as linearly
        # dependent, and the direction as zero.
        precision = 10.0 * torch.finfo(gramian.dtype).eps

        # If the gradient of the primary objective is zero, the update is zero.
        if sq_norms[0] > 0.0:
            normalized_G = G * torch.outer(scales, scales)
            w = _solve_qp(normalized_G, taus, dependence_tol=max(1e-9, precision))
            # Squared norm of d~ = sum_i w_i s_i g_i, and the scale of its rounding error when it is
            # computed from the Gramian.
            direction_sq_norm = w @ normalized_G @ w
            magnitude = (w.abs() @ normalized_G.diagonal().sqrt()) ** 2
            # Rescale the direction to the norm of the primary gradient, unless it is zero up to
            # rounding errors.
            if direction_sq_norm > precision * magnitude:
                weights = w * scales * (sq_norms[0] / direction_sq_norm).sqrt()

        return weights.to(device=gramian.device, dtype=gramian.dtype)

    def _get_taus(self, m: int) -> Tensor:
        if isinstance(self.tau, Tensor):
            if self.tau.shape[0] != m - 1:
                raise ValueError(
                    "When `tau` is a vector, it must have one value per secondary objective. Found "
                    f"`tau.shape[0] = {self.tau.shape[0]}` for {m - 1} secondary objectives.",
                )
            return self.tau.to(device="cpu", dtype=torch.float64)
        return torch.full([m - 1], self.tau, dtype=torch.float64)

    def _update_scales(self, sq_norms: Tensor) -> Tensor:
        """
        Updates the moving average of the squared gradient norms with the current ones, and returns
        the scale by which each gradient should be multiplied to be normalized.
        """

        self._ensure_state(sq_norms.shape[0])
        sq_norm_ema = cast(Tensor, self._sq_norm_ema).to(sq_norms)
        n_steps = int(cast(Tensor, self._n_steps)) + 1
        sq_norm_ema = self.beta * sq_norm_ema + (1.0 - self.beta) * sq_norms
        self._sq_norm_ema = sq_norm_ema
        self._n_steps = torch.tensor(n_steps)

        debiased_sq_norm_ema = sq_norm_ema / (1.0 - self.beta**n_steps)
        denominators = (debiased_sq_norm_ema + self.eps).sqrt()
        # A zero moving average means that the gradient has always been zero, so its scale does not
        # matter. Using 0 avoids dividing by zero when eps = 0.
        return torch.where(denominators > 0.0, 1.0 / denominators, 0.0)

    def _ensure_state(self, m: int) -> None:
        # The moving average is always kept on cpu and in float64, where the weights are computed,
        # so it only depends on the number of objectives.
        if self._state_key != m or self._sq_norm_ema is None:
            self._sq_norm_ema = torch.zeros(m, dtype=torch.float64)
            self._n_steps = torch.tensor(0)
            self._state_key = m

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tau={self.tau!r}, beta={self.beta!r}, eps={self.eps!r})"


class PCD(GramianWeightedAggregator, Stateful, _NonDifferentiable):
    r"""
    :class:`~torchjd.Stateful`
    :class:`~torchjd.aggregation.GramianWeightedAggregator` implementing Priority-Constrained
    Descent (PCD), from `Not All Objectives Are Born Equal: Priority-Constrained Descent for
    Hierarchical Multi-Objective Optimization <https://openreview.net/forum?id=HT01yGHLEt>`_ (TMLR
    2026, `arXiv:2606.29521 <https://arxiv.org/abs/2606.29521>`_).

    The first row :math:`g_1` of the input matrix is the gradient of the primary objective, and the
    other rows :math:`g_2, \dots, g_m` are the gradients of the secondary objectives. PCD follows the
    primary gradient as closely as possible, subject to each secondary objective receiving at least
    a fraction :math:`\tau` of the first-order progress that a step along its own normalized
    gradient would give:

    .. math::

        \tilde d = \mathop{\mathrm{arg\,min}}_{d} \ \frac{1}{2} \left\| d - \tilde g_1 \right\|^2
        \quad \text{subject to} \quad \tilde g_j^\top d \geq \tau \left\| \tilde g_j \right\|^2
        \quad \text{for } j = 2, \dots, m,

    where:

    - :math:`\tilde g_i = s_i g_i` is the normalized gradient of objective :math:`i`,
    - :math:`s_i = 1 / \sqrt{\hat v_i + \epsilon}` is its scale,
    - :math:`\hat v_i` is the bias-corrected exponential moving average of :math:`\|g_i\|^2` over
      the successive calls, with decay :math:`\beta`.

    The output is :math:`\tilde d` rescaled to the norm of :math:`g_1`. Because of the
    normalization, :math:`\tau` is a scale-free fraction: multiplying a secondary row of the input
    by the same positive constant at every call leaves the output unchanged, up to the effect of
    :math:`\epsilon`.

    Special cases:

    - If :math:`g_1 = 0`, the output is zero.
    - If the constraints cannot be satisfied simultaneously (which requires :math:`m \geq 3`, e.g.
      two anti-parallel secondary gradients), they are dropped and the output is :math:`g_1`.
    - If :math:`\tilde d = 0`, up to the rounding errors of the Gramian, the output is zero.
    - If the input contains ``nan`` or ``inf``, the output is ``nan`` and the moving average is left
      unchanged.

    :param tau: The fraction :math:`\tau \in [0, 1]` of normalized first-order progress guaranteed
        to each secondary objective. Either a float, shared by all secondary objectives, or a vector
        with one value per secondary objective (i.e. of length :math:`m - 1`).
    :param beta: The decay :math:`\beta \in [0, 1)` of the exponential moving average of the
        squared gradient norms.
    :param eps: The non-negative constant :math:`\epsilon` added to the moving average before
        taking its inverse square root.

    .. note::
        PCD is not symmetric in the objectives: the first row of the input matrix (e.g. the
        first loss given to :func:`~torchjd.autojac.backward`) is the primary objective.

    .. note::
        This aggregator is stateful: it keeps the moving average of the squared gradient norms
        across calls. Use :meth:`reset` to clear it. It is also cleared automatically when the
        number of rows changes.

    .. note::
        The reference implementation is available at
        `github.com/DaraVaram/priority-constrained-descent
        <https://github.com/DaraVaram/priority-constrained-descent>`_.
    """

    gramian_weighting: PCDWeighting

    def __init__(self, tau: float | Tensor = 0.02, beta: float = 0.999, eps: float = 1e-8) -> None:
        super().__init__(PCDWeighting(tau=tau, beta=beta, eps=eps))

    @property
    def tau(self) -> float | Tensor:
        return self.gramian_weighting.tau

    @tau.setter
    def tau(self, value: float | Tensor) -> None:
        self.gramian_weighting.tau = value

    @property
    def beta(self) -> float:
        return self.gramian_weighting.beta

    @beta.setter
    def beta(self, value: float) -> None:
        self.gramian_weighting.beta = value

    @property
    def eps(self) -> float:
        return self.gramian_weighting.eps

    @eps.setter
    def eps(self, value: float) -> None:
        self.gramian_weighting.eps = value

    def reset(self) -> None:
        """Clears the moving average of the squared gradient norms."""

        self.gramian_weighting.reset()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tau={self.tau!r}, beta={self.beta!r}, eps={self.eps!r})"


def _solve_qp(
    G: Tensor,
    taus: Tensor,
    tol: float = 1e-9,
    dependence_tol: float = 1e-9,
    max_iters: int | None = None,
) -> Tensor:
    r"""
    Solves the quadratic program of PCD, given the Gramian :math:`G` of the (normalized) gradients
    :math:`g_1, \dots, g_m`, whose first row and column correspond to the primary objective:

    .. math::

        \min_d \frac{1}{2} \|d - g_1\|^2 \quad \text{s.t.} \quad
        g_j^\top d \geq \tau_j \|g_j\|^2 \quad \text{for } j = 2, \dots, m

    Returns the weights :math:`w = [1, \mu_2, \dots, \mu_m]` such that the solution is
    :math:`\sum_i w_i g_i`, where :math:`\mu_j \geq 0` are the KKT multipliers of the constraints.
    If the constraints are infeasible, returns :math:`[1, 0, \dots, 0]`.

    This is the dual active-set method of Goldfarb and Idnani (1983), for an identity Hessian. The
    stationarity condition :math:`d = g_1 + \sum_j \mu_j g_j` holds at every iteration, so that all
    the required inner products can be read from :math:`G`.

    A constraint counts as satisfied when it is violated by at most ``tol`` times the largest
    diagonal entry of :math:`G`. A gradient counts as a linear combination of the gradients of the
    active constraints when its component orthogonal to them has a squared norm of at most
    ``dependence_tol`` times its own. The latter should be above the relative precision of
    :math:`G`, or rounding errors can make dependent gradients look independent, with huge
    multipliers.

    Each iteration adds or drops one constraint. In exact arithmetic, the method terminates after
    finitely many iterations (about :math:`m` in practice). ``max_iters`` (by default :math:`10 m`)
    only guards against cycling caused by rounding errors: when it is reached, the current iterate
    is returned.
    """

    m = G.shape[0]
    weights = torch.zeros(m, dtype=G.dtype)
    weights[0] = 1.0
    if m == 1:
        return weights

    if max_iters is None:
        max_iters = 10 * m
    Gs = G[1:, 1:]
    b = taus * Gs.diagonal() - G[1:, 0]  # b_j > 0 iff g_1 violates the constraint of objective j
    mu = torch.zeros(m - 1, dtype=G.dtype)
    atol = tol * max(G.diagonal().max().item(), torch.finfo(G.dtype).tiny)
    active: list[int] = []
    n_iters = 0

    while n_iters < max_iters:
        slacks = Gs @ mu - b
        slacks[active] = 0.0  # The active constraints hold with equality, up to rounding errors.
        p = int(slacks.argmin())
        if slacks[p] >= -atol:
            break

        # Add the violated constraint p to the active set, possibly dropping some others first.
        slack_p = slacks[p]
        while n_iters < max_iters:
            n_iters += 1

            # In the dual space, the step direction is +1 for mu_p and -r for the active mu_j.
            # In the primal space, it is z = g_p - sum_j r_j g_j, the component of g_p orthogonal to
            # the gradients of the active constraints.
            if len(active) > 0:
                r = torch.linalg.solve(Gs[active][:, active], Gs[active, p])
                z_sq_norm = Gs[p, p] - Gs[p, active] @ r
                ratios = torch.where(r > 0.0, mu[active] / r, torch.inf)
                k = int(ratios.argmin())
                partial_step = ratios[k].item()
            else:
                r = torch.zeros(0, dtype=G.dtype)
                z_sq_norm = Gs[p, p]
                k = -1
                partial_step = torch.inf

            # If z = 0, g_p is a linear combination of the active gradients and no primal step is
            # possible.
            is_z_non_zero = z_sq_norm > dependence_tol * Gs[p, p]
            full_step = (-slack_p / z_sq_norm).item() if is_z_non_zero else torch.inf

            step = min(partial_step, full_step)
            if step == torch.inf:
                return weights  # The constraints are infeasible.

            mu[active] -= step * r
            mu[p] += step
            if full_step <= partial_step:
                active.append(p)
                break

            # Drop the constraint whose multiplier reached zero, and try adding p again.
            mu[active[k]] = 0.0
            del active[k]
            slack_p = Gs[p] @ mu - b[p]

    weights[1:] = mu.clamp(min=0.0)
    return weights
