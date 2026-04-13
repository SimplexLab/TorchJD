from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from torchjd._linalg import PSDMatrix

from ._aggregator_bases import GramianWeightedAggregator
from ._utils.non_differentiable import raise_non_differentiable_error
from ._weighting_bases import Weighting


class GradVac(GramianWeightedAggregator):
    r"""
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` implementing the aggregation step of
    Gradient Vaccine (GradVac) from `Gradient Vaccine: Investigating and Improving Multi-task
    Optimization in Massively Multilingual Models (ICLR 2021 Spotlight)
    <https://openreview.net/forum?id=F1vEjWK-lH_>`_.

    For each task :math:`i`, the order in which other tasks :math:`j` are visited is drawn at
    random. For each pair :math:`(i, j)`, the cosine similarity :math:`\phi_{ij}` between the
    (possibly already modified) gradient of task :math:`i` and the original gradient of task
    :math:`j` is compared to an EMA target :math:`\hat{\phi}_{ij}`. When
    :math:`\phi_{ij} < \hat{\phi}_{ij}`, a closed-form correction adds a scaled copy of
    :math:`g_j` to :math:`g_i^{(\mathrm{PC})}`. The EMA is then updated with
    :math:`\hat{\phi}_{ij} \leftarrow (1-\beta)\hat{\phi}_{ij} + \beta \phi_{ij}`. The aggregated
    vector is the sum of the modified rows.

    This aggregator is stateful: it keeps :math:`\hat{\phi}` across calls. Use :meth:`reset` when
    the number of tasks or dtype changes.

    :param beta: EMA decay for :math:`\hat{\phi}` (paper default ``0.5``). You may read or assign
        the :attr:`beta` attribute between steps to tune the EMA update.
    :param eps: Small non-negative constant added to denominators when computing cosines and the
        vaccine weight (default ``1e-8``); set to ``0`` to omit this stabilization. You may read or
        assign the :attr:`eps` attribute between steps to tune numerical behavior.

    .. note::
        For each task :math:`i`, the order of other tasks :math:`j` is shuffled independently
        using the global PyTorch RNG (``torch.randperm``). Seed it with ``torch.manual_seed`` if
        you need reproducibility.

    .. note::
        To apply GradVac with per-layer or per-parameter-group granularity, create a separate
        :class:`GradVac` instance for each group and call
        :func:`~torchjd.autojac.jac_to_grad` once per group after
        :func:`~torchjd.autojac.mtl_backward`. Each instance maintains its own EMA state,
        matching the per-block targets :math:`\hat{\phi}_{ijk}` from the original paper. See
        the :doc:`Grouping </examples/grouping>` example for details.
    """

    def __init__(self, beta: float = 0.5, eps: float = 1e-8) -> None:
        weighting = GradVacWeighting(beta=beta, eps=eps)
        super().__init__(weighting)
        self._gradvac_weighting = weighting
        self.register_full_backward_pre_hook(raise_non_differentiable_error)

    @property
    def beta(self) -> float:
        """EMA decay coefficient for :math:`\\hat{\\phi}` (paper default ``0.5``)."""

        return self._gradvac_weighting.beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._gradvac_weighting.beta = value

    @property
    def eps(self) -> float:
        """Small non-negative constant added to denominators for numerical stability."""

        return self._gradvac_weighting.eps

    @eps.setter
    def eps(self, value: float) -> None:
        self._gradvac_weighting.eps = value

    def reset(self) -> None:
        """Clears EMA state so the next forward starts from zero targets."""

        self._gradvac_weighting.reset()

    def __repr__(self) -> str:
        return f"GradVac(beta={self.beta!r}, eps={self.eps!r})"


class GradVacWeighting(Weighting[PSDMatrix]):
    r"""
    :class:`~torchjd.aggregation._weighting_bases.Weighting` giving the weights of
    :class:`~torchjd.aggregation.GradVac`.

    All required quantities (gradient norms, cosine similarities, and their updates after the
    vaccine correction) are derived purely from the Gramian, without needing the full Jacobian.
    If :math:`g_i^{(\mathrm{PC})} = \sum_k c_{ik} g_k`, then:

    .. math::

        \|g_i^{(\mathrm{PC})}\|^2 = \mathbf{c}_i G \mathbf{c}_i^\top,\qquad
        g_i^{(\mathrm{PC})} \cdot g_j = \mathbf{c}_i G_{:,j}

    where :math:`G` is the Gramian.  The correction :math:`g_i^{(\mathrm{PC})} \mathrel{+}= w
    g_j` then becomes :math:`c_{ij} \mathrel{+}= w`, and the updated dot products follow
    immediately.

    This weighting is stateful: it keeps :math:`\hat{\phi}` across calls. Use :meth:`reset` when
    the number of tasks or dtype changes.

    :param beta: EMA decay for :math:`\hat{\phi}` (paper default ``0.5``).
    :param eps: Small non-negative constant added to denominators (default ``1e-8``).
    """

    def __init__(self, beta: float = 0.5, eps: float = 1e-8) -> None:
        super().__init__()
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"Parameter `beta` must be in [0, 1]. Found beta={beta!r}.")
        if eps < 0.0:
            raise ValueError(f"Parameter `eps` must be non-negative. Found eps={eps!r}.")

        self._beta = beta
        self._eps = eps
        self._phi_t: Tensor | None = None
        self._state_key: tuple[int, torch.dtype] | None = None

    @property
    def beta(self) -> float:
        """EMA decay coefficient for :math:`\\hat{\\phi}` (paper default ``0.5``)."""

        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Attribute `beta` must be in [0, 1]. Found beta={value!r}.")
        self._beta = value

    @property
    def eps(self) -> float:
        """Small non-negative constant added to denominators for numerical stability."""

        return self._eps

    @eps.setter
    def eps(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"Attribute `eps` must be non-negative. Found eps={value!r}.")
        self._eps = value

    def reset(self) -> None:
        """Clears EMA state so the next forward starts from zero targets."""

        self._phi_t = None
        self._state_key = None

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        # Move all computations on cpu to avoid moving memory between cpu and gpu at each iteration
        device = gramian.device
        dtype = gramian.dtype
        cpu = torch.device("cpu")

        G = cast(PSDMatrix, gramian.to(device=cpu))
        m = G.shape[0]

        self._ensure_state(m, dtype)
        phi_t = cast(Tensor, self._phi_t)

        beta = self._beta
        eps = self._eps

        # C[i, :] holds coefficients such that g_i^PC = sum_k C[i,k] * g_k (original gradients).
        # Initially each modified gradient equals the original, so C = I.
        C = torch.eye(m, device=cpu, dtype=dtype)

        for i in range(m):
            # Dot products of g_i^PC with every original g_j, shape (m,).
            cG = C[i] @ G

            others = [j for j in range(m) if j != i]
            perm = torch.randperm(len(others))
            shuffled_js = [others[idx] for idx in perm.tolist()]

            for j in shuffled_js:
                dot_ij = cG[j]
                norm_i_sq = (cG * C[i]).sum()
                norm_i = norm_i_sq.clamp(min=0.0).sqrt()
                norm_j = G[j, j].clamp(min=0.0).sqrt()
                denom = norm_i * norm_j + eps
                phi_ijk = dot_ij / denom

                phi_hat = phi_t[i, j]
                if phi_ijk < phi_hat:
                    sqrt_1_phi2 = (1.0 - phi_ijk * phi_ijk).clamp(min=0.0).sqrt()
                    sqrt_1_hat2 = (1.0 - phi_hat * phi_hat).clamp(min=0.0).sqrt()
                    denom_w = norm_j * sqrt_1_hat2 + eps
                    w = norm_i * (phi_hat * sqrt_1_phi2 - phi_ijk * sqrt_1_hat2) / denom_w
                    C[i, j] = C[i, j] + w
                    cG = cG + w * G[j]

                phi_t[i, j] = (1.0 - beta) * phi_hat + beta * phi_ijk

        weights = C.sum(dim=0)
        return weights.to(device)

    def _ensure_state(self, m: int, dtype: torch.dtype) -> None:
        key = (m, dtype)
        if self._state_key != key or self._phi_t is None:
            self._phi_t = torch.zeros(m, m, dtype=dtype)
            self._state_key = key
