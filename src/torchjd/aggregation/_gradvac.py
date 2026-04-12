from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from torchjd._linalg import Matrix

from ._aggregator_bases import Aggregator
from ._utils.non_differentiable import raise_non_differentiable_error


def _all_layer_group_sizes(encoder: nn.Module) -> tuple[int, ...]:
    """
    Block sizes per leaf submodule with parameters, matching the ``all_layer`` grouping: iterate
    ``encoder.modules()`` and append the total number of elements in each module that has no child
    submodules and registers at least one parameter.
    """

    return tuple(
        sum(w.numel() for w in module.parameters())
        for module in encoder.modules()
        if len(list(module.children())) == 0 and next(module.parameters(), None) is not None
    )


def _all_matrix_group_sizes(shared_params: Iterable[Tensor]) -> tuple[int, ...]:
    """One block per tensor in ``shared_params`` order (``all_matrix`` / shared-parameter layout)."""

    return tuple(p.numel() for p in shared_params)


class GradVac(Aggregator):
    r"""
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` implementing Gradient Vaccine
    (GradVac) from `Gradient Vaccine: Investigating and Improving Multi-task Optimization in
    Massively Multilingual Models (ICLR 2021 Spotlight)
    <https://openreview.net/forum?id=F1vEjWK-lH_>`_.

    The input matrix is a Jacobian :math:`J \in \mathbb{R}^{m \times n}` whose rows are per-task
    gradients. For each task :math:`i` and each parameter block :math:`k`, the order in which other
    tasks :math:`j` are visited is drawn at random (independently for each :math:`k`); for each pair
    :math:`(i, j)` on block :math:`k`, the cosine correlation :math:`\phi_{ijk}` between the
    (possibly already modified) gradient of task :math:`i` and the original gradient of task
    :math:`j` on that block is compared to an EMA target :math:`\hat{\phi}_{ijk}`. When
    :math:`\phi_{ijk} < \hat{\phi}_{ijk}`, a closed-form correction adds a scaled copy of
    :math:`g_j` to the block of :math:`g_i^{(\mathrm{PC})}`. The EMA is then updated with
    :math:`\hat{\phi}_{ijk} \leftarrow (1-\beta)\hat{\phi}_{ijk} + \beta \phi_{ijk}`. The aggregated
    vector is the sum of the modified rows.

    This aggregator is stateful: it keeps :math:`\hat{\phi}` across calls. Use :meth:`reset` when
    the number of tasks, parameter dimension, grouping, device, or dtype changes.

    **Parameter granularity** is selected by ``group_type`` (default ``"whole_model"``). It defines
    how each task gradient row is partitioned into blocks :math:`k` so that cosines and EMA targets
    :math:`\hat{\phi}_{ijk}` are computed **per block** rather than only globally:

    * ``"whole_model"``: the full row of length :math:`n` is a single block. Cosine similarity is
      taken between entire task gradients. Do not pass ``encoder`` or ``shared_params``.
    * ``"all_layer"``: one block per leaf ``nn.Module`` under ``encoder`` that holds parameters
      (same rule as iterating ``encoder.modules()`` and selecting leaves with parameters). Pass
      ``encoder``; ``shared_params`` must be omitted.
    * ``"all_matrix"``: one block per tensor in ``shared_params``, in iteration order. That order
      must match how Jacobian columns are laid out for those shared parameters. Pass
      ``shared_params``; ``encoder`` must be omitted.

    :param beta: EMA decay for :math:`\hat{\phi}` (paper default ``0.5``). You may read or assign the
        :attr:`beta` attribute between steps to tune the EMA update.
    :param group_type: Granularity of parameter grouping; see **Parameter granularity** above.
    :param encoder: Module whose subtree defines ``all_layer`` blocks when
        ``group_type == "all_layer"``.
    :param shared_params: Iterable of parameter tensors defining ``all_matrix`` block sizes and
        order when ``group_type == "all_matrix"``. It is materialized once at construction.
    :param eps: Small non-negative constant added to denominators when computing cosines and the
        vaccine weight (default ``1e-8``); set to ``0`` to omit this stabilization. You may read or
        assign the :attr:`eps` attribute between steps to tune numerical behavior.

    .. note::
        GradVac is not compatible with autogram: it needs full Jacobian rows and per-block inner
        products, not only a Gram matrix. Only the autojac path is supported.

    .. note::
        For each task :math:`i` and block :math:`k`, the order of other tasks :math:`j` is shuffled
        independently using the global PyTorch RNG (``torch.randperm``). Seed it with
        ``torch.manual_seed`` if you need reproducibility.
    """

    def __init__(
        self,
        beta: float = 0.5,
        group_type: Literal["whole_model", "all_layer", "all_matrix"] = "whole_model",
        encoder: nn.Module | None = None,
        shared_params: Iterable[Tensor] | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"Parameter `beta` must be in [0, 1]. Found beta={beta!r}.")
        params_tuple: tuple[Tensor, ...] = ()
        fixed_block_sizes: tuple[int, ...] | None
        if group_type == "whole_model":
            if encoder is not None:
                raise ValueError(
                    'Parameter `encoder` must be None when `group_type == "whole_model"`.'
                )
            if shared_params is not None:
                raise ValueError(
                    'Parameter `shared_params` must be None when `group_type == "whole_model"`.'
                )
            fixed_block_sizes = None
        elif group_type == "all_layer":
            if encoder is None:
                raise ValueError(
                    'Parameter `encoder` is required when `group_type == "all_layer"`.'
                )
            if shared_params is not None:
                raise ValueError(
                    'Parameter `shared_params` must be None when `group_type == "all_layer"`.'
                )
            fixed_block_sizes = _all_layer_group_sizes(encoder)
            if sum(fixed_block_sizes) == 0:
                raise ValueError("Parameter `encoder` has no parameters in any leaf module.")
        else:
            if shared_params is None:
                raise ValueError(
                    'Parameter `shared_params` is required when `group_type == "all_matrix"`.'
                )
            if encoder is not None:
                raise ValueError(
                    'Parameter `encoder` must be None when `group_type == "all_matrix"`.'
                )
            params_tuple = tuple(shared_params)
            if len(params_tuple) == 0:
                raise ValueError(
                    'Parameter `shared_params` must be non-empty when `group_type == "all_matrix"`.'
                )
            fixed_block_sizes = _all_matrix_group_sizes(params_tuple)

        if eps < 0.0:
            raise ValueError(f"Parameter `eps` must be non-negative. Found eps={eps!r}.")

        self._beta = beta
        self._group_type = group_type
        self._encoder = encoder
        self._shared_params_len = len(params_tuple)
        self._fixed_block_sizes = fixed_block_sizes
        self._eps = eps

        self._phi_t: Tensor | None = None
        self._state_key: tuple[int, int, tuple[int, ...], torch.device, torch.dtype] | None = None

        self.register_full_backward_pre_hook(raise_non_differentiable_error)

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

    def __repr__(self) -> str:
        enc = "None" if self._encoder is None else f"{self._encoder.__class__.__name__}(...)"
        sp = "None" if self._group_type != "all_matrix" else f"n_params={self._shared_params_len}"
        return (
            f"{self.__class__.__name__}(beta={self._beta!r}, group_type={self._group_type!r}, "
            f"encoder={enc}, shared_params={sp}, eps={self._eps!r})"
        )

    def _resolve_segment_sizes(self, n: int) -> tuple[int, ...]:
        if self._group_type == "whole_model":
            return (n,)
        sizes = cast(tuple[int, ...], self._fixed_block_sizes)
        if sum(sizes) != n:
            raise ValueError(
                "The Jacobian width `n` must equal the sum of block sizes implied by "
                f"`encoder` or `shared_params` for this `group_type`. Found n={n}, "
                f"sum(block_sizes)={sum(sizes)}.",
            )
        return sizes

    def _ensure_state(
        self,
        m: int,
        n: int,
        sizes: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        key = (m, n, sizes, device, dtype)
        num_groups = len(sizes)
        if self._state_key != key or self._phi_t is None:
            self._phi_t = torch.zeros(m, m, num_groups, device=device, dtype=dtype)
            self._state_key = key

    def forward(self, matrix: Matrix, /) -> Tensor:
        grads = matrix
        m, n = grads.shape
        if m == 0 or n == 0:
            return torch.zeros(n, dtype=grads.dtype, device=grads.device)

        sizes = self._resolve_segment_sizes(n)
        device = grads.device
        dtype = grads.dtype
        self._ensure_state(m, n, sizes, device, dtype)
        phi_t = cast(Tensor, self._phi_t)
        beta = self.beta
        eps = self.eps

        pc_grads = grads.clone()
        offsets = [0]
        for s in sizes:
            offsets.append(offsets[-1] + s)

        for i in range(m):
            others = [j for j in range(m) if j != i]
            for k in range(len(sizes)):
                perm = torch.randperm(len(others))
                shuffled_js = [others[idx] for idx in perm.tolist()]
                beg, end = offsets[k], offsets[k + 1]
                for j in shuffled_js:
                    slice_i = pc_grads[i, beg:end]
                    slice_j = grads[j, beg:end]

                    norm_i = slice_i.norm()
                    norm_j = slice_j.norm()
                    denom = norm_i * norm_j + eps
                    phi_ijk = slice_i.dot(slice_j) / denom

                    phi_hat = phi_t[i, j, k]
                    if phi_ijk < phi_hat:
                        sqrt_1_phi2 = (1.0 - phi_ijk * phi_ijk).clamp(min=0.0).sqrt()
                        sqrt_1_hat2 = (1.0 - phi_hat * phi_hat).clamp(min=0.0).sqrt()
                        denom_w = norm_j * sqrt_1_hat2 + eps
                        w = norm_i * (phi_hat * sqrt_1_phi2 - phi_ijk * sqrt_1_hat2) / denom_w
                        pc_grads[i, beg:end] = slice_i + slice_j * w

                    phi_t[i, j, k] = (1.0 - beta) * phi_hat + beta * phi_ijk

        return pc_grads.sum(dim=0)
