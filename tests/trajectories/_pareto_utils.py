from collections.abc import Callable
from functools import partial

import numpy as np
import torch
from torch import Tensor, vmap

from trajectories._objectives import ElementWiseQuadratic, Objective, WithSPSMappingMixin


def compute_normalized_2d_pf_distances(
    objective: Objective, y0_min: float, y0_max: float, y1_min: float, y1_max: float, n: int
) -> Tensor:
    y0_len = y0_max - y0_min
    y0_start = y0_min + y0_len / (n * 2)
    y0_end = y0_max - y0_len / (n * 2)

    y1_len = y1_max - y1_min
    y1_start = y1_min + y1_len / (n * 2)
    y1_end = y1_max - y1_len / (n * 2)

    y0s = torch.linspace(y0_start, y0_end, n, dtype=torch.float32)
    y1s = torch.linspace(y1_start, y1_end, n, dtype=torch.float32)

    Y0, Y1 = torch.meshgrid(y0s, y1s, indexing="ij")  # shape: (n, n)
    Y = torch.stack([Y0, Y1], dim=-1)

    pf_dist = make_2d_pf_distance_fn(objective)
    distances = vmap(vmap(pf_dist))(Y)

    max_distance = torch.max(distances[distances.isfinite()])
    distances = distances / max_distance
    distances[distances.isnan()] = -1.0

    return distances


def sample_2d_spss(objective: Objective) -> Tensor:
    assert objective.n_values == 2 and isinstance(objective, WithSPSMappingMixin)

    eps = 1e-5

    n_samples = 1 if isinstance(objective, ElementWiseQuadratic) else 1000

    sps_mapping = objective.sps_mapping

    ws_np = np.linspace([0 + eps, 1 - eps], [1 - eps, 0 + eps], n_samples)
    ws = torch.tensor(ws_np)
    sps_points = torch.stack([sps_mapping(w) for w in ws])
    return sps_points


def sample_2d_pf(objective: Objective) -> Tensor:
    sps_points = sample_2d_spss(objective)
    pf_points = torch.stack([objective(x) for x in sps_points])
    return pf_points


def make_2d_pf_distance_fn(objective: Objective) -> Callable[[Tensor], Tensor]:
    pf_points = sample_2d_pf(objective)

    def compute_2d_pf_distance(pf_points: Tensor, y: Tensor) -> Tensor:
        """Compute the distance from a point y to a piecewise-linear Pareto front.

        The Pareto front is approximated as a polyline: the ordered sequence of
        ``pf_points`` defines consecutive line segments, and the distance returned
        is the minimum Euclidean distance from ``y`` to any of those segments.

        For each segment [A, B] the closest point on the segment to ``y`` is found
        via orthogonal projection:

            t = dot(y - A, B - A) / ||B - A||²

        ``t`` is clamped to [0, 1] so that the closest point is constrained to the
        segment rather than the infinite line through A and B. This ensures correct
        distances when ``y`` lies "outside" the extent of the front (i.e. beyond
        either endpoint).

        :param pf_points: Pareto front points of shape ``(k, n)``, ordered along the
            front. Adjacent points define the segments of the polyline.
        :param y: Query point of shape ``(n,)`` whose distance to the front is sought.
        """
        if len(pf_points) == 1:
            return (y - pf_points[0]).norm()

        pf_first = pf_points[:-1, :]
        pf_second = pf_points[1:, :]
        d = pf_second - pf_first
        t = ((y - pf_first) * d).sum(dim=1) / (d * d).sum(dim=1)
        closest = pf_first + t.clamp(0, 1).unsqueeze(1) * d

        # Clamp at 0 so that points below the PF have a distance of 0 to it.
        distances = torch.clamp(y - closest, min=0).norm(dim=1)
        return torch.min(distances)

    return partial(compute_2d_pf_distance, pf_points)
