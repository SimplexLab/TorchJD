import torch
from torch import Tensor


def _projection2simplex(y: Tensor) -> Tensor:
    """Euclidean projection of ``y`` onto the probability simplex."""

    m = len(y)
    sorted_y = torch.sort(y, descending=True)[0]
    tmpsum = y.new_zeros(())
    tmax_f = (torch.sum(y) - 1.0) / m
    for i in range(m - 1):
        tmpsum = tmpsum + sorted_y[i]
        tmax = (tmpsum - 1.0) / (i + 1.0)
        if tmax > sorted_y[i + 1]:
            tmax_f = tmax
            break
    return torch.max(y - tmax_f, y.new_zeros(m))
