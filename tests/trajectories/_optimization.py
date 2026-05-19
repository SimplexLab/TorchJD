import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from torchjd import autojac
from torchjd.aggregation import Aggregator
from trajectories._objectives import Objective


def optimize(
    objective: Objective,
    initial_x: Tensor,
    aggregator: Aggregator,
    lr: float,
    n_iters: int,
) -> tuple[list[Tensor], list[Tensor]]:
    xs = []
    ys = []
    x = initial_x.clone().requires_grad_()
    optimizer = torch.optim.SGD([x], lr=lr)
    for i in range(n_iters):
        print(f"{i + 1}/{n_iters}", end="\r")
        xs.append(x.detach().clone())
        y = objective(x)
        ys.append(y.detach().clone())
        optimizer.zero_grad()
        autojac.backward(y)
        autojac.jac_to_grad([x], aggregator)
        optimizer.step()
    print(" " * 50, end="\r")

    return xs, ys


def compute_gradient_cosine_similarities(
    objective: Objective, x0_min: float, x0_max: float, x1_min: float, x1_max: float, n: int
) -> Tensor:
    if objective.n_values != 2:
        raise ValueError("Objective should have 2 values.")

    x0_len = x0_max - x0_min
    x0_start = x0_min + x0_len / (n * 2)
    x0_end = x0_max - x0_len / (n * 2)

    x1_len = x1_max - x1_min
    x1_start = x1_min + x1_len / (n * 2)
    x1_end = x1_max - x1_len / (n * 2)

    x0s = np.linspace(x0_start, x0_end, n, dtype=np.float32)
    x1s = np.linspace(x1_start, x1_end, n, dtype=np.float32)

    similarities = torch.zeros(n, n)
    for i, x0 in enumerate(x0s):
        for j, x1 in enumerate(x1s):
            x = torch.tensor([x0, x1])
            similarities[i][j] = compute_cosine_similarity(objective, x)

    return similarities


def compute_cosine_similarity(objective: Objective, x: Tensor) -> Tensor:
    J = objective.jacobian(x)
    return F.cosine_similarity(J[0].unsqueeze(0), J[1].unsqueeze(0), eps=1e-19).squeeze()
