# ![image](docs/source/icons/favicon-32x32.png) TorchJD

[![Doc](https://img.shields.io/badge/Doc-torchjd.org-blue?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8%2BCjwhLS0gQ3JlYXRlZCB1c2luZyBLcml0YTogaHR0cDovL2tyaXRhLm9yZyAtLT4KCjxzdmcKICAgd2lkdGg9IjIwNDcuNzJwdCIKICAgaGVpZ2h0PSIyMDQ3LjcycHQiCiAgIHZpZXdCb3g9IjAgMCAyMDQ3LjcyIDIwNDcuNzIiCiAgIHZlcnNpb249IjEuMSIKICAgaWQ9InN2ZzEiCiAgIHNvZGlwb2RpOmRvY25hbWU9IlRvcmNoSkRfbG9nb19jaXJjdWxhci5zdmciCiAgIGlua3NjYXBlOnZlcnNpb249IjEuMy4yICgwOTFlMjBlZjBmLCAyMDIzLTExLTI1KSIKICAgeG1sbnM6aW5rc2NhcGU9Imh0dHA6Ly93d3cuaW5rc2NhcGUub3JnL25hbWVzcGFjZXMvaW5rc2NhcGUiCiAgIHhtbG5zOnNvZGlwb2RpPSJodHRwOi8vc29kaXBvZGkuc291cmNlZm9yZ2UubmV0L0RURC9zb2RpcG9kaS0wLmR0ZCIKICAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogICB4bWxuczpzdmc9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8c29kaXBvZGk6bmFtZWR2aWV3CiAgICAgaWQ9Im5hbWVkdmlldzEiCiAgICAgcGFnZWNvbG9yPSIjZmZmZmZmIgogICAgIGJvcmRlcmNvbG9yPSIjNjY2NjY2IgogICAgIGJvcmRlcm9wYWNpdHk9IjEuMCIKICAgICBpbmtzY2FwZTpzaG93cGFnZXNoYWRvdz0iMiIKICAgICBpbmtzY2FwZTpwYWdlb3BhY2l0eT0iMC4wIgogICAgIGlua3NjYXBlOnBhZ2VjaGVja2VyYm9hcmQ9IjAiCiAgICAgaW5rc2NhcGU6ZGVza2NvbG9yPSIjZDFkMWQxIgogICAgIGlua3NjYXBlOmRvY3VtZW50LXVuaXRzPSJwdCIKICAgICBpbmtzY2FwZTp6b29tPSIwLjE2Mjk4NjE1IgogICAgIGlua3NjYXBlOmN4PSIxMzk1LjgyNDEiCiAgICAgaW5rc2NhcGU6Y3k9Ijg3NC4zMDczOSIKICAgICBpbmtzY2FwZTp3aW5kb3ctd2lkdGg9IjI1NjAiCiAgICAgaW5rc2NhcGU6d2luZG93LWhlaWdodD0iMTM3MSIKICAgICBpbmtzY2FwZTp3aW5kb3cteD0iMCIKICAgICBpbmtzY2FwZTp3aW5kb3cteT0iMCIKICAgICBpbmtzY2FwZTp3aW5kb3ctbWF4aW1pemVkPSIxIgogICAgIGlua3NjYXBlOmN1cnJlbnQtbGF5ZXI9InN2ZzEiIC8%2BCiAgPGRlZnMKICAgICBpZD0iZGVmczEiIC8%2BCiAgPHBhdGgKICAgICBpZD0ic2hhcGUxIgogICAgIGZpbGw9IiMwMDAwMDAiCiAgICAgZmlsbC1ydWxlPSJldmVub2RkIgogICAgIGQ9Ik0yNTUuMjE1IDg5OS44NzVMMjU1Ljk2NCAyNTUuOTY0TDc2Ny44OTMgMjU1Ljk2NEw3NjcuODkzIDBMMCAwTDAuMDMxMjUzMyA4OTguODQ0QzAuMDMxNzMwNSA4OTguODE0IDg0LjU3MjYgODk5Ljg3NSAyNTUuMjE1IDg5OS44NzVaIgogICAgIHN0eWxlPSJmaWxsOiMxYTgxZWI7ZmlsbC1vcGFjaXR5OjEiCiAgICAgdHJhbnNmb3JtPSJtYXRyaXgoMS4wMDAwMDAwMTQzMDcwNyAwIDAgMS4wMDAwMDAwMTQzMDcwNyAxMjcuOTgyMjI2NTIyMDU2IDEyNy45ODIyMjY1MjIwNTYpIiAvPgogIDxwYXRoCiAgICAgaWQ9InNoYXBlMDEiCiAgICAgdHJhbnNmb3JtPSJtYXRyaXgoLTEuMDAwMDAwMDA5MjIxODUgMCAwIC0xLjAwMDAwMDAwOTIyMTg1IDE5MTkuOTEzNjE3Mzk4NzEgMTkxMC4zMzcxOTY5MzEyNSkiCiAgICAgZmlsbD0iIzAwMDAwMCIKICAgICBmaWxsLXJ1bGU9ImV2ZW5vZGQiCiAgICAgZD0iTTc2OC4wNzQgMTc3Mi42MUMtMjgyLjAwNCAxNTk4LjY1IC0yMjkuNzEyIDE1MS44MjEgNzY4LjA3NCAwQzc2Ny4wODMgMjkuOTMzNyA3NjguMDk2IDE0Mi43NiA3NjguMDc0IDI2MC44ODZDNDEuNDc0NiA0NTYuOTAzIDEzNy40MjMgMTM4MC4wNiA3NjguMDc0IDE1MTMuNjQiCiAgICAgc3R5bGU9ImZpbGw6IzFhODFlYjtmaWxsLW9wYWNpdHk6MSIgLz4KICA8cGF0aAogICAgIGlkPSJzaGFwZTAyIgogICAgIGZpbGw9IiMwMDAwMDAiCiAgICAgZmlsbC1ydWxlPSJldmVub2RkIgogICAgIGQ9Ik03NjcuOTA5IDg4Ny4zMzhDMjYzLjQwMiA4MDMuOTI2IDAuMDc1OTQyMSAzODcuOTY0IDAgMC4wODU2NDk3QzE0LjY4NjggLTAuMDI4NTQ5OSA5OS4wNTUxIC0wLjAyODU0OTkgMjU1LjAxMSAwLjA4NTY0OTdDMjU1LjMxMSAyODEuMTE0IDQ0OC43ODYgNTYyLjE2MyA3NjcuOTA5IDYyNi40OTkiCiAgICAgc3R5bGU9ImZpbGw6IzFhODFlYjtmaWxsLW9wYWNpdHk6MSIKICAgICB0cmFuc2Zvcm09Im1hdHJpeCgwLjk5OTk5OTk2MDczODQ0IDAgMCAwLjk5OTk5OTk2MDczODQ0IDEyNy45NjY1OTE0OTQzMjggMTAyMy43NzIxNDc4MzE0KSIgLz4KICA8ZWxsaXBzZQogICAgIHN0eWxlPSJmaWxsOiMxYTgxZWI7c3Ryb2tlLXdpZHRoOjEuMDY3OTtmaWxsLW9wYWNpdHk6MSIKICAgICBpZD0icGF0aDEiCiAgICAgY3g9IjEwMjYuMzYxIgogICAgIGN5PSIxMDE0LjIyMTEiCiAgICAgcng9IjE4My4yNTU0MyIKICAgICByeT0iMTgzLjUxNTU4IiAvPgo8L3N2Zz4K)](https://torchjd.org)
[![Static Badge](https://img.shields.io/badge/%F0%9F%92%AC_ChatBot-chat.torchjd.org-blue?logo=%F0%9F%92%AC)](https://chat.torchjd.org)
[![Tests](https://github.com/SimplexLab/TorchJD/actions/workflows/checks.yml/badge.svg)](https://github.com/SimplexLab/TorchJD/actions/workflows/checks.yml)
[![codecov](https://codecov.io/gh/SimplexLab/TorchJD/graph/badge.svg?token=8AUCZE76QH)](https://codecov.io/gh/SimplexLab/TorchJD)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/SimplexLab/TorchJD/main.svg)](https://results.pre-commit.ci/latest/github/SimplexLab/TorchJD/main)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchjd)](https://pypi.org/project/torchjd/)
[![Static Badge](https://img.shields.io/badge/Discord%20-%20community%20-%20%235865F2?logo=discord&logoColor=%23FFFFFF&label=Discord)](https://discord.gg/76KkRnb3nk)

TorchJD is a library extending autograd to enable
[Jacobian descent](https://arxiv.org/pdf/2406.16232) with PyTorch. It can be used to train neural
networks with multiple objectives. In particular, it supports multi-task learning, with a wide
variety of aggregators from the literature. It also enables the instance-wise risk minimization
paradigm. The full documentation is available at [torchjd.org](https://torchjd.org), with several
usage examples.

## Jacobian descent (JD)
Jacobian descent is an extension of gradient descent supporting the optimization of vector-valued
functions. This algorithm can be used to train neural networks with multiple loss functions. In this
context, JD iteratively updates the parameters of the model using the Jacobian matrix of the vector
of losses (the matrix stacking each individual loss' gradient). For more details, please refer to
Section 2.1 of the [paper](https://arxiv.org/pdf/2406.16232).

### How does this compare to averaging the different losses and using gradient descent?

Averaging the losses and computing the gradient of the mean is mathematically equivalent to
computing the Jacobian and averaging its rows. However, this approach has limitations. If two
gradients are conflicting (they have a negative inner product), simply averaging them can result in
an update vector that is conflicting with one of the two gradients. Averaging the losses and making
a step of gradient descent can thus lead to an increase of one of the losses.

This is illustrated in the following picture, in which the two objectives' gradients $g_1$ and $g_2$
are conflicting, and averaging them gives an update direction that is detrimental to the first
objective. Note that in this picture, the dual cone, represented in green, is the set of vectors
that have a non-negative inner product with both $g_1$ and $g_2$.

![image](docs/source/_static/gradients_cone_projections_upgrad_mean.svg)

With Jacobian descent, $g_1$ and $g_2$ are computed individually and carefully aggregated using an
aggregator $\mathcal A$. In this example, the aggregator is the Unconflicting Projection of
Gradients $\mathcal A_{\text{UPGrad}}$: it
projects each gradient onto the dual cone, and averages the projections. This ensures that the
update will always be beneficial to each individual objective (given a sufficiently small step
size). In addition to $\mathcal A_{\text{UPGrad}}$, TorchJD supports
[more than 10 aggregators from the literature](https://torchjd.org/stable/docs/aggregation).

## Installation
<!-- start installation -->
TorchJD can be installed directly with pip:
```bash
pip install torchjd
```
<!-- end installation -->
Some aggregators may have additional dependencies. Please refer to the
[installation documentation](https://torchjd.org/stable/installation) for them.

## Usage

Compared to standard `torch`, `torchjd` simply changes the way to obtain the `.grad` fields of your
model parameters.

### Using the `autojac` engine

The autojac engine is for computing and aggregating Jacobians efficiently.

#### 1. `backward` + `jac_to_grad`
In standard `torch`, you generally combine your `losses` into a single scalar `loss`, and call
`loss.backward()` to compute the gradient of the loss with respect to each model parameter and to
store it in the `.grad` fields of those parameters. The basic usage of `torchjd` is to replace this
`loss.backward()` by a call to
[`torchjd.autojac.backward(losses)`](https://torchjd.org/stable/docs/autojac/backward/). Instead of
computing the gradient of a scalar loss, it will compute the Jacobian of a vector of losses, and
store it in the `.jac` fields of the model parameters. You then have to call
[`torchjd.autojac.jac_to_grad`](https://torchjd.org/stable/docs/autojac/jac_to_grad/) to aggregate
this Jacobian using the specified
[`Aggregator`](https://torchjd.org/stable/docs/aggregation#torchjd.aggregation.Aggregator), and to
store the result into the `.grad` fields of the model parameters.

#### 2. `mtl_backward` + `jac_to_grad`
In the case of multi-task learning, an alternative to
[`torchjd.autojac.backward`](https://torchjd.org/stable/docs/autojac/backward/) is
[`torchjd.autojac.mtl_backward`](https://torchjd.org/stable/docs/autojac/mtl_backward/). It computes
the gradient of each task-specific loss with respect to the corresponding task's parameters, and
stores it in their `.grad` fields. It also computes the Jacobian of the vector of losses with
respect to the shared parameters and stores it their `.jac` field. Then, the
[`torchjd.autojac.jac_to_grad`](https://torchjd.org/stable/docs/autojac/jac_to_grad/) function can
be called to aggregate this Jacobian and replace the `.jac` fields by `.grad` fields for the shared
parameters.

The following example shows how to use TorchJD to train a multi-task model with Jacobian descent,
using [UPGrad](https://torchjd.org/stable/docs/aggregation/upgrad/).

```diff
  import torch
  from torch.nn import Linear, MSELoss, ReLU, Sequential
  from torch.optim import SGD

+ from torchjd.autojac import jac_to_grad, mtl_backward
+ from torchjd.aggregation import UPGrad

  shared_module = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
  task1_module = Linear(3, 1)
  task2_module = Linear(3, 1)
  params = [
      *shared_module.parameters(),
      *task1_module.parameters(),
      *task2_module.parameters(),
  ]

  loss_fn = MSELoss()
  optimizer = SGD(params, lr=0.1)
+ aggregator = UPGrad()

  inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
  task1_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the first task
  task2_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the second task

  for input, target1, target2 in zip(inputs, task1_targets, task2_targets):
      features = shared_module(input)
      output1 = task1_module(features)
      output2 = task2_module(features)
      loss1 = loss_fn(output1, target1)
      loss2 = loss_fn(output2, target2)

-     loss = loss1 + loss2
-     loss.backward()
+     mtl_backward([loss1, loss2], features=features)
+     jac_to_grad(shared_module.parameters(), aggregator)
      optimizer.step()
      optimizer.zero_grad()
```

> [!NOTE]
> In this example, the Jacobian is only with respect to the shared parameters. The task-specific
> parameters are simply updated via the gradient of their task’s loss with respect to them.

> [!TIP]
> Once your model parameters all have a `.grad` field, it's the role of the
> [optimizer](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) to update the
> parameters values. This is exactly the same as in standard `torch`.

#### 3. `jac`

If you're simply interested in computing Jacobians without storing them in the `.jac` fields, you
can also use the [`torchjd.autojac.jac`](https://torchjd.org/stable/docs/autojac/jac/) function,
that is analog to
[`torch.autograd.grad`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html),
except that it computes the Jacobian of a vector of losses rather than the gradient of a scalar
loss.

### Using the `autogram` engine

The Gramian of the Jacobian, defined as the Jacobian multiplied by its transpose, contains all the
dot products between individual gradients. It thus contains all the information about conflict and
gradient imbalance. It turns out that most aggregators from the literature
(e.g. [UPGrad](https://torchjd.org/stable/docs/aggregation/upgrad/)) make a linear combinations of
the rows of the Jacobian, whose weights only depend only on the Gramian of the Jacobian.

An alternative implementation of Jacobian descent is thus to:
- Compute this Gramian incrementally (layer by layer), without ever storing the full Jacobian in
  memory.
- Extract the weights from it using a
  [`Weighting`](https://torchjd.org/stable/docs/aggregation#torchjd.aggregation.Weighting).
- Combine the losses using those weights and make a step of gradient descent on the combined loss.

The main advantage of this approach is to save memory because the Jacobian (that is typically large)
never has to be stored in memory. The [`torchjd.autogram.Engine`](https://torchjd.org/stable/docs/autogram/engine/) is precisely made
to compute the Gramian of the Jacobian efficiently.

The following example shows how to use the `autogram` engine to minimize the vector of per-instance losses with
Jacobian descent using [UPGrad](https://torchjd.org/stable/docs/aggregation/upgrad/).

```diff
  import torch
  from torch.nn import Linear, MSELoss, ReLU, Sequential
  from torch.optim import SGD

+ from torchjd.autogram import Engine
+ from torchjd.aggregation import UPGradWeighting

  model = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU(), Linear(3, 1), ReLU())

- loss_fn = MSELoss()
+ loss_fn = MSELoss(reduction="none")
  optimizer = SGD(model.parameters(), lr=0.1)

+ weighting = UPGradWeighting()
+ engine = Engine(model, batch_dim=0)

  inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
  targets = torch.randn(8, 16)  # 8 batches of 16 targets for the first task

  for input, target in zip(inputs, targets):
      output = model(input).squeeze(dim=1)  # shape [16]
-     loss = loss_fn(output, target)  # shape [1]
+     losses = loss_fn(output, target)  # shape [16]

-     loss.backward()
+     gramian = engine.compute_gramian(losses)  # shape: [16, 16]
+     weights = weighting(gramian)  # shape: [16]
+     losses.backward(weights)
      optimizer.step()
      optimizer.zero_grad()
```

You can even go one step further by considering the multiple tasks and each element of the batch
independently. We call that Instance-Wise Multitask Learning (IWMTL).

```python
import torch
from torch.nn import Linear, MSELoss, ReLU, Sequential
from torch.optim import SGD

from torchjd.aggregation import Flattening, UPGradWeighting
from torchjd.autogram import Engine

shared_module = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
task1_module = Linear(3, 1)
task2_module = Linear(3, 1)
params = [
    *shared_module.parameters(),
    *task1_module.parameters(),
    *task2_module.parameters(),
]

optimizer = SGD(params, lr=0.1)
mse = MSELoss(reduction="none")
weighting = Flattening(UPGradWeighting())
engine = Engine(shared_module, batch_dim=0)

inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
task1_targets = torch.randn(8, 16)  # 8 batches of 16 targets for the first task
task2_targets = torch.randn(8, 16)  # 8 batches of 16 targets for the second task

for input, target1, target2 in zip(inputs, task1_targets, task2_targets):
    features = shared_module(input)  # shape: [16, 3]
    out1 = task1_module(features).squeeze(1)  # shape: [16]
    out2 = task2_module(features).squeeze(1)  # shape: [16]

    # Compute the matrix of losses: one loss per element of the batch and per task
    losses = torch.stack([mse(out1, target1), mse(out2, target2)], dim=1)  # shape: [16, 2]

    # Compute the gramian (inner products between pairs of gradients of the losses)
    gramian = engine.compute_gramian(losses)  # shape: [16, 2, 2, 16]

    # Obtain the weights that lead to no conflict between reweighted gradients
    weights = weighting(gramian)  # shape: [16, 2]

    # Do the standard backward pass, but weighted using the obtained weights
    losses.backward(weights)
    optimizer.step()
    optimizer.zero_grad()
```

> [!NOTE]
> Here, because the losses are a matrix instead of a simple vector, we compute a *generalized
> Gramian* and we extract weights from it using a
> [GeneralizedWeighting](https://torchjd.org/stable/docs/aggregation/#torchjd.aggregation.GeneralizedWeighting).

More usage examples can be found [here](https://torchjd.org/stable/examples/).

## Supported Aggregators and Weightings
TorchJD provides many existing aggregators from the literature, listed in the following table.

<!-- recommended aggregators first, then alphabetical order -->
| Aggregator                                                                                                 | Weighting                                                                                                              | Publication                                                                                                                                                          |
|------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [UPGrad](https://torchjd.org/stable/docs/aggregation/upgrad/#torchjd.aggregation.UPGrad) (recommended) | [UPGradWeighting](https://torchjd.org/stable/docs/aggregation/upgrad/#torchjd.aggregation.UPGradWeighting)              | [Jacobian Descent For Multi-Objective Optimization](https://arxiv.org/pdf/2406.16232)                                                                                |
| [AlignedMTL](https://torchjd.org/stable/docs/aggregation/aligned_mtl#torchjd.aggregation.AlignedMTL)       | [AlignedMTLWeighting](https://torchjd.org/stable/docs/aggregation/aligned_mtl#torchjd.aggregation.AlignedMTLWeighting) | [Independent Component Alignment for Multi-Task Learning](https://arxiv.org/pdf/2305.19000)                                                                          |
| [CAGrad](https://torchjd.org/stable/docs/aggregation/cagrad#torchjd.aggregation.CAGrad)                    | [CAGradWeighting](https://torchjd.org/stable/docs/aggregation/cagrad#torchjd.aggregation.CAGradWeighting)              | [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048)                                                                         |
| [ConFIG](https://torchjd.org/stable/docs/aggregation/config#torchjd.aggregation.ConFIG)                    | -                                                                                                                      | [ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks](https://arxiv.org/pdf/2408.11104)                                                       |
| [Constant](https://torchjd.org/stable/docs/aggregation/constant#torchjd.aggregation.Constant)              | [ConstantWeighting](https://torchjd.org/stable/docs/aggregation/constant#torchjd.aggregation.ConstantWeighting)        | -                                                                                                                                                                    |
| [DualProj](https://torchjd.org/stable/docs/aggregation/dualproj#torchjd.aggregation.DualProj)              | [DualProjWeighting](https://torchjd.org/stable/docs/aggregation/dualproj#torchjd.aggregation.DualProjWeighting)        | [Gradient Episodic Memory for Continual Learning](https://arxiv.org/pdf/1706.08840)                                                                                  |
| [GradDrop](https://torchjd.org/stable/docs/aggregation/graddrop#torchjd.aggregation.GradDrop)              | -                                                                                                                      | [Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout](https://arxiv.org/pdf/2010.06808)                                                    |
| [IMTLG](https://torchjd.org/stable/docs/aggregation/imtl_g#torchjd.aggregation.IMTLG)                      | [IMTLGWeighting](https://torchjd.org/stable/docs/aggregation/imtl_g#torchjd.aggregation.IMTLGWeighting)                | [Towards Impartial Multi-task Learning](https://discovery.ucl.ac.uk/id/eprint/10120667/)                                                                             |
| [Krum](https://torchjd.org/stable/docs/aggregation/krum#torchjd.aggregation.Krum)                          | [KrumWeighting](https://torchjd.org/stable/docs/aggregation/krum#torchjd.aggregation.KrumWeighting)                    | [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf)  |
| [Mean](https://torchjd.org/stable/docs/aggregation/mean#torchjd.aggregation.Mean)                          | [MeanWeighting](https://torchjd.org/stable/docs/aggregation/mean#torchjd.aggregation.MeanWeighting)                    | -                                                                                                                                                                    |
| [MGDA](https://torchjd.org/stable/docs/aggregation/mgda#torchjd.aggregation.MGDA)                          | [MGDAWeighting](https://torchjd.org/stable/docs/aggregation/mgda#torchjd.aggregation.MGDAWeighting)                    | [Multiple-gradient descent algorithm (MGDA) for multiobjective optimization](https://www.sciencedirect.com/science/article/pii/S1631073X12000738)                    |
| [NashMTL](https://torchjd.org/stable/docs/aggregation/nash_mtl#torchjd.aggregation.NashMTL)                | -                                                                                                                      | [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017)                                                                                         |
| [PCGrad](https://torchjd.org/stable/docs/aggregation/pcgrad#torchjd.aggregation.PCGrad)                    | [PCGradWeighting](https://torchjd.org/stable/docs/aggregation/pcgrad#torchjd.aggregation.PCGradWeighting)              | [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782)                                                                                         |
| [Random](https://torchjd.org/stable/docs/aggregation/random#torchjd.aggregation.Random)                    | [RandomWeighting](https://torchjd.org/stable/docs/aggregation/random#torchjd.aggregation.RandomWeighting)              | [Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning](https://arxiv.org/pdf/2111.10603)                                              |
| [Sum](https://torchjd.org/stable/docs/aggregation/sum#torchjd.aggregation.Sum)                             | [SumWeighting](https://torchjd.org/stable/docs/aggregation/sum#torchjd.aggregation.SumWeighting)                       | -                                                                                                                                                                    |
| [Trimmed Mean](https://torchjd.org/stable/docs/aggregation/trimmed_mean#torchjd.aggregation.TrimmedMean)   | -                                                                                                                      | [Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates](https://proceedings.mlr.press/v80/yin18a/yin18a.pdf)                                      |

## Contribution
Please read the [Contribution page](CONTRIBUTING.md).

## Citation
If you use TorchJD for your research, please cite:
```
@article{jacobian_descent,
  title={Jacobian Descent For Multi-Objective Optimization},
  author={Quinton, Pierre and Rey, Valérian},
  journal={arXiv preprint arXiv:2406.16232},
  year={2024}
}
```
