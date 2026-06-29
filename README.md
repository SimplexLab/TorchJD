<div align="center">
  <img src="docs/source/_static/logo-light-mode.png" alt="TorchJD" width="400"/>
</div>

---

[![Doc](https://img.shields.io/badge/Doc-torchjd.org-blue?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPDI5NGJXd2dkbVZ5YzJsdmJqMGlNUzR3SWlCbGJtTnZaR2x1WnowaVZWUkdMVGdpSUhOMFlXNWtZV3h2Ym1VOUltNXZJajglMkJDajtoUjNKbFlYUmxaQ0IxYzJsdVp5QkxjbWwwWVRvZ2FIUjBjRG92TDJ0eWFYUmhMbTl5WndBdExUNEtDand4TFMwZ1EzSmxZWFJsWkNCMWMybHVaeUJLY21sallUb2daWE1nYUhSMGNEb3ZMMnh2WTJGc2FRQXRMVDRLT0NBZ0lHZHBkSFJvUFNJeU1EUTNMamN5Y0hRaUNpQWdJR2hsYVdkb2REMGlNakEwTnk0M01uQjBJZ29nSUNCSllXd2dkR2hsUFNJeU1EUTNMamN5SURJd05EY3VNamtoQ2lBZ0lIWnBaWGRDYjNnOUlqQWdNQ0F5TURRdU55Y2dNakEwTnk0eU1TQXlNakF3TURBdElERXVNQ0F3SURNME5pNHlNVE1nTkRZdU9ESXpJREF3UXpVd01EQWdOVFl3TURBaUNnPT0=)](https://torchjd.org)
[![Static Badge](https://img.shields.io/badge/%F0%9F%92%AC_ChatBot-chat.torchjd.org-blue?logo=%F0%9F%92%AC)](https://chat.torchjd.org)
[![Tests](https://github.com/SimplexLab/TorchJD/actions/workflows/checks.yml/badge.svg)](https://github.com/SimplexLab/TorchJD/actions/workflows/checks.yml)
[![codecov](https://codecov.io/gh/SimplexLab/TorchJD/graph/badge.svg?token=8AUCZE76QH)](https://codecov.io/gh/SimplexLab/TorchJD)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchjd)](https://pypi.org/project/torchjd/)
[![Static Badge](https://img.shields.io/badge/PyTorch-%3E%3D2.3-blue?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Static Badge](https://img.shields.io/badge/Discord%20-%20community%20-%20%235865F2?logo=discord&logoColor=%23FFFFFF&label=Discord)](https://discord.gg/76KkRnb3nk)

**TorchJD** is a PyTorch library for training neural networks with **multiple losses**. It supports two complementary approaches:

- **Scalarization** — combine losses into a single scalar before backprop, using methods from the literature (geometric mean, softmax weighting, etc.)
- **Jacobian descent** — compute the full Jacobian matrix and aggregate it into a conflict-aware update direction using state-of-the-art aggregators (UPGrad, MGDA, CAGrad, and many more)

The full documentation is available at [torchjd.org](https://torchjd.org).

## Installation

```bash
pip install "torchjd[quadprog_projector]"
```

This includes the dependencies required by UPGrad and DualProj. Some other aggregators may have
additional dependencies — refer to the [installation docs](https://torchjd.org/stable/installation).

## Quick start

### Scalarization

Scalarization methods combine losses into a single scalar loss, which is then optimized with standard gradient descent. This is the simplest approach and is often a strong baseline.

```python
import torch
from torch.nn import Linear, MSELoss, ReLU, Sequential
from torch.optim import SGD

from torchjd.scalarization import GeometricMean

model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
optimizer = SGD(model.parameters(), lr=0.1)
criterion = MSELoss()
scalarizer = GeometricMean()

inputs = torch.randn(16, 10)
task1_targets, task2_targets = torch.randn(16, 1), torch.randn(16, 1)

output = model(inputs)
losses = torch.stack([criterion(output, task1_targets), criterion(output, task2_targets)])
loss = scalarizer(losses)  # combines losses into a single scalar
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

### Jacobian descent

Jacobian descent computes the per-task gradients individually and aggregates them into a single conflict-aware update direction. This avoids the issue where averaging conflicting gradients harms one of the objectives.

```python
import torch
from torch.nn import Linear, MSELoss, ReLU, Sequential
from torch.optim import SGD

from torchjd.autojac import mtl_backward, jac_to_grad
from torchjd.aggregation import UPGrad

shared = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
task1_head = Linear(3, 1)
task2_head = Linear(3, 1)
params = [*shared.parameters(), *task1_head.parameters(), *task2_head.parameters()]

optimizer = SGD(params, lr=0.1)
criterion = MSELoss()
aggregator = UPGrad()

inputs = torch.randn(16, 10)
features = shared(inputs)
loss1 = criterion(task1_head(features), torch.randn(16, 1))
loss2 = criterion(task2_head(features), torch.randn(16, 1))

mtl_backward([loss1, loss2], features=features)
jac_to_grad(shared.parameters(), aggregator)
optimizer.step()
optimizer.zero_grad()
```

More usage examples — including the memory-efficient `autogram` engine, instance-wise risk minimization, and partial Jacobian descent — can be found [in the docs](https://torchjd.org/stable/examples/).

## Supported Scalarizers

| Scalarizer | Description |
|---|---|
| [Mean](https://torchjd.org/stable/docs/scalarization) | Average of losses (equal weighting) |
| [Sum](https://torchjd.org/stable/docs/scalarization) | Sum of losses |
| [Linear](https://torchjd.org/stable/docs/scalarization) | Fixed user-supplied weights |
| [GeometricMean](https://torchjd.org/stable/docs/scalarization) | Geometric mean (GLS) — [MultiNet++](https://arxiv.org/pdf/1902.08325) |
| [Random](https://torchjd.org/stable/docs/scalarization) | Random weights sampled each step — [RLW](https://arxiv.org/pdf/2111.10603) |

## Supported Aggregators and Weightings

| Aggregator | Weighting | Publication |
|---|---|---|
| [UPGrad](https://torchjd.org/stable/docs/aggregation/upgrad/#torchjd.aggregation.UPGrad)  | [UPGradWeighting](https://torchjd.org/stable/docs/aggregation/upgrad/#torchjd.aggregation.UPGradWeighting) | [Jacobian Descent For Multi-Objective Optimization](https://arxiv.org/pdf/2406.16232) |
| [AlignedMTL](https://torchjd.org/stable/docs/aggregation/aligned_mtl#torchjd.aggregation.AlignedMTL) | [AlignedMTLWeighting](https://torchjd.org/stable/docs/aggregation/aligned_mtl#torchjd.aggregation.AlignedMTLWeighting) | [Independent Component Alignment for Multi-Task Learning](https://arxiv.org/pdf/2305.19000) |
| [CAGrad](https://torchjd.org/stable/docs/aggregation/cagrad#torchjd.aggregation.CAGrad) | [CAGradWeighting](https://torchjd.org/stable/docs/aggregation/cagrad#torchjd.aggregation.CAGradWeighting) | [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048) |
| [ConFIG](https://torchjd.org/stable/docs/aggregation/config#torchjd.aggregation.ConFIG) | - | [ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks](https://arxiv.org/pdf/2408.11104) |
| [Constant](https://torchjd.org/stable/docs/aggregation/constant#torchjd.aggregation.Constant) | [ConstantWeighting](https://torchjd.org/stable/docs/aggregation/constant#torchjd.aggregation.ConstantWeighting) | - |
| - | [CRMOGMWeighting](https://torchjd.org/stable/docs/aggregation/cr_mogm/#torchjd.aggregation.CRMOGMWeighting) | [On the Convergence of Stochastic Multi-Objective Gradient Manipulation and Beyond](https://proceedings.neurips.cc/paper_files/paper/2022/file/f91bd64a3620aad8e70a27ad9cb3ca57-Paper-Conference.pdf) |
| [DualProj](https://torchjd.org/stable/docs/aggregation/dualproj#torchjd.aggregation.DualProj) | [DualProjWeighting](https://torchjd.org/stable/docs/aggregation/dualproj#torchjd.aggregation.DualProjWeighting) | [Gradient Episodic Memory for Continual Learning](https://arxiv.org/pdf/1706.08840) |
| [ExcessMTL](https://torchjd.org/stable/docs/aggregation/excess_mtl#torchjd.aggregation.ExcessMTL) | [ExcessMTLWeighting](https://torchjd.org/stable/docs/aggregation/excess_mtl#torchjd.aggregation.ExcessMTLWeighting) | [Robust Multi-Task Learning with Excess Risks](https://proceedings.mlr.press/v235/he24n.html) |
| [FairGrad](https://torchjd.org/stable/docs/aggregation/fairgrad#torchjd.aggregation.FairGrad) | [FairGradWeighting](https://torchjd.org/stable/docs/aggregation/fairgrad#torchjd.aggregation.FairGradWeighting) | [Fair Resource Allocation in Multi-Task Learning](https://arxiv.org/pdf/2402.15638) |
| [GradDrop](https://torchjd.org/stable/docs/aggregation/graddrop#torchjd.aggregation.GradDrop) | - | [Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout](https://arxiv.org/pdf/2010.06808) |
| [GradVac](https://torchjd.org/stable/docs/aggregation/gradvac#torchjd.aggregation.GradVac) | [GradVacWeighting](https://torchjd.org/stable/docs/aggregation/gradvac#torchjd.aggregation.GradVacWeighting) | [Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models](https://arxiv.org/pdf/2010.05874) |
| [IMTLG](https://torchjd.org/stable/docs/aggregation/imtl_g#torchjd.aggregation.IMTLG) | [IMTLGWeighting](https://torchjd.org/stable/docs/aggregation/imtl_g#torchjd.aggregation.IMTLGWeighting) | [Towards Impartial Multi-task Learning](https://discovery.ucl.ac.uk/id/eprint/10120667/) |
| [Krum](https://torchjd.org/stable/docs/aggregation/krum#torchjd.aggregation.Krum) | [KrumWeighting](https://torchjd.org/stable/docs/aggregation/krum#torchjd.aggregation.KrumWeighting) | [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf) |
| [Mean](https://torchjd.org/stable/docs/aggregation/mean#torchjd.aggregation.Mean) | [MeanWeighting](https://torchjd.org/stable/docs/aggregation/mean#torchjd.aggregation.MeanWeighting) | - |
| [MGDA](https://torchjd.org/stable/docs/aggregation/mgda#torchjd.aggregation.MGDA) | [MGDAWeighting](https://torchjd.org/stable/docs/aggregation/mgda#torchjd.aggregation.MGDAWeighting) | [Multiple-gradient descent algorithm (MGDA) for multiobjective optimization](https://comptes-rendus.academie-sciences.fr/mathematique/articles/10.1016/j.crma.2012.03.014/) |
| - | [MoDoWeighting](https://torchjd.org/stable/docs/aggregation/modo/#torchjd.aggregation.MoDoWeighting) | [Three-Way Trade-Off in Multi-Objective Learning](https://www.jmlr.org/papers/volume25/23-1287/23-1287.pdf) |
| [NashMTL](https://torchjd.org/stable/docs/aggregation/nash_mtl#torchjd.aggregation.NashMTL) | - | [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017) |
| [PCGrad](https://torchjd.org/stable/docs/aggregation/pcgrad#torchjd.aggregation.PCGrad) | [PCGradWeighting](https://torchjd.org/stable/docs/aggregation/pcgrad#torchjd.aggregation.PCGradWeighting) | [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782) |
| [Random](https://torchjd.org/stable/docs/aggregation/random#torchjd.aggregation.Random) | [RandomWeighting](https://torchjd.org/stable/docs/aggregation/random#torchjd.aggregation.RandomWeighting) | [Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning](https://arxiv.org/pdf/2111.10603) |
| - | [SDMGradWeighting](https://torchjd.org/stable/docs/aggregation/sdmgrad#torchjd.aggregation.SDMGradWeighting) | [Direction-oriented Multi-objective Learning](https://arxiv.org/pdf/2305.18409) |
| [Sum](https://torchjd.org/stable/docs/aggregation/sum#torchjd.aggregation.Sum) | [SumWeighting](https://torchjd.org/stable/docs/aggregation/sum#torchjd.aggregation.SumWeighting) | - |
| [Trimmed Mean](https://torchjd.org/stable/docs/aggregation/trimmed_mean#torchjd.aggregation.TrimmedMean) | - | [Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates](https://proceedings.mlr.press/v80/yin18a/yin18a.pdf) |

## Release Methodology

TorchJD follows [semantic versioning](https://semver.org/). Since the library is still in beta (`0.x.y`), we sometimes make interface changes in minor versions. Breaking changes are always documented in the [changelog](CHANGELOG.md) with migration instructions.

## Contribution

Please read the [Contributing guide](CONTRIBUTING.md).

Thanks to our amazing contributors:

[![Contributors](https://stg.contrib.rocks/image?repo=SimplexLab/TorchJD&max=240&columns=18)](https://github.com/SimplexLab/TorchJD/graphs/contributors)

## Citation

```bibtex
@article{jacobian_descent,
  title={Jacobian Descent For Multi-Objective Optimization},
  author={Quinton, Pierre and Rey, Valérian},
  journal={arXiv preprint arXiv:2406.16232},
  year={2024}
}
```
