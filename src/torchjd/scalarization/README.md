# Scalarization

A `Scalarizer` reduces a tensor of values (typically a vector of per-task or per-instance losses)
into a single scalar that can be optimized with a standard `loss.backward()`. Scalarizers are the
simple baseline against which the Jacobian-descent [aggregators](../aggregation) are compared:
instead of combining the per-loss gradients, a scalarizer combines the losses directly.

Full documentation for every scalarizer is at
[torchjd.org](https://torchjd.org/latest/docs/scalarization/).

## Usage

```python
import torch
from torch.nn import Linear
from torchjd.scalarization import Mean

model = Linear(3, 2)
scalarizer = Mean()

features = torch.randn(8, 3)
losses = model(features).pow(2).mean(dim=0)  # one loss per output dimension
loss = scalarizer(losses)
loss.backward()  # gradients flow to the model parameters
```

## Available scalarizers

- **Constant**: combines the values with constant, pre-determined weights.
- **COSMOS**: linear scalarization minus a cosine-similarity penalty toward a preference direction.
- **DWA**: weights each value by the relative rate at which its loss decreased over the two previous
  epochs.
- **FAMO**: decreases all task losses at an approximately equal rate, learning the task weights
  internally.
- **GeometricMean**: geometric mean of the values (also known as GLS).
- **IMTLL**: learns a per-task scale and combines the values as the sum of `exp(s_i) * L_i - s_i`.
- **Mean**: mean of the values.
- **PBI**: decomposes the values along a preference direction and penalizes the perpendicular
  distance.
- **Random**: combines the values with positive random weights summing to one.
- **STCH**: smooth approximation of the weighted, shifted maximum of the values.
- **Sum**: sum of the values.
- **UW**: weights the values using learned per-task uncertainties.

`UW`, `IMTLL`, and `FAMO` are trainable, and `DWA` and `FAMO` carry state between calls, so they
need a little more than a single call (an optimizer, a per-epoch `step()`, or an `update()`). See
the documentation for the exact usage.
