# TorchJD: GradNorm Integration

This fork adds the `GradNormScalarizer` to the `TorchJD` library to support dynamic loss balancing in multi-task learning.

## Key Features
- Dynamic gradient norm balancing.
- Easy integration with existing `Scalarizer` interface.

## Usage
```python
from torchjd.scalarization import GradNormScalarizer

# Initialize the scalarizer
scalarizer = GradNormScalarizer(num_tasks=3)
