"""
A :class:`~torchjd.scalarization.Scalarizer` reduces a tensor of values of any shape into a single
scalar value. This is the simple baseline
against which :class:`Aggregators <torchjd.aggregation.Aggregator>` are compared: instead of
combining the per-loss gradients via the Jacobian or its Gramian, a
:class:`~torchjd.scalarization.Scalarizer` combines the losses directly, and a standard call to
:meth:`~torch.Tensor.backward` produces the gradient.

The following example shows how to use :class:`~torchjd.scalarization.Mean` to combine a vector of
losses into a single scalar loss.

>>> from torch import tensor
>>> from torchjd.scalarization import Mean
>>>
>>> scalarizer = Mean()
>>> losses = tensor([1.0, 2.0, 3.0])
>>> loss = scalarizer(losses)
>>> loss
tensor(2.)
"""

from ._constant import Constant
from ._mean import Mean
from ._random import Random
from ._scalarizer_base import Scalarizer
from ._sum import Sum

__all__ = [
    "Constant",
    "Mean",
    "Random",
    "Scalarizer",
    "Sum",
]
