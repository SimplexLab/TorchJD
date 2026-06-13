r"""
When doing Jacobian descent, the Jacobian matrix has to be aggregated into a vector to store in the
``.grad`` fields of the model parameters. The
:class:`~torchjd.aggregation.Aggregator` is responsible for these aggregations.

When using the :doc:`autogram <../autogram/index>` engine, we rather need to extract a vector
of weights from the Gramian of the Jacobian. The
:class:`~torchjd.aggregation.Weighting` is responsible for this.

.. note::
    Most aggregators rely on computing the Gramian of the Jacobian, extracting a vector of weights
    from this Gramian using a :class:`~torchjd.aggregation.Weighting`
    [:class:`~torchjd.linalg.PSDMatrix`], and then combining the rows of the Jacobian using these
    weights. For all of them, we provide both the
    :class:`~torchjd.aggregation.Aggregator` interface (to be used in autojac) and the
    :class:`~torchjd.aggregation.Weighting` interface (to be used in autogram).
    For the rest, we only provide the :class:`~torchjd.aggregation.Aggregator`
    interface -- they are not compatible with autogram.

:class:`Aggregators <torchjd.aggregation.Aggregator>` and
:class:`Weightings <torchjd.aggregation.Weighting>` are callables that take a Jacobian matrix or a
Gramian matrix as inputs, respectively. The following example shows how to use UPGrad to either
aggregate a Jacobian (of shape ``[m, n]``, where ``m`` is the number of objectives and ``n`` is the
number of parameters), or obtain the weights from the Gramian of the Jacobian (of shape ``[m, m]``).

>>> from torch import tensor
>>> from torchjd.aggregation import UPGrad, UPGradWeighting
>>>
>>> aggregator = UPGrad()
>>> jacobian = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
>>> aggregation = aggregator(jacobian)
>>> aggregation
tensor([0.2929, 1.9004, 1.9004])
>>> weighting = UPGradWeighting()
>>> gramian = jacobian @ jacobian.T
>>> weights = weighting(gramian)
>>> weights
tensor([1.1109, 0.7894])
"""

from ._aggregator_bases import Aggregator, GramianWeightedAggregator, WeightedAggregator
from ._aligned_mtl import AlignedMTL, AlignedMTLWeighting
from ._cagrad import CAGrad, CAGradWeighting
from ._config import ConFIG
from ._constant import Constant, ConstantWeighting
from ._cr_mogm import CRMOGMWeighting
from ._dualproj import DualProj, DualProjWeighting
from ._fairgrad import FairGrad, FairGradWeighting
from ._graddrop import GradDrop
from ._gradvac import GradVac, GradVacWeighting
from ._imtl_g import IMTLG, IMTLGWeighting
from ._krum import Krum, KrumWeighting
from ._mean import Mean, MeanWeighting
from ._mgda import MGDA, MGDAWeighting
from ._modo import MoDoWeighting
from ._nash_mtl import NashMTL
from ._pcgrad import PCGrad, PCGradWeighting
from ._random import Random, RandomWeighting
from ._sum import Sum, SumWeighting
from ._trimmed_mean import TrimmedMean
from ._upgrad import UPGrad, UPGradWeighting
from ._weighting_bases import Weighting

__all__ = [
    "Aggregator",
    "AlignedMTL",
    "AlignedMTLWeighting",
    "CAGrad",
    "CAGradWeighting",
    "ConFIG",
    "Constant",
    "ConstantWeighting",
    "CRMOGMWeighting",
    "DualProj",
    "DualProjWeighting",
    "FairGrad",
    "FairGradWeighting",
    "GradDrop",
    "GradVac",
    "GradVacWeighting",
    "GramianWeightedAggregator",
    "IMTLG",
    "IMTLGWeighting",
    "Krum",
    "KrumWeighting",
    "Mean",
    "MeanWeighting",
    "MGDA",
    "MGDAWeighting",
    "MoDoWeighting",
    "NashMTL",
    "PCGrad",
    "PCGradWeighting",
    "Random",
    "RandomWeighting",
    "Sum",
    "SumWeighting",
    "TrimmedMean",
    "UPGrad",
    "UPGradWeighting",
    "WeightedAggregator",
    "Weighting",
]
