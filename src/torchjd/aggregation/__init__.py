r"""
When doing Jacobian descent, the Jacobian matrix has to be aggregated into a vector to store in the
``.grad`` fields of the model parameters. The
:class:`~torchjd.aggregation._aggregator_bases.Aggregator` is responsible for these aggregations.

When using the :doc:`autogram <../autogram/index>` engine, we rather need to extract a vector
of weights from the Gramian of the Jacobian. The
:class:`~torchjd.aggregation._weighting_bases.Weighting` is responsible for this.

.. note::
    Most aggregators rely on computing the Gramian of the Jacobian, extracting a vector of weights
    from this Gramian using a :class:`~torchjd.aggregation._weighting_bases.Weighting`, and then
    combining the rows of the Jacobian using these weights. For all of them, we provide both the
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` interface (to be used in autojac) and
    the :class:`~torchjd.aggregation._weighting_bases.Weighting` interface (to be used in autogram).
    For the rest, we only provide the :class:`~torchjd.aggregation._aggregator_bases.Aggregator`
    interface -- they are not compatible with autogram.

:class:`Aggregators <torchjd.aggregation._aggregator_bases.Aggregator>` and :class:`Weightings
<torchjd.aggregation._weighting_bases.Weighting>` are callables that take a Jacobian matrix or a
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

When dealing with a more general tensor of objectives, of shape ``[m_1, ..., m_k]`` (i.e. not
necessarily a simple vector), the Jacobian will be of shape ``[m_1, ..., m_k, n]``, and its Gramian
will be called a `generalized Gramian`, of shape ``[m_1, ..., m_k, m_k, ..., m_1]``. One can use a
:class:`GeneralizedWeighting<torchjd.aggregation._weighting_bases.GeneralizedWeighting>` to extract
a tensor of weights (of shape ``[m_1, ..., m_k]``) from such a generalized Gramian. The simplest
:class:`GeneralizedWeighting<torchjd.aggregation._weighting_bases.GeneralizedWeighting>` is
:class:`Flattening<torchjd.aggregation._flattening.Flattening>`: it simply "flattens" the
generalized Gramian into a square Gramian matrix (of shape ``[m_1 * ... * m_k, m_1 * ... * m_k]``),
applies a normal weighting to it to obtain a vector of weights, and returns the reshaped tensor of
weights.

>>> from torch import ones
>>> from torchjd.aggregation import Flattening, UPGradWeighting
>>>
>>> weighting = Flattening(UPGradWeighting())
>>> # Generate a generalized Gramian filled with ones, for the sake of the example
>>> generalized_gramian = ones((2, 3, 3, 2))
>>> weights = weighting(generalized_gramian)
>>> weights
tensor([[0.1667, 0.1667, 0.1667],
        [0.1667, 0.1667, 0.1667]])
"""

from ._aggregator_bases import Aggregator
from ._aligned_mtl import AlignedMTL, AlignedMTLWeighting
from ._config import ConFIG
from ._constant import Constant, ConstantWeighting
from ._dualproj import DualProj, DualProjWeighting
from ._flattening import Flattening
from ._graddrop import GradDrop
from ._imtl_g import IMTLG, IMTLGWeighting
from ._krum import Krum, KrumWeighting
from ._mean import Mean, MeanWeighting
from ._mgda import MGDA, MGDAWeighting
from ._pcgrad import PCGrad, PCGradWeighting
from ._random import Random, RandomWeighting
from ._stch import STCH, STCHWeighting
from ._sum import Sum, SumWeighting
from ._trimmed_mean import TrimmedMean
from ._upgrad import UPGrad, UPGradWeighting
from ._utils.check_dependencies import (
    OptionalDepsNotInstalledError as _OptionalDepsNotInstalledError,
)
from ._weighting_bases import GeneralizedWeighting, Weighting

__all__ = [
    "Aggregator",
    "AlignedMTL",
    "AlignedMTLWeighting",
    "ConFIG",
    "Constant",
    "ConstantWeighting",
    "DualProj",
    "DualProjWeighting",
    "Flattening",
    "GeneralizedWeighting",
    "GradDrop",
    "IMTLG",
    "IMTLGWeighting",
    "Krum",
    "KrumWeighting",
    "Mean",
    "MeanWeighting",
    "MGDA",
    "MGDAWeighting",
    "PCGrad",
    "PCGradWeighting",
    "Random",
    "RandomWeighting",
    "STCH",
    "STCHWeighting",
    "Sum",
    "SumWeighting",
    "TrimmedMean",
    "UPGrad",
    "UPGradWeighting",
    "Weighting",
]

try:
    from ._cagrad import CAGrad, CAGradWeighting

    __all__ += ["CAGrad", "CAGradWeighting"]
except _OptionalDepsNotInstalledError:  # The required dependencies are not installed
    pass

try:
    from ._nash_mtl import NashMTL

    __all__ += ["NashMTL"]
except _OptionalDepsNotInstalledError:  # The required dependencies are not installed
    pass
