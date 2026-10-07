"""
The autogram package provides an engine to efficiently compute the Gramian of the Jacobian of a
tensor of outputs (generally losses) with respect to some modules' parameters. This Gramian contains
all the inner products between pairs of gradients, and is thus a sufficient statistic for most
weighting methods. The algorithm is formally defined in Section 6 of `Jacobian Descent For
Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

Due to computing the Gramian iteratively over the layers, without ever having to store the full
Jacobian in memory, this method is much more memory-efficient than
:doc:`autojac <../autojac/index>`, which makes it often much faster. Note that we're still working
on making autogram faster and more memory-efficient, and its interface may change in future
releases.

The list of Weightings compatible with ``autogram`` is:

* :class:`~torchjd.aggregation.UPGradWeighting`
* :class:`~torchjd.aggregation.AlignedMTLWeighting`
* :class:`~torchjd.aggregation.CAGradWeighting`
* :class:`~torchjd.aggregation.ConstantWeighting`
* :class:`~torchjd.aggregation.CRMOGMWeighting` (when wrapping another weighting from this list)
* :class:`~torchjd.aggregation.DualProjWeighting`
* :class:`~torchjd.aggregation.FairGradWeighting`
* :class:`~torchjd.aggregation.GradVacWeighting`
* :class:`~torchjd.aggregation.IMTLGWeighting`
* :class:`~torchjd.aggregation.KrumWeighting`
* :class:`~torchjd.aggregation.MeanWeighting`
* :class:`~torchjd.aggregation.MGDAWeighting`
* :class:`~torchjd.aggregation.PCGradWeighting`
* :class:`~torchjd.aggregation.RandomWeighting`
* :class:`~torchjd.aggregation.SumWeighting`
"""

from ._engine import Engine

__all__ = ["Engine"]
