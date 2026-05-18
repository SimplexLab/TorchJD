:hide-toc:

CR-MOGM
=======

.. autoclass:: torchjd.aggregation.CRMOGMWeighting
    :members: __call__, reset

.. note::
    The usage example in the docstring above imports
    ``WeightedAggregator`` / ``GramianWeightedAggregator`` from
    ``torchjd.aggregation._aggregator_bases``, which is a private module. These two
    aggregator base classes are not currently part of the public ``torchjd.aggregation``
    namespace, so this private-module import is the only path that works today. Promoting
    them to the public namespace is a separate decision left to the maintainers.
