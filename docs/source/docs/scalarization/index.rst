scalarization
=============

.. automodule:: torchjd.scalarization
   :no-members:

Abstract base classes
---------------------

.. autoclass:: torchjd.scalarization.Scalarizer
    :members: __call__

.. py:class:: torchjd.scalarization.Stateful

    Mixin adding a reset method.

    .. py:method:: reset()

        Resets the internal state.


.. toctree::
    :hidden:
    :maxdepth: 1

    constant.rst
    geometric_mean.rst
    mean.rst
    random.rst
    stch.rst
    sum.rst
    uw.rst
