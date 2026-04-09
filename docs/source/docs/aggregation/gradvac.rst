:hide-toc:

GradVac
=======

.. autodata:: torchjd.aggregation.DEFAULT_GRADVAC_EPS

The constructor argument ``group_type`` (default ``0``) sets **parameter granularity** for the
per-block cosine statistics in GradVac:

* ``0`` — **whole model** (``whole_model``): one block per task gradient row. Omit ``encoder`` and
  ``shared_params``.
* ``1`` — **all layer** (``all_layer``): one block per leaf submodule with parameters under
  ``encoder`` (same traversal as ``encoder.modules()`` in the reference formulation).
* ``2`` — **all matrix** (``all_matrix``): one block per tensor in ``shared_params``, in order. Use
  the same tensors as for the shared-parameter Jacobian columns (e.g. the parameters you would pass
  to a shared-gradient helper).

.. autoclass:: torchjd.aggregation.GradVac
    :members:
    :undoc-members:
    :exclude-members: forward
