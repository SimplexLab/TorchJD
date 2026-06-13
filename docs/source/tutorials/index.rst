Tutorials
=========

This section contains tutorials that walk you through more advanced usage patterns of TorchJD.

- :doc:`Instance-Wise Risk Minimization (IWRM) <iwrm>` demonstrates how to minimize the vector of
  per-instance losses, using stochastic sub-Jacobian descent (SSJD), compared to the usual
  minimization of the average loss (ERM) with stochastic gradient descent (SGD).
- :doc:`Partial Jacobian Descent for IWRM <partial_jd>` shows how to base the aggregation decision
  on the Jacobian of the losses with respect to only a subset of the model's parameters, offering a
  trade-off between computational cost and aggregation precision.
- :doc:`Multi-Task Learning (MTL) <mtl>` walks through multi-task learning where Jacobian descent
  optimizes the vector of per-task losses of a multi-task model, using the dedicated
  backpropagation function :doc:`mtl_backward <../reference/autojac/mtl_backward>`.
- :doc:`Instance-Wise Multi-Task Learning (IWMTL) <iwmtl>` shows how to combine multi-task
  learning with instance-wise risk minimization: one loss per task and per element of the batch,
  using the :doc:`autogram.Engine <../reference/autogram/engine>`.
- :doc:`Recurrent Neural Network (RNN) <rnn>` shows how to apply Jacobian descent to RNN training,
  with one loss per output sequence element.
- :doc:`Monitoring Aggregations <monitoring>` shows how to monitor the aggregation performed by the
  aggregator, to check if Jacobian descent is prescribed for your use-case.
- :doc:`PyTorch Lightning Integration <lightning_integration>` showcases how to combine TorchJD
  with PyTorch Lightning, by providing an example implementation of a multi-task
  ``LightningModule`` optimized by Jacobian descent.
- :doc:`Grouping <grouping>` shows how to apply an aggregator independently per parameter group
  (e.g. per layer), so that conflict resolution happens at a finer granularity than the full
  parameter vector.
- :doc:`Automatic Mixed Precision <amp>` shows how to combine mixed precision training with TorchJD.

.. toctree::
    :hidden:

    iwrm.rst
    partial_jd.rst
    mtl.rst
    iwmtl.rst
    rnn.rst
    monitoring.rst
    lightning_integration.rst
    amp.rst
    grouping.rst
