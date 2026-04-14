Examples
========

This section contains some usage examples for TorchJD.

- :doc:`Basic Usage <basic_usage>` provides a toy example using :doc:`torchjd.backward
  <../docs/autojac/backward>` to make a step of Jacobian descent with the :doc:`UPGrad
  <../docs/aggregation/upgrad>` aggregator.
- :doc:`Instance-Wise Risk Minimization (IWRM) <iwrm>` provides an example in which we minimize the
  vector of per-instance losses, using stochastic sub-Jacobian descent (SSJD). It is compared to the
  usual minimization of the average loss, called empirical risk minimization (ERM), using stochastic
  gradient descent (SGD).
- :doc:`Partial Jacobian Descent for IWRM <partial_jd>` provides an example in which we minimize the
  vector of per-instance losses using stochastic sub-Jacobian descent, similar to our :doc:`IWRM <iwrm>`
  example. However, this method bases the aggregation decision on the Jacobian of the losses with respect
  to **only a subset** of the model's parameters, offering a trade-off between computational cost and
  aggregation precision.
- :doc:`Multi-Task Learning (MTL) <mtl>` provides an example of multi-task learning where Jacobian
  descent is used to optimize the vector of per-task losses of a multi-task model, using the
  dedicated backpropagation function :doc:`mtl_backward <../docs/autojac/mtl_backward>`.
- :doc:`Instance-Wise Multi-Task Learning (IWMTL) <iwmtl>` shows how to combine multi-task learning
  with instance-wise risk minimization: one loss per task and per element of the batch, using the
  :doc:`autogram.Engine <../docs/autogram/engine>` and a :doc:`GeneralizedWeighting
  <../docs/aggregation/index>`.
- :doc:`Recurrent Neural Network (RNN) <rnn>` shows how to apply Jacobian descent to RNN training,
  with one loss per output sequence element.
- :doc:`Monitoring Aggregations <monitoring>` shows how to monitor the aggregation performed by the
  aggregator, to check if Jacobian descent is prescribed for your use-case.
- :doc:`PyTorch Lightning Integration <lightning_integration>` showcases how to combine
  TorchJD with PyTorch Lightning, by providing an example implementation of a multi-task
  ``LightningModule`` optimized by Jacobian descent.
- :doc:`Automatic Mixed Precision <amp>` shows how to combine mixed precision training with TorchJD.

.. toctree::
    :hidden:

    basic_usage.rst
    iwrm.rst
    partial_jd.rst
    mtl.rst
    iwmtl.rst
    rnn.rst
    monitoring.rst
    lightning_integration.rst
    amp.rst
