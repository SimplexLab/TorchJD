:hide-toc:

|

.. image:: _static/logo-dark-mode.png
    :width: 400
    :alt: torchjd
    :align: center
    :class: only-dark, no-scaled-link

.. image:: _static/logo-light-mode.png
    :width: 400
    :alt: torchjd
    :align: center
    :class: only-light, no-scaled-link

|

TorchJD is a library enabling Jacobian descent with PyTorch, to train neural networks with multiple
objectives. It is based on the theory from `Jacobian Descent For Multi-Objective Optimization
<https://arxiv.org/pdf/2406.16232>`_ and several other related publications.

The main purpose is to jointly optimize multiple objectives without combining them into a single
scalar loss. When the objectives are conflicting, this can be the key to a successful and stable
optimization. To get started, check out our :doc:`basic usage example
<examples/basic_usage>`.

Gradient descent relies on gradients to optimize a single objective. Jacobian descent takes this
idea a step further, using the Jacobian to optimize multiple objectives. An important component of
Jacobian descent is the aggregator, which maps the Jacobian to an optimization step. In the page
:doc:`Aggregation <docs/aggregation/index>`, we provide an overview of the various aggregators
available in TorchJD, and their corresponding weightings.

A straightforward application of Jacobian descent is multi-task learning, in which the vector of
per-task losses has to be minimized. To start using TorchJD for multi-task learning, follow our
:doc:`MTL example <examples/mtl>`.

Another more interesting application is to consider separately the loss of each element in the
batch. This is what we define as :doc:`Instance-Wise Risk Minimization <examples/iwrm>` (IWRM).

The Gramian-based Jacobian descent algorithm provides a very efficient alternative way of
performing Jacobian descent. It consists in computing
the Gramian of the Jacobian iteratively during the backward pass (without ever storing the full
Jacobian in memory), weighting the losses using the information of the Gramian, and then computing
the gradient of the obtained weighted loss. The iterative computation of the Gramian corresponds to
Algorithm 3 of
`Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_. The
documentation and usage example of this algorithm is provided in
:doc:`autogram.Engine <docs/autogram/engine>`.

The original usage of the autogram engine is to compute the Gramian of the Jacobian very efficiently
for :doc:`IWRM <examples/iwrm>`. Another direct application is when considering one loss per element
of the batch and per task, in the context of multi-task learning. We call this
:doc:`Instance-Wise Risk Multi-Task Learning <examples/iwmtl>` (IWMTL).

TorchJD is open-source, under MIT License. The source code is available on
`GitHub <https://github.com/SimplexLab/TorchJD>`_.

.. toctree::
    :caption: Getting Started
    :hidden:

    jacobian_descent.md
    installation.md
    examples/index.rst

.. toctree::
    :caption: API Reference
    :hidden:

    docs/autogram/index.rst
    docs/autojac/index.rst
    docs/aggregation/index.rst
