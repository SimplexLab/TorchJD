Instance-Wise Multi-Task Learning (IWMTL)
=========================================

When training a model with multiple tasks, the gradients of the individual tasks are likely to
conflict. This is particularly true when looking at the individual (per-sample) gradients.
The :doc:`autogram engine <../docs/autogram/engine>` can be used to efficiently compute the Gramian
of the Jacobian of the matrix of per-sample and per-task losses. Weights can then be extracted from
this Gramian to reweight the gradients and resolve conflict entirely.

The following example shows how to do that.

.. code-block:: python
    :emphasize-lines: 5-6, 18-20, 31-32, 34-35, 37-38, 40-41

    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import Flattening, UPGradWeighting
    from torchjd.autogram import Engine

    shared_module = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
    task1_module = Linear(3, 1)
    task2_module = Linear(3, 1)
    params = [
        *shared_module.parameters(),
        *task1_module.parameters(),
        *task2_module.parameters(),
    ]

    optimizer = SGD(params, lr=0.1)
    mse = MSELoss(reduction="none")
    weighting = Flattening(UPGradWeighting())
    engine = Engine(shared_module, batch_dim=0)

    inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
    task1_targets = torch.randn(8, 16)  # 8 batches of 16 targets for the first task
    task2_targets = torch.randn(8, 16)  # 8 batches of 16 targets for the second task

    for input, target1, target2 in zip(inputs, task1_targets, task2_targets):
        features = shared_module(input)  # shape: [16, 3]
        out1 = task1_module(features).squeeze(1)  # shape: [16]
        out2 = task2_module(features).squeeze(1)  # shape: [16]

        # Compute the matrix of losses: one loss per element of the batch and per task
        losses = torch.stack([mse(out1, target1), mse(out2, target2)], dim=1)  # shape: [16, 2]

        # Compute the gramian (inner products between pairs of gradients of the losses)
        gramian = engine.compute_gramian(losses)  # shape: [16, 2, 2, 16]

        # Obtain the weights that lead to no conflict between reweighted gradients
        weights = weighting(gramian)  # shape: [16, 2]

        # Do the standard backward pass, but weighted using the obtained weights
        losses.backward(weights)
        optimizer.step()
        optimizer.zero_grad()

.. note::
    In this example, the tensor of losses is a matrix rather than a vector. The gramian is thus a
    4D tensor rather than a matrix, and a
    :class:`~torchjd.aggregation._weighting_bases.GeneralizedWeighting`, such as
    :class:`~torchjd.aggregation._flattening.Flattening`, has to be used to extract a matrix of
    weights from it. More information about ``GeneralizedWeighting`` can be found in the
    :doc:`../../docs/aggregation/index` page.
