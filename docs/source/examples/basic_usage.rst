Basic Usage
===========

This example shows how to use TorchJD to perform an iteration of Jacobian descent on a regression
model with two objectives. In this example, a batch of inputs is forwarded through the model and two
corresponding batches of labels are used to compute two losses. These losses are then backwarded
through the model. The obtained Jacobian matrix, consisting of the gradients of the two losses with
respect to the parameters, is then aggregated using :doc:`UPGrad <../docs/aggregation/upgrad>`, and
the parameters are updated using the resulting aggregation.



Import several classes from ``torch`` and ``torchjd``:

.. testcode::

    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd import autojac
    from torchjd.aggregation import UPGrad
    from torchjd.autojac import jac_to_grad

Define the model and the optimizer, as usual:

.. testcode::

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 2))
    optimizer = SGD(model.parameters(), lr=0.1)

Define the aggregator that will be used to combine the Jacobian matrix:

.. testcode::

    aggregator = UPGrad()

In essence, :doc:`UPGrad <../docs/aggregation/upgrad>` projects each gradient onto the dual cone of
the rows of the Jacobian and averages the results. This ensures that locally, no loss will be
negatively affected by the update.

Now that everything is defined, we can train the model. Define the input and the associated target:

.. testcode::

    input = torch.randn(16, 10)  # Batch of 16 random input vectors of length 10
    target1 = torch.randn(16)  # First batch of 16 targets
    target2 = torch.randn(16)  # Second batch of 16 targets

Here, we generate fake inputs and labels for the sake of the example.

We can now compute the losses associated to each element of the batch.

.. testcode::

    loss_fn = MSELoss()
    output = model(input)
    loss1 = loss_fn(output[:, 0], target1)
    loss2 = loss_fn(output[:, 1], target2)

The last steps are similar to gradient descent-based optimization, but using the two losses.

Perform the Jacobian descent backward pass:

.. testcode::

    autojac.backward([loss1, loss2])
    jac_to_grad(model.parameters(), aggregator)

The first function will populate the ``.jac`` field of each model parameter with the corresponding
Jacobian, and the second one will aggregate these Jacobians and store the result in the ``.grad``
field of the parameters. It also deletes the ``.jac`` fields save some memory.

Update each parameter based on its ``.grad`` field, using the ``optimizer``:

.. testcode::

    optimizer.step()

The model's parameters have been updated!

As usual, you should now reset the ``.grad`` field of each model parameter:

.. testcode::

    optimizer.zero_grad()
