Partial Jacobian Descent for IWRM
=================================

This example demonstrates how to perform Partial Jacobian Descent using TorchJD. This technique
minimizes a vector of per-instance losses by resolving conflict only based on a submatrix of the
Jacobian — specifically, the portion corresponding to a selected subset of the model's parameters.
This approach offers a trade-off between the precision of the aggregation decision and the
computational cost associated with computing the Gramian of the full Jacobian. For a complete,
non-partial version, see the :doc:`IWRM <iwrm>` example.

In this example, our model consists of three ``Linear`` layers separated by ``ReLU`` layers. We
perform the partial descent by considering only the parameters of the last two ``Linear`` layers. By
doing this, we avoid computing the Jacobian and its Gramian with respect to the parameters of the
first ``Linear`` layer, thereby reducing memory usage and computation time.

.. code-block:: python
    :emphasize-lines: 16-18

    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import UPGradWeighting
    from torchjd.autogram import Engine

    X = torch.randn(8, 16, 10)
    Y = torch.randn(8, 16)

    model = Sequential(Linear(10, 8), ReLU(), Linear(8, 5), ReLU(), Linear(5, 1))
    loss_fn = MSELoss(reduction="none")

    weighting = UPGradWeighting()

    # Create the autogram engine that will compute the Gramian of the
    # Jacobian with respect to the two last Linear layers' parameters.
    engine = Engine(model[2:], batch_dim=0)

    params = model.parameters()
    optimizer = SGD(params, lr=0.1)

    for x, y in zip(X, Y):
        y_hat = model(x).squeeze(dim=1)  # shape: [16]
        losses = loss_fn(y_hat, y)  # shape: [16]
        gramian = engine.compute_gramian(losses)
        weights = weighting(gramian)
        losses.backward(weights)
        optimizer.step()
        optimizer.zero_grad()
