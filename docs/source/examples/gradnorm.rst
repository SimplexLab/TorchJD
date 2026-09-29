Adaptive loss balancing with GradNorm
======================================

GradNorm adjusts task weights using their gradient norms and their progress relative to the
initial losses. The paper computes these norms over the last shared layer. The resulting weights
still apply to every model parameter, including the task heads.

This example uses :class:`~torchjd.autogram.Engine` to compute the Gramian for that layer and
:class:`~torchjd.aggregation.GradNormWeighting` to form the auxiliary balancing loss. A separate
optimizer learns the task weights. Their sum is restored to the number of tasks after each step.
Both optimizers use gradients computed with the weights from before the step.

.. testcode::

    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import Adam, SGD

    from torchjd.aggregation import GradNormWeighting
    from torchjd.autogram import Engine

    shared = Sequential(Linear(5, 4), ReLU(), Linear(4, 3))
    heads = [Linear(3, 1), Linear(3, 1)]
    parameters = [*shared.parameters(), *(p for head in heads for p in head.parameters())]
    model_optimizer = SGD(parameters, lr=0.01)
    weighting = GradNormWeighting(n_tasks=2, alpha=1.5)
    weight_optimizer = Adam(weighting.parameters(), lr=0.001)
    criterion = MSELoss()
    engine = Engine(shared[2], batch_dim=None)

    inputs = torch.randn(4, 8, 5)
    targets = torch.randn(4, 8, 2)

    for features, target in zip(inputs, targets):
        model_optimizer.zero_grad()
        weight_optimizer.zero_grad()
        representation = shared(features)
        losses = torch.stack([
            criterion(head(representation).squeeze(1), target[:, i])
            for i, head in enumerate(heads)
        ])
        gramian = engine.compute_gramian(losses)
        weighting.set_losses(losses)
        weights = weighting(gramian)
        weighting.balancing_loss().backward()
        losses.backward(weights)
        model_optimizer.step()
        weight_optimizer.step()
        weighting.renormalize()

The auxiliary backward affects only the task weights. The model backward uses detached weights,
so its gradients cannot alter the balancing update. Computing the Gramian on all model parameters
instead would include the task heads in the norms, which differs from the paper's choice.
The losses are already averaged over the batch, so the engine uses ``batch_dim=None``.

Save both optimizers' states along with the model and weighting ``state_dict()`` to resume
training. The weighting stores its initial losses and learned weights; the next batch must still
call ``set_losses`` and the forward. Its ``reset()`` method starts a new loss baseline and restores
unit weights. Reset the external optimizer as well when starting a new experiment.
