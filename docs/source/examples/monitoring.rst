Monitoring aggregations
=======================

The :doc:`Aggregator <../docs/aggregation/index>` class is a subclass of :class:`torch.nn.Module`.
This allows registering hooks, which can be used to monitor some information about aggregations.
The following code example demonstrates registering a hook to compute and print the cosine
similarity between the aggregation performed by :doc:`UPGrad <../docs/aggregation/upgrad>` and the
average of the gradients, and another hook to compute and print the weights of the weighting of
:doc:`UPGrad <../docs/aggregation/upgrad>`.

Updating the parameters of the model with the average gradient is equivalent to using gradient
descent on the average of the losses. Observing a cosine similarity smaller than 1 means that
Jacobian descent is doing something different than gradient descent. With
:doc:`UPGrad <../docs/aggregation/upgrad>`, this happens when the original gradients conflict (i.e.
they have a negative inner product).

.. code-block:: python
    :emphasize-lines: 9-11, 13-18, 33-34

    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.nn.functional import cosine_similarity
    from torch.optim import SGD

    from torchjd.aggregation import UPGrad
    from torchjd.autojac import mtl_backward, jac_to_grad

    def print_weights(_, __, weights: torch.Tensor) -> None:
        """Prints the extracted weights."""
        print(f"Weights: {weights}")

    def print_gd_similarity(_, inputs: tuple[torch.Tensor, ...], aggregation: torch.Tensor) -> None:
        """Prints the cosine similarity between the aggregation and the average gradient."""
        matrix = inputs[0]
        gd_output = matrix.mean(dim=0)
        similarity = cosine_similarity(aggregation, gd_output, dim=0)
        print(f"Cosine similarity: {similarity.item():.4f}")

    shared_module = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
    task1_module = Linear(3, 1)
    task2_module = Linear(3, 1)
    params = [
        *shared_module.parameters(),
        *task1_module.parameters(),
        *task2_module.parameters(),
    ]

    loss_fn = MSELoss()
    optimizer = SGD(params, lr=0.1)
    aggregator = UPGrad()

    aggregator.gramian_weighting.register_forward_hook(print_weights)
    aggregator.register_forward_hook(print_gd_similarity)

    inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
    task1_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the first task
    task2_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the second task

    for input, target1, target2 in zip(inputs, task1_targets, task2_targets):
        features = shared_module(input)
        output1 = task1_module(features)
        output2 = task2_module(features)
        loss1 = loss_fn(output1, target1)
        loss2 = loss_fn(output2, target2)

        mtl_backward(tensors=[loss1, loss2], features=features)
        jac_to_grad(shared_module.parameters(), aggregator)
        optimizer.step()
        optimizer.zero_grad()
