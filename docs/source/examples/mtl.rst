Multi-Task Learning (MTL)
=========================

In the context of multi-task learning, multiple tasks are performed simultaneously on a common
input. Typically, a feature extractor is applied to the input to obtain a shared representation,
useful for all tasks. Then, task-specific heads are applied to these features to obtain each task's
result. A loss can then be computed for each task. Fundamentally, multi-task learning is a
multi-objective optimization problem in which we minimize the vector of task losses.

A common trick to train multi-task models is to cast the problem as single-objective, by minimizing
a weighted sum of the losses. This works well in some cases, but sometimes conflict among tasks can
make the optimization of the shared parameters very hard. Besides, the weight associated to each
loss can be considered as a hyper-parameter. Finding their optimal value is generally expensive.

Alternatively, the vector of losses can be directly minimized using Jacobian descent. The following
example shows how to use TorchJD to train a very simple multi-task model with two regression tasks.
For the sake of the example, we generate a fake dataset consisting of 8 batches of 16 random input
vectors of dimension 10, and their corresponding scalar labels for both tasks.


.. code-block:: python
    :emphasize-lines: 5-6, 19, 32-33

    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import UPGrad
    from torchjd.autojac import mtl_backward, jac_to_grad

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

.. note::
    In this example, the Jacobian is only with respect to the shared parameters. The task-specific
    parameters are simply updated via the gradient of their task's loss with respect to them.
