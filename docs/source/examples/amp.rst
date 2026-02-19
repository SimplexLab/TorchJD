Automatic Mixed Precision (AMP)
===============================

In some cases, to save memory and reduce computation time, you may want to use `automatic mixed
precision <https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html>`_. Since the
`torch.amp.GradScaler <https://pytorch.org/docs/stable/amp.html#gradient-scaling>`_ class already
works on multiple losses, it's pretty straightforward to combine TorchJD and AMP. As usual, the
forward pass should be wrapped within a `torch.autocast
<https://pytorch.org/docs/stable/amp.html#torch.autocast>`_ context, and as usual, the loss (in our
case, the losses) should preferably be scaled with a `GradScaler
<https://pytorch.org/docs/stable/amp.html#gradient-scaling>`_ to avoid gradient underflow. The
following example shows the resulting code for a multi-task learning use-case.

.. code-block:: python
    :emphasize-lines: 2, 17, 27, 34-35, 37-38

    import torch
    from torch.amp import GradScaler
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
    scaler = GradScaler(device="cpu")
    loss_fn = MSELoss()
    optimizer = SGD(params, lr=0.1)
    aggregator = UPGrad()

    inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
    task1_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the first task
    task2_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the second task

    for input, target1, target2 in zip(inputs, task1_targets, task2_targets):
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            features = shared_module(input)
            output1 = task1_module(features)
            output2 = task2_module(features)
            loss1 = loss_fn(output1, target1)
            loss2 = loss_fn(output2, target2)

        scaled_losses = scaler.scale([loss1, loss2])
        mtl_backward(scaled_losses, features=features)
        jac_to_grad(shared_module.parameters(), aggregator)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

.. hint::
    Within the ``torch.autocast`` context, some operations may be done in ``float16`` type. For
    those operations, the tensors saved for the backward pass will also be of ``float16`` type.
    However, the Jacobian computed by ``mtl_backward`` will be of type ``float32``, so the ``.grad``
    fields of the model parameters will also be of type ``float32``. This is in line with the
    behavior of PyTorch, that would also compute all gradients in ``float32`` type.

.. note::
    :doc:`torchjd.backward <../docs/autojac/backward>` can be similarly combined with AMP.
