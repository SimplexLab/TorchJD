Recurrent Neural Network (RNN)
==============================

When training recurrent neural networks for sequence modelling, we can easily obtain one loss per
element of the output sequences. If the gradients of these losses are likely to conflict, Jacobian
descent can be leveraged to enhance optimization.

.. code-block:: python
    :emphasize-lines: 5-6, 10, 17, 19-20

    import torch
    from torch.nn import RNN
    from torch.optim import SGD

    from torchjd.aggregation import UPGrad
    from torchjd.autojac import backward, jac_to_grad

    rnn = RNN(input_size=10, hidden_size=20, num_layers=2)
    optimizer = SGD(rnn.parameters(), lr=0.1)
    aggregator = UPGrad()

    inputs = torch.randn(8, 5, 3, 10)  # 8 batches of 3 sequences of length 5 and of dim 10.
    targets = torch.randn(8, 5, 3, 20)  # 8 batches of 3 sequences of length 5 and of dim 20.

    for input, target in zip(inputs, targets):
        output, _ = rnn(input)  # output is of shape [5, 3, 20].
        losses = ((output - target) ** 2).mean(dim=[1, 2])  # 1 loss per sequence element.

        backward(losses, parallel_chunk_size=1)
        jac_to_grad(rnn.parameters(), aggregator)
        optimizer.step()
        optimizer.zero_grad()

.. note::
    At the time of writing, there seems to be an incompatibility between ``torch.vmap`` and
    ``torch.nn.RNN`` when running on CUDA (see `this issue
    <https://github.com/SimplexLab/torchjd/issues/220>`_ for more info), so we advise to set the
    ``parallel_chunk_size`` to ``1`` to avoid using ``torch.vmap``. To improve performance, you can
    check whether ``parallel_chunk_size=None`` (maximal parallelization) works on your side.
