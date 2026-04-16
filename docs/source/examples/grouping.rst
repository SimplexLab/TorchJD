Grouping
========

The aggregation can be made independently on groups of parameters, at different granularities. The
`Gradient Vaccine paper <https://arxiv.org/pdf/2010.05874>`_ introduces four strategies to partition
the parameters:

1. **Together** (baseline): one group covering all parameters. Corresponds to the `whole_model`
   stategy in the paper.

2. **Per network**: one group per top-level sub-network (e.g. encoder and decoder separately).
   Corresponds to the `enc_dec` stategy in the paper.

3. **Per layer**: one group per leaf module of the network. Corresponds to the `all_layer` stategy
   in the paper.

4. **Per tensor**: one group per individual parameter tensor. Corresponds to the `all_matrix`
   stategy in the paper.

In TorchJD, grouping is achieved by calling :func:`~torchjd.autojac.jac_to_grad` once per group
after :func:`~torchjd.autojac.backward` or :func:`~torchjd.autojac.mtl_backward`, with a dedicated
aggregator instance per group. For :class:`~torchjd.aggregation.Stateful` aggregators, each instance
should independently maintains its own state (e.g. the EMA :math:`\hat{\phi}` state in
:class:`~torchjd.aggregation.GradVac`, matching the per-block targets from the original paper).

.. note::
    The grouping is orthogonal to the choice between
    :func:`~torchjd.autojac.backward` vs :func:`~torchjd.autojac.mtl_backward`. Those functions
    determine *which* parameters receive Jacobians; grouping then determines *how* those Jacobians
    are partitioned for aggregation.

.. note::
    The examples below use :class:`~torchjd.aggregation.GradVac`, but the same pattern applies to
    any :class:`~torchjd.aggregation.Aggregator`.

1. Together
-----------

A single :class:`~torchjd.aggregation.Aggregator` instance aggregates all shared parameters
together. Cosine similarities are computed between the full task gradient vectors.

.. testcode::
    :emphasize-lines: 14, 21

    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import GradVac
    from torchjd.autojac import jac_to_grad, mtl_backward

    encoder = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
    task1_head, task2_head = Linear(3, 1), Linear(3, 1)
    optimizer = SGD([*encoder.parameters(), *task1_head.parameters(), *task2_head.parameters()], lr=0.1)
    loss_fn = MSELoss()
    inputs, t1, t2 = torch.randn(8, 16, 10), torch.randn(8, 16, 1), torch.randn(8, 16, 1)

    aggregator = GradVac()

    for x, y1, y2 in zip(inputs, t1, t2):
        features = encoder(x)
        loss1 = loss_fn(task1_head(features), y1)
        loss2 = loss_fn(task2_head(features), y2)
        mtl_backward([loss1, loss2], features=features)
        jac_to_grad(encoder.parameters(), aggregator)
        optimizer.step()
        optimizer.zero_grad()

2. Per network
--------------

One :class:`~torchjd.aggregation.Aggregator` instance per top-level sub-network. Here the model
is split into an encoder and a decoder; cosine similarities are computed separately within each.
Passing ``features=dec_out`` to :func:`~torchjd.autojac.mtl_backward` causes both sub-networks
to receive Jacobians, which are then aggregated independently.

.. testcode::
    :emphasize-lines: 8-9, 15-16, 24-25

    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import GradVac
    from torchjd.autojac import jac_to_grad, mtl_backward

    encoder = Sequential(Linear(10, 5), ReLU())
    decoder = Sequential(Linear(5, 3), ReLU())
    task1_head, task2_head = Linear(3, 1), Linear(3, 1)
    optimizer = SGD([*encoder.parameters(), *decoder.parameters(), *task1_head.parameters(), *task2_head.parameters()], lr=0.1)
    loss_fn = MSELoss()
    inputs, t1, t2 = torch.randn(8, 16, 10), torch.randn(8, 16, 1), torch.randn(8, 16, 1)

    encoder_aggregator = GradVac()
    decoder_aggregator = GradVac()

    for x, y1, y2 in zip(inputs, t1, t2):
        enc_out = encoder(x)
        dec_out = decoder(enc_out)
        loss1 = loss_fn(task1_head(dec_out), y1)
        loss2 = loss_fn(task2_head(dec_out), y2)
        mtl_backward([loss1, loss2], features=dec_out)
        jac_to_grad(encoder.parameters(), encoder_aggregator)
        jac_to_grad(decoder.parameters(), decoder_aggregator)
        optimizer.step()
        optimizer.zero_grad()

3. Per layer
------------

One :class:`~torchjd.aggregation.Aggregator` instance per leaf module. Cosine similarities are
computed per-layer between the task gradients.

.. testcode::
    :emphasize-lines: 14-15, 22-23

    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import GradVac
    from torchjd.autojac import jac_to_grad, mtl_backward

    encoder = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
    task1_head, task2_head = Linear(3, 1), Linear(3, 1)
    optimizer = SGD([*encoder.parameters(), *task1_head.parameters(), *task2_head.parameters()], lr=0.1)
    loss_fn = MSELoss()
    inputs, t1, t2 = torch.randn(8, 16, 10), torch.randn(8, 16, 1), torch.randn(8, 16, 1)

    leaf_layers = [m for m in encoder.modules() if list(m.parameters()) and not list(m.children())]
    aggregators = [GradVac() for _ in leaf_layers]

    for x, y1, y2 in zip(inputs, t1, t2):
        features = encoder(x)
        loss1 = loss_fn(task1_head(features), y1)
        loss2 = loss_fn(task2_head(features), y2)
        mtl_backward([loss1, loss2], features=features)
        for layer, aggregator in zip(leaf_layers, aggregators):
            jac_to_grad(layer.parameters(), aggregator)
        optimizer.step()
        optimizer.zero_grad()

4. Per parameter
----------------

One :class:`~torchjd.aggregation.Aggregator` instance per individual parameter tensor. Cosine
similarities are computed per-tensor between the task gradients (e.g. weights and biases of each
layer are treated as separate groups).

.. testcode::
    :emphasize-lines: 14-15, 22-23

    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import GradVac
    from torchjd.autojac import jac_to_grad, mtl_backward

    encoder = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
    task1_head, task2_head = Linear(3, 1), Linear(3, 1)
    optimizer = SGD([*encoder.parameters(), *task1_head.parameters(), *task2_head.parameters()], lr=0.1)
    loss_fn = MSELoss()
    inputs, t1, t2 = torch.randn(8, 16, 10), torch.randn(8, 16, 1), torch.randn(8, 16, 1)

    shared_params = list(encoder.parameters())
    aggregators = [GradVac() for _ in shared_params]

    for x, y1, y2 in zip(inputs, t1, t2):
        features = encoder(x)
        loss1 = loss_fn(task1_head(features), y1)
        loss2 = loss_fn(task2_head(features), y2)
        mtl_backward([loss1, loss2], features=features)
        for param, aggregator in zip(shared_params, aggregators):
            jac_to_grad([param], aggregator)
        optimizer.step()
        optimizer.zero_grad()
