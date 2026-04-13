Grouping
========

When applying a conflict-resolving aggregator such as :class:`~torchjd.aggregation.GradVac` in
multi-task learning, the cosine similarities between task gradients can be computed at different
granularities. The GradVac paper introduces four strategies, each partitioning the shared
parameter vector differently:

1. **Whole Model** (default) — one group covering all shared parameters.
2. **Encoder-Decoder** — one group per top-level sub-network (e.g. encoder and decoder separately).
3. **All Layers** — one group per leaf module of the encoder.
4. **All Matrices** — one group per individual parameter tensor.

In TorchJD, grouping is achieved by calling :func:`~torchjd.autojac.jac_to_grad` once per group
after :func:`~torchjd.autojac.mtl_backward`, with a dedicated aggregator instance per group.
For stateful aggregators such as :class:`~torchjd.aggregation.GradVac`, each instance
independently maintains its own EMA state :math:`\hat{\phi}`, matching the per-block targets from
the original paper.

.. note::
    The grouping is orthogonal to the choice of
    :func:`~torchjd.autojac.backward` vs :func:`~torchjd.autojac.mtl_backward`. Those functions
    determine *which* parameters receive Jacobians; grouping then determines *how* those Jacobians
    are partitioned for aggregation. Calling :func:`~torchjd.autojac.jac_to_grad` once on all shared
    parameters corresponds to the Whole Model strategy. Splitting those parameters into
    sub-networks and calling :func:`~torchjd.autojac.jac_to_grad` separately on each — with a
    dedicated aggregator per sub-network — gives an arbitrary custom grouping, such as the
    Encoder-Decoder strategy described in the GradVac paper for encoder-decoder architectures.

.. note::
    The examples below use :class:`~torchjd.aggregation.GradVac`, but the same pattern applies to
    any aggregator.

1. Whole Model
--------------

A single :class:`~torchjd.aggregation.GradVac` instance aggregates all shared parameters
together. Cosine similarities are computed between the full task gradient vectors.

.. testcode::
    :emphasize-lines: 14, 19

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

    gradvac = GradVac()

    for x, y1, y2 in zip(inputs, t1, t2):
        features = encoder(x)
        mtl_backward([loss_fn(task1_head(features), y1), loss_fn(task2_head(features), y2)], features=features)
        jac_to_grad(encoder.parameters(), gradvac)
        optimizer.step()
        optimizer.zero_grad()

2. Encoder-Decoder
------------------

One :class:`~torchjd.aggregation.GradVac` instance per top-level sub-network. Here the model
is split into an encoder and a decoder; cosine similarities are computed separately within each.
Passing ``features=dec_out`` to :func:`~torchjd.autojac.mtl_backward` causes both sub-networks
to receive Jacobians, which are then aggregated independently.

.. testcode::
    :emphasize-lines: 8-9, 15-16, 22-23

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

    encoder_gradvac = GradVac()
    decoder_gradvac = GradVac()

    for x, y1, y2 in zip(inputs, t1, t2):
        enc_out = encoder(x)
        dec_out = decoder(enc_out)
        mtl_backward([loss_fn(task1_head(dec_out), y1), loss_fn(task2_head(dec_out), y2)], features=dec_out)
        jac_to_grad(encoder.parameters(), encoder_gradvac)
        jac_to_grad(decoder.parameters(), decoder_gradvac)
        optimizer.step()
        optimizer.zero_grad()

3. All Layers
-------------

One :class:`~torchjd.aggregation.GradVac` instance per leaf module. Cosine similarities are
computed between the per-layer blocks of the task gradients.

.. testcode::
    :emphasize-lines: 14-15, 20-21

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

    leaf_layers = [m for m in encoder.modules() if not list(m.children()) and list(m.parameters())]
    gradvacs = [GradVac() for _ in leaf_layers]

    for x, y1, y2 in zip(inputs, t1, t2):
        features = encoder(x)
        mtl_backward([loss_fn(task1_head(features), y1), loss_fn(task2_head(features), y2)], features=features)
        for layer, gradvac in zip(leaf_layers, gradvacs):
            jac_to_grad(layer.parameters(), gradvac)
        optimizer.step()
        optimizer.zero_grad()

4. All Matrices
---------------

One :class:`~torchjd.aggregation.GradVac` instance per individual parameter tensor. Cosine
similarities are computed between the per-tensor blocks of the task gradients (e.g. weights and
biases of each layer are treated as separate groups).

.. testcode::
    :emphasize-lines: 14-15, 20-21

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
    gradvacs = [GradVac() for _ in shared_params]

    for x, y1, y2 in zip(inputs, t1, t2):
        features = encoder(x)
        mtl_backward([loss_fn(task1_head(features), y1), loss_fn(task2_head(features), y2)], features=features)
        for param, gradvac in zip(shared_params, gradvacs):
            jac_to_grad([param], gradvac)
        optimizer.step()
        optimizer.zero_grad()
