Instance-Wise Risk Minimization (IWRM)
======================================

This example shows how to use TorchJD to minimize the vector of per-instance losses. This learning
paradigm, called IWRM, is multi-objective, as opposed to the usual empirical risk minimization
(ERM), which seeks to minimize the average loss. While a step of ERM may increase the loss of some
samples of the batch, a step of IWRM using :doc:`UPGrad <../docs/aggregation/upgrad>` guarantees
that no loss from the batch is increased (given a sufficiently small learning rate).

.. hint::
    A proper definition of IWRM and its empirical results on some deep learning tasks are
    available in `Jacobian Descent For Multi-Objective Optimization
    <https://arxiv.org/pdf/2406.16232>`_.

TorchJD offers two methods to perform IWRM. The :doc:`autojac <../docs/autojac/index>` engine
backpropagates the Jacobian of each sample's loss. It uses an
:doc:`Aggregator <../docs/aggregation/index>` to combine the rows of this Jacobian to fill the
``.grad`` fields of the model's parameters. Because it has to store the full Jacobian, this approach
uses a lot of memory.

The recommended approach, called the :doc:`autogram engine <../docs/autogram/engine>`, works by
backpropagating the Gramian of the Jacobian of each sample's loss with respect to the model's
parameters. This method is more memory-efficient and generally much faster because it avoids
storing the full Jacobians. A vector of weights is then computed by applying a
:doc:`Weighting <../docs/aggregation/index>` to the obtained Gramian, and a normal step of gradient
descent is then done on the weighted sum of the losses.

Both approaches (autojac and autogram) are mathematically equivalent, and should thus give the same
results up to small numerical differences. Even though the autogram engine is generally much faster
than the autojac engine, there are some layers that are incompatible with it. These limitations are
documented :doc:`here <../docs/autogram/engine>`.

For the sake of the example, we generate a fake dataset consisting of 8 batches of 16 random input
vectors of dimension 10, and their corresponding scalar labels. We train a very simple regression
model to retrieve the label from the corresponding input. To minimize the average loss (ERM), we use
stochastic gradient descent (SGD), where each gradient is computed from the average loss over a
batch of data. When minimizing per-instance losses (IWRM), we use either autojac, with
:doc:`UPGrad <../docs/aggregation/upgrad>` to aggregate the Jacobian, or autogram, with
:doc:`UPGradWeighting <../docs/aggregation/upgrad>` to extract weights from the Gramian.

.. tab-set::
    .. tab-item:: autograd (baseline)

        .. code-block:: python

            import torch
            from torch.nn import Linear, MSELoss, ReLU, Sequential
            from torch.optim import SGD




            X = torch.randn(8, 16, 10)
            Y = torch.randn(8, 16)

            model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
            loss_fn = MSELoss()

            params = model.parameters()
            optimizer = SGD(params, lr=0.1)



            for x, y in zip(X, Y):
                y_hat = model(x).squeeze(dim=1)  # shape: [16]
                loss = loss_fn(y_hat, y)  # shape: [] (scalar)
                loss.backward()


                optimizer.step()
                optimizer.zero_grad()

        In this baseline example, the update may negatively affect the loss of some elements of the
        batch.

    .. tab-item:: autojac

        .. code-block:: python
            :emphasize-lines: 5-6, 12, 16, 21-23

            import torch
            from torch.nn import Linear, MSELoss, ReLU, Sequential
            from torch.optim import SGD

            from torchjd.aggregation import UPGrad
            from torchjd.autojac import backward, jac_to_grad

            X = torch.randn(8, 16, 10)
            Y = torch.randn(8, 16)

            model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
            loss_fn = MSELoss(reduction="none")

            params = model.parameters()
            optimizer = SGD(params, lr=0.1)
            aggregator = UPGrad()


            for x, y in zip(X, Y):
                y_hat = model(x).squeeze(dim=1)  # shape: [16]
                losses = loss_fn(y_hat, y)  # shape: [16]
                backward(losses)
                jac_to_grad(model.parameters(), aggregator)

                optimizer.step()
                optimizer.zero_grad()

        Here, we compute the Jacobian of the per-sample losses with respect to the model parameters
        and use it to update the model such that no loss from the batch is (locally) increased.

    .. tab-item:: autogram (recommended)

        .. code-block:: python
            :emphasize-lines: 5-6, 12, 16-17, 21-24

            import torch
            from torch.nn import Linear, MSELoss, ReLU, Sequential
            from torch.optim import SGD

            from torchjd.aggregation import UPGradWeighting
            from torchjd.autogram import Engine

            X = torch.randn(8, 16, 10)
            Y = torch.randn(8, 16)

            model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
            loss_fn = MSELoss(reduction="none")

            params = model.parameters()
            optimizer = SGD(params, lr=0.1)
            weighting = UPGradWeighting()
            engine = Engine(model, batch_dim=0)

            for x, y in zip(X, Y):
                y_hat = model(x).squeeze(dim=1)  # shape: [16]
                losses = loss_fn(y_hat, y)  # shape: [16]
                gramian = engine.compute_gramian(losses)  # shape: [16, 16]
                weights = weighting(gramian)  # shape: [16]
                losses.backward(weights)
                optimizer.step()
                optimizer.zero_grad()

        Here, the per-sample gradients are never fully stored in memory, leading to large
        improvements in memory usage and speed compared to autojac, in most practical cases. The
        results should be the same as with autojac (up to tiny numerical imprecisions), as long as
        the model always treats each instance independently from other instances in the batch (e.g.
        no batch-normalization is used).

Note that in all three cases, we use the `torch.optim.SGD
<https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_ optimizer to update the
parameters of the model in the opposite direction of their ``.grad`` field. The difference comes
from how this field is computed.
