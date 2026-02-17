PyTorch Lightning Integration
=============================

To use Jacobian descent with TorchJD in a :class:`~lightning.pytorch.core.LightningModule`, you need
to turn off automatic optimization by setting ``automatic_optimization`` to ``False`` and to
customize the ``training_step`` method to make it call the appropriate TorchJD method
(:doc:`backward <../docs/autojac/backward>` or :doc:`mtl_backward <../docs/autojac/mtl_backward>`).

The following code example demonstrates a basic multi-task learning setup using a
:class:`~lightning.pytorch.core.LightningModule` that will call :doc:`mtl_backward
<../docs/autojac/mtl_backward>` at each training iteration.

.. code-block:: python
    :emphasize-lines: 9-10, 18, 31-32

    import torch
    from lightning import LightningModule, Trainer
    from lightning.pytorch.utilities.types import OptimizerLRScheduler
    from torch.nn import Linear, ReLU, Sequential
    from torch.nn.functional import mse_loss
    from torch.optim import Adam
    from torch.utils.data import DataLoader, TensorDataset

    from torchjd.aggregation import UPGrad
    from torchjd.autojac import mtl_backward, jac_to_grad

    class Model(LightningModule):
        def __init__(self):
            super().__init__()
            self.feature_extractor = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
            self.task1_head = Linear(3, 1)
            self.task2_head = Linear(3, 1)
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx) -> None:
            input, target1, target2 = batch

            features = self.feature_extractor(input)
            output1 = self.task1_head(features)
            output2 = self.task2_head(features)

            loss1 = mse_loss(output1, target1)
            loss2 = mse_loss(output2, target2)

            opt = self.optimizers()
            mtl_backward(tensors=[loss1, loss2], features=features)
            jac_to_grad(self.feature_extractor.parameters(), UPGrad())
            opt.step()
            opt.zero_grad()

        def configure_optimizers(self) -> OptimizerLRScheduler:
            optimizer = Adam(self.parameters(), lr=1e-3)
            return optimizer

    model = Model()

    inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
    task1_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the first task
    task2_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the second task

    dataset = TensorDataset(inputs, task1_targets, task2_targets)
    train_loader = DataLoader(dataset)
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )

    trainer.fit(model=model, train_dataloaders=train_loader)

.. warning::
    This will not handle automatic scaling in low-precision settings. There is currently no easy
    fix.

.. warning::
    TorchJD is incompatible with compiled models, so you must ensure that your model is not
    compiled.
