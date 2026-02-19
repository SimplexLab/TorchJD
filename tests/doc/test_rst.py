"""
This file contains the tests corresponding to the extra usage examples contained in the `.rst` files
of the documentation. When there are multiple examples within a single `.rst` file, we use nested
functions here to test them.
"""

from typing import no_type_check

from pytest import mark


def test_amp() -> None:
    import torch
    from torch.amp import GradScaler
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import UPGrad
    from torchjd.autojac import jac_to_grad, mtl_backward

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

    for input, target1, target2 in zip(inputs, task1_targets, task2_targets, strict=False):
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


def test_basic_usage() -> None:
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd import autojac
    from torchjd.aggregation import UPGrad
    from torchjd.autojac import jac_to_grad

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 2))
    optimizer = SGD(model.parameters(), lr=0.1)

    aggregator = UPGrad()
    input = torch.randn(16, 10)  # Batch of 16 random input vectors of length 10
    target1 = torch.randn(16)  # First batch of 16 targets
    target2 = torch.randn(16)  # Second batch of 16 targets

    loss_fn = MSELoss()
    output = model(input)
    loss1 = loss_fn(output[:, 0], target1)
    loss2 = loss_fn(output[:, 1], target2)

    autojac.backward([loss1, loss2])
    jac_to_grad(model.parameters(), aggregator)
    optimizer.step()
    optimizer.zero_grad()


def test_iwmtl() -> None:
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import Flattening, UPGradWeighting
    from torchjd.autogram import Engine

    shared_module = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
    task1_module = Linear(3, 1)
    task2_module = Linear(3, 1)
    params = [
        *shared_module.parameters(),
        *task1_module.parameters(),
        *task2_module.parameters(),
    ]

    optimizer = SGD(params, lr=0.1)
    mse = MSELoss(reduction="none")
    weighting = Flattening(UPGradWeighting())
    engine = Engine(shared_module, batch_dim=0)

    inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
    task1_targets = torch.randn(8, 16)  # 8 batches of 16 targets for the first task
    task2_targets = torch.randn(8, 16)  # 8 batches of 16 targets for the second task

    for input, target1, target2 in zip(inputs, task1_targets, task2_targets, strict=False):
        features = shared_module(input)  # shape: [16, 3]
        out1 = task1_module(features).squeeze(1)  # shape: [16]
        out2 = task2_module(features).squeeze(1)  # shape: [16]

        # Compute the matrix of losses: one loss per element of the batch and per task
        losses = torch.stack([mse(out1, target1), mse(out2, target2)], dim=1)  # shape: [16, 2]

        # Compute the gramian (inner products between pairs of gradients of the losses)
        gramian = engine.compute_gramian(losses)  # shape: [16, 2, 2, 16]

        # Obtain the weights that lead to no conflict between reweighted gradients
        weights = weighting(gramian)  # shape: [16, 2]

        # Do the standard backward pass, but weighted using the obtained weights
        losses.backward(weights)
        optimizer.step()
        optimizer.zero_grad()


def test_iwrm() -> None:
    def test_autograd() -> None:
        import torch
        from torch.nn import Linear, MSELoss, ReLU, Sequential
        from torch.optim import SGD

        X = torch.randn(8, 16, 10)
        Y = torch.randn(8, 16)

        model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
        loss_fn = MSELoss()

        params = model.parameters()
        optimizer = SGD(params, lr=0.1)

        for x, y in zip(X, Y, strict=False):
            y_hat = model(x).squeeze(dim=1)  # shape: [16]
            loss = loss_fn(y_hat, y)  # shape: [] (scalar)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_autojac() -> None:
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

        for x, y in zip(X, Y, strict=False):
            y_hat = model(x).squeeze(dim=1)  # shape: [16]
            losses = loss_fn(y_hat, y)  # shape: [16]
            backward(losses)
            jac_to_grad(model.parameters(), aggregator)
            optimizer.step()
            optimizer.zero_grad()

    def test_autogram() -> None:
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

        for x, y in zip(X, Y, strict=False):
            y_hat = model(x).squeeze(dim=1)  # shape: [16]
            losses = loss_fn(y_hat, y)  # shape: [16]
            gramian = engine.compute_gramian(losses)  # shape: [16, 16]
            weights = weighting(gramian)  # shape: [16]
            losses.backward(weights)
            optimizer.step()
            optimizer.zero_grad()

    test_autograd()
    test_autojac()
    test_autogram()


@mark.filterwarnings(
    "ignore::DeprecationWarning",
    "ignore::FutureWarning",
    "ignore::lightning.fabric.utilities.warnings.PossibleUserWarning",
)
@no_type_check  # Typing is annoying with Lightning, which would make the example too hard to read.
def test_lightning_integration() -> None:
    # Extra ----------------------------------------------------------------------------------------
    import logging

    logging.disable(logging.INFO)
    # ----------------------------------------------------------------------------------------------

    import torch
    from lightning import LightningModule, Trainer
    from lightning.pytorch.utilities.types import OptimizerLRScheduler
    from torch.nn import Linear, ReLU, Sequential
    from torch.nn.functional import mse_loss
    from torch.optim import Adam
    from torch.utils.data import DataLoader, TensorDataset

    from torchjd.aggregation import UPGrad
    from torchjd.autojac import jac_to_grad, mtl_backward

    class Model(LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.feature_extractor = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
            self.task1_head = Linear(3, 1)
            self.task2_head = Linear(3, 1)
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx) -> None:  # noqa: ANN001
            input, target1, target2 = batch

            features = self.feature_extractor(input)
            output1 = self.task1_head(features)
            output2 = self.task2_head(features)

            loss1 = mse_loss(output1, target1)
            loss2 = mse_loss(output2, target2)

            opt = self.optimizers()

            mtl_backward([loss1, loss2], features=features)
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


def test_monitoring() -> None:
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.nn.functional import cosine_similarity
    from torch.optim import SGD

    from torchjd.aggregation import UPGrad
    from torchjd.autojac import jac_to_grad, mtl_backward

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

    for input, target1, target2 in zip(inputs, task1_targets, task2_targets, strict=False):
        features = shared_module(input)
        output1 = task1_module(features)
        output2 = task2_module(features)
        loss1 = loss_fn(output1, target1)
        loss2 = loss_fn(output2, target2)

        mtl_backward([loss1, loss2], features=features)
        jac_to_grad(shared_module.parameters(), aggregator)
        optimizer.step()
        optimizer.zero_grad()


def test_mtl() -> None:
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import UPGrad
    from torchjd.autojac import jac_to_grad, mtl_backward

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

    for input, target1, target2 in zip(inputs, task1_targets, task2_targets, strict=False):
        features = shared_module(input)
        output1 = task1_module(features)
        output2 = task2_module(features)
        loss1 = loss_fn(output1, target1)
        loss2 = loss_fn(output2, target2)

        mtl_backward([loss1, loss2], features=features)
        jac_to_grad(shared_module.parameters(), aggregator)
        optimizer.step()
        optimizer.zero_grad()


def test_partial_jd() -> None:
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

    for x, y in zip(X, Y, strict=False):
        y_hat = model(x).squeeze(dim=1)  # shape: [16]
        losses = loss_fn(y_hat, y)  # shape: [16]
        gramian = engine.compute_gramian(losses)
        weights = weighting(gramian)
        losses.backward(weights)
        optimizer.step()
        optimizer.zero_grad()


def test_rnn() -> None:
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

    for input, target in zip(inputs, targets, strict=False):
        output, _ = rnn(input)  # output is of shape [5, 3, 20].
        losses = ((output - target) ** 2).mean(dim=[1, 2])  # 1 loss per sequence element.

        backward(losses, parallel_chunk_size=1)
        jac_to_grad(rnn.parameters(), aggregator)
        optimizer.step()
        optimizer.zero_grad()
