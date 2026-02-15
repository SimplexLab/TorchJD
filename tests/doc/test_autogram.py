"""This file contains tests for the usage examples related to autogram."""


def test_engine():
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import UPGradWeighting
    from torchjd.autogram import Engine

    # Generate data (8 batches of 16 examples of dim 5) for the sake of the example
    inputs = torch.randn(8, 16, 5)
    targets = torch.randn(8, 16)

    model = Sequential(Linear(5, 4), ReLU(), Linear(4, 1))
    optimizer = SGD(model.parameters())

    criterion = MSELoss(reduction="none")  # Important to use reduction="none"
    weighting = UPGradWeighting()

    # Create the engine before the backward pass, and only once.
    engine = Engine(model, batch_dim=0)

    for input, target in zip(inputs, targets, strict=True):
        output = model(input).squeeze(dim=1)  # shape: [16]
        losses = criterion(output, target)  # shape: [16]

        gramian = engine.compute_gramian(losses)  # shape: [16, 16]
        weights = weighting(gramian)  # shape: [16]
        losses.backward(weights)
        optimizer.step()
        optimizer.zero_grad()
