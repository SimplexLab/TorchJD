from copy import deepcopy

import torch
from pytest import mark, raises
from settings import DEVICE, DTYPE
from torch import Tensor, nn
from torch.optim import SGD, Adam
from torch.testing import assert_close
from utils.tensors import eye_, ones_, randn_, tensor_, zeros_

from torchjd.aggregation import GradNorm, GradNormWeighting
from torchjd.autogram import Engine
from torchjd.autojac import jac

from ._asserts import assert_expected_structure, assert_non_differentiable
from ._inputs import scaled_matrices, typical_matrices


def test_representations() -> None:
    assert repr(GradNormWeighting(3)) == "GradNormWeighting(n_tasks=3, alpha=1.5)"
    assert repr(GradNorm(3, alpha=0.0)) == "GradNorm(n_tasks=3, alpha=0.0)"
    assert str(GradNorm(3)) == "GradNorm"


@mark.parametrize("matrix", typical_matrices + scaled_matrices)
def test_expected_structure(matrix: Tensor) -> None:
    aggregator = GradNorm(matrix.shape[0]).to(device=DEVICE, dtype=DTYPE)
    aggregator.set_losses(ones_(matrix.shape[0]))
    assert_expected_structure(aggregator, matrix)


def test_initial_weights_and_detached_target() -> None:
    weighting = GradNormWeighting(3).to(device=DEVICE, dtype=DTYPE)
    losses = tensor_([2.0, 4.0, 8.0], requires_grad=True)
    gramian = torch.diag(tensor_([1.0, 4.0, 25.0])).requires_grad_()
    weighting.set_losses(losses)
    assert_close(weighting(gramian), ones_(3))
    balancing_loss = weighting.balancing_loss()
    assert_close(balancing_loss, tensor_(14 / 3))
    balancing_loss.backward()
    assert_close(weighting.weights.grad, tensor_([-1.0, -2.0, 5.0]))
    assert losses.grad is None
    assert gramian.grad is None


@mark.parametrize("alpha", [0.0, 1.0, 1.5])
def test_training_rates_use_initial_losses(alpha: float) -> None:
    weighting = GradNormWeighting(3, alpha=alpha).to(device=DEVICE, dtype=DTYPE)
    initial = tensor_([2.0, 4.0, 8.0])
    weighting.set_losses(initial)
    initial.mul_(10)
    weighting.set_losses(tensor_([2.0, 2.0, 2.0]))
    weighting(torch.diag(tensor_([1.0, 4.0, 25.0])))
    targets = (8 / 3) * tensor_([12 / 7, 6 / 7, 3 / 7]).pow(alpha)
    expected = (tensor_([1.0, 2.0, 5.0]) - targets).abs().sum()
    assert_close(weighting.balancing_loss(), expected)


def test_two_sgd_steps_match_algorithm_one() -> None:
    weighting = GradNormWeighting(3, alpha=0.0).to(device=DEVICE, dtype=DTYPE)
    parameter = weighting.weights
    optimizer = SGD(weighting.parameters(), lr=0.1)
    diagonals = [tensor_([1.0, 4.0, 25.0]), tensor_([4.0, 1.0, 16.0])]
    expected = [tensor_([33 / 28, 9 / 7, 15 / 28]), tensor_([411 / 350, 291 / 175, 57 / 350])]
    for diagonal, weights in zip(diagonals, expected, strict=True):
        optimizer.zero_grad()
        weighting.set_losses(ones_(3))
        before = weighting(torch.diag(diagonal))
        weighting.balancing_loss().backward()
        optimizer.step()
        weighting.renormalize()
        assert_close(weighting.weights, weights)
        assert_close(weighting.weights.sum(), tensor_(3.0))
        assert (weighting.weights >= 0).all()
        assert not before.requires_grad
        assert not torch.allclose(before, weighting.weights)
        assert weighting.weights is parameter


def test_auxiliary_backward_does_not_change_model_gradients() -> None:
    parameter = nn.Parameter(tensor_([1.0, 2.0]))
    losses = torch.stack([parameter.square().sum(), 3 * (parameter - 1).square().sum()])
    weighting = GradNormWeighting(2).to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        weighting.weights.copy_(tensor_([0.5, 1.5]))
    J = torch.stack([torch.autograd.grad(loss, parameter, retain_graph=True)[0] for loss in losses])
    weighting.set_losses(losses)
    weights = weighting(J @ J.T)
    losses.backward(weights)
    expected_grad = tensor_([1.0, 11.0])
    assert_close(parameter.grad, expected_grad)
    assert weighting.weights.grad is None
    weighting.balancing_loss().backward()
    assert_close(parameter.grad, expected_grad)
    assert weighting.weights.grad is not None


def test_last_shared_layer_matches_direct_autograd() -> None:
    shared = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).to(DEVICE, DTYPE)
    heads = nn.ModuleList([nn.Linear(2, 1), nn.Linear(2, 1)]).to(DEVICE, DTYPE)
    weighting = GradNormWeighting(2).to(DEVICE, DTYPE)
    optimizer = Adam(weighting.parameters(), lr=0.001)
    reference_weights = nn.Parameter(ones_(2))
    reference_optimizer = Adam([reference_weights], lr=0.001)
    parameters = [*shared.parameters(), *heads.parameters()]
    engine = Engine(shared[2], batch_dim=None)
    initial_losses = None
    for _ in range(3):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        for parameter in parameters:
            parameter.grad = None
        representation = shared(randn_(4, 3))
        losses = torch.stack([head(representation).square().mean() for head in heads])
        if initial_losses is None:
            initial_losses = losses.detach().clone()
        direct_norms = []
        for weight, loss in zip(reference_weights, losses, strict=True):
            gradients = torch.autograd.grad(
                weight * loss, list(shared[2].parameters()), retain_graph=True, create_graph=True
            )
            direct_norms.append(torch.cat([g.flatten() for g in gradients]).norm())
        norms = torch.stack(direct_norms)
        ratios = losses.detach() / initial_losses
        targets = (norms.mean() * (ratios / ratios.mean()).pow(1.5)).detach()
        reference_loss = (norms - targets).abs().sum()
        reference_gradient = torch.autograd.grad(reference_loss, reference_weights)[0]
        expected_model = torch.autograd.grad(
            losses, parameters, grad_outputs=reference_weights.detach(), retain_graph=True
        )
        weighting.set_losses(losses)
        weights = weighting(engine.compute_gramian(losses))
        assert_close(weighting.balancing_loss(), reference_loss)
        weighting.balancing_loss().backward()
        assert_close(weighting.weights.grad, reference_gradient)
        assert all(parameter.grad is None for parameter in parameters)
        losses.backward(weights)
        for parameter, expected in zip(parameters, expected_model, strict=True):
            assert_close(parameter.grad, expected)
        optimizer.step()
        weighting.renormalize()
        reference_weights.grad = reference_gradient
        reference_optimizer.step()
        with torch.no_grad():
            reference_weights.mul_(2 / reference_weights.sum())
        assert_close(weighting.weights, reference_weights)


def test_aggregator_matches_weighting_and_updates() -> None:
    aggregator = GradNorm(3).to(DEVICE, DTYPE)
    weighting = GradNormWeighting(3).to(DEVICE, DTYPE)
    optimizers = [SGD(module.parameters(), lr=0.01) for module in (aggregator, weighting)]
    for _ in range(2):
        J = randn_(3, 4)
        losses = ones_(3)
        aggregator.set_losses(losses)
        weighting.set_losses(losses)
        assert_close(aggregator(J), weighting(J @ J.T) @ J)
        assert_close(aggregator.balancing_loss(), weighting.balancing_loss())
        for module, optimizer in zip((aggregator, weighting), optimizers, strict=True):
            optimizer.zero_grad()
            module.balancing_loss().backward()
            optimizer.step()
            module.renormalize()
    aggregator.reset()
    aggregator.set_losses(losses)
    assert_close(aggregator(J), J.sum(dim=0))


def test_non_differentiable() -> None:
    aggregator = GradNorm(3).to(DEVICE, DTYPE)
    aggregator.set_losses(ones_(3))
    assert_non_differentiable(aggregator, ones_(3, 5, requires_grad=True))


@mark.parametrize("n_columns", [0, 4])
def test_zero_gradients(n_columns: int) -> None:
    aggregator = GradNorm(2).to(DEVICE, DTYPE)
    aggregator.set_losses(ones_(2))
    assert_close(aggregator(zeros_(2, n_columns)), zeros_(n_columns))
    loss = aggregator.balancing_loss()
    assert_close(loss, tensor_(0.0))
    loss.backward()
    assert_close(aggregator.gramian_weighting.weights.grad, zeros_(2))


def test_single_task() -> None:
    weighting = GradNormWeighting(1).to(DEVICE, DTYPE)
    weighting.set_losses(ones_(1))
    assert_close(weighting(eye_(1)), ones_(1))
    weighting.balancing_loss().backward()
    assert_close(weighting.weights.grad, zeros_(1))


@mark.parametrize(
    "losses, expected_gradient", [([0.0, 1.0], [1.0, -1.0]), ([0.0, 0.0], [0.0, 0.0])]
)
def test_zero_current_losses(losses: list[float], expected_gradient: list[float]) -> None:
    weighting = GradNormWeighting(2, alpha=1.0).to(DEVICE, DTYPE)
    weighting.set_losses(ones_(2))
    weighting.set_losses(tensor_(losses))
    weighting(eye_(2))
    weighting.balancing_loss().backward()
    assert_close(weighting.weights.grad, tensor_(expected_gradient))


def test_checkpoint_restores_weights_and_baseline() -> None:
    weighting = GradNormWeighting(2).to(DEVICE, DTYPE)
    optimizer = Adam(weighting.parameters(), lr=0.01)
    weighting.set_losses(tensor_([2.0, 3.0]))
    weighting(torch.diag(tensor_([1.0, 9.0])))
    weighting.balancing_loss().backward()
    optimizer.step()
    weighting.renormalize()
    restored = GradNormWeighting(2).to(DEVICE, DTYPE)
    restored.load_state_dict(deepcopy(weighting.state_dict()))
    restored_optimizer = Adam(restored.parameters(), lr=0.01)
    restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    with raises(ValueError, match="set_losses"):
        restored(eye_(2))
    for module, opt in ((weighting, optimizer), (restored, restored_optimizer)):
        opt.zero_grad()
        module.set_losses(tensor_([1.0, 1.0]))
        module(torch.diag(tensor_([4.0, 1.0])))
        module.balancing_loss().backward()
        opt.step()
        module.renormalize()
    assert_close(weighting.weights, restored.weights)


def test_reset_keeps_parameter_and_replaces_baseline() -> None:
    weighting = GradNormWeighting(2).to(DEVICE, DTYPE)
    parameter = weighting.weights
    optimizer = SGD(weighting.parameters(), lr=0.1)
    weighting.set_losses(tensor_([2.0, 3.0]))
    weighting(torch.diag(tensor_([1.0, 9.0])))
    weighting.balancing_loss().backward()
    optimizer.step()
    weighting.reset()
    assert weighting.weights is parameter
    assert weighting.weights.grad is None
    assert optimizer.param_groups[0]["params"][0] is parameter
    fresh = GradNormWeighting(2).to(DEVICE, DTYPE)
    for module in (weighting, fresh):
        module.set_losses(tensor_([5.0, 1.0]))
        module(torch.diag(tensor_([1.0, 9.0])))
    assert_close(weighting.weights, fresh.weights)
    assert_close(weighting.balancing_loss(), fresh.balancing_loss())


def test_to_moves_baseline_and_batch_statistics() -> None:
    weighting = GradNormWeighting(2).to(device=DEVICE, dtype=torch.float32)
    weighting.set_losses(tensor_([2.0, 3.0]).float())
    weighting(eye_(2).float())
    weighting = weighting.double()
    assert weighting.balancing_loss().dtype == torch.float64
    weighting.set_losses(tensor_([1.0, 1.0]).double())
    assert weighting(eye_(2).double()).dtype == torch.float64


@mark.parametrize("cls", [GradNorm, GradNormWeighting])
@mark.parametrize("n_tasks", [0, -1])
def test_invalid_task_count(cls: type[GradNorm | GradNormWeighting], n_tasks: int) -> None:
    with raises(ValueError, match="n_tasks"):
        cls(n_tasks)


@mark.parametrize("cls", [GradNorm, GradNormWeighting])
@mark.parametrize("alpha", [-1.0, float("nan"), float("inf")])
def test_alpha_validation(cls: type[GradNorm | GradNormWeighting], alpha: float) -> None:
    with raises(ValueError, match="alpha"):
        cls(2, alpha=alpha)
    module = cls(2)
    module.alpha = 0.5
    assert module.alpha == 0.5
    with raises(ValueError, match="alpha"):
        module.alpha = alpha


@mark.parametrize("losses", [[0.0, 1.0], [-1.0, 1.0], [float("nan"), 1.0], [float("inf"), 1.0]])
def test_invalid_initial_losses(losses: list[float]) -> None:
    weighting = GradNormWeighting(2).to(DEVICE, DTYPE)
    with raises(ValueError, match="losses"):
        weighting.set_losses(tensor_(losses))


def test_shape_and_call_order_validation() -> None:
    weighting = GradNormWeighting(2).to(DEVICE, DTYPE)
    with raises(ValueError, match="set_losses"):
        weighting(eye_(2))
    with raises(ValueError, match="forward"):
        weighting.balancing_loss()
    with raises(ValueError, match="shape"):
        weighting.set_losses(ones_(3))
    weighting.set_losses(ones_(2))
    with raises(ValueError, match="shape"):
        weighting(eye_(3))
    weighting(eye_(2))
    weighting.set_losses(ones_(2))
    with raises(ValueError, match="forward"):
        weighting.balancing_loss()


def test_dtype_validation() -> None:
    weighting = GradNormWeighting(2).to(device=DEVICE, dtype=torch.float64)
    with raises(ValueError, match="dtype"):
        weighting.set_losses(ones_(2).float())
    weighting.set_losses(ones_(2).double())
    with raises(ValueError, match="dtype"):
        weighting(eye_(2).float())


@mark.parametrize("weights", [[0.0, 0.0], [-1.0, 2.0], [float("nan"), 1.0], [float("inf"), 1.0]])
def test_invalid_weight_updates(weights: list[float]) -> None:
    weighting = GradNormWeighting(2).to(DEVICE, DTYPE)
    with torch.no_grad():
        weighting.weights.copy_(tensor_(weights))
    with raises(ValueError, match="Weights"):
        weighting.renormalize()


def test_fresh_checkpoint_can_initialize() -> None:
    weighting = GradNormWeighting(2).to(DEVICE, DTYPE)
    weighting.load_state_dict(GradNormWeighting(2).state_dict())
    weighting.set_losses(ones_(2))
    assert_close(weighting(eye_(2)), ones_(2))


def test_autojac_usage() -> None:
    parameter = nn.Parameter(tensor_([1.0, 2.0]))
    losses = parameter.square()
    J = jac(losses, [parameter], retain_graph=True)[0]
    weighting = GradNormWeighting(2).to(DEVICE, DTYPE)
    weighting.set_losses(losses)
    losses.backward(weighting(J @ J.T))
    assert_close(parameter.grad, tensor_([2.0, 4.0]))
