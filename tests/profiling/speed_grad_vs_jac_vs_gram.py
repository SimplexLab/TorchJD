import gc
import time
from collections.abc import Callable

import torch
from settings import DEVICE
from torch import Tensor
from utils.architectures import (
    AlexNet,
    Cifar10Model,
    FreeParam,
    GroupNormMobileNetV3Small,
    InstanceNormMobileNetV2,
    InstanceNormResNet18,
    ModuleFactory,
    NoFreeParam,
    SqueezeNet,
    WithTransformerLarge,
)
from utils.forward_backwards import (
    autograd_forward_backward,
    autograd_gramian_forward_backward,
    autogram_forward_backward,
    autojac_forward_backward,
    make_mse_loss_fn,
)
from utils.tensors import make_inputs_and_targets

from torchjd.aggregation import Mean
from torchjd.autogram import Engine

PARAMETRIZATIONS = [
    (ModuleFactory(WithTransformerLarge), 8),
    (ModuleFactory(FreeParam), 64),
    (ModuleFactory(NoFreeParam), 64),
    (ModuleFactory(Cifar10Model), 64),
    (ModuleFactory(AlexNet), 8),
    (ModuleFactory(InstanceNormResNet18), 16),
    (ModuleFactory(GroupNormMobileNetV3Small), 16),
    (ModuleFactory(SqueezeNet), 4),
    (ModuleFactory(InstanceNormMobileNetV2), 2),
]


def main() -> None:
    for factory, batch_size in PARAMETRIZATIONS:
        compare_autograd_autojac_and_autogram_speed(factory, batch_size)
        print("\n")


def compare_autograd_autojac_and_autogram_speed(factory: ModuleFactory, batch_size: int) -> None:
    model = factory()
    inputs, targets = make_inputs_and_targets(model, batch_size)
    loss_fn = make_mse_loss_fn(targets)

    A = Mean()
    W = A.weighting

    print(f"\nTimes for forward + backward on {factory} with BS={batch_size}, A={A} on {DEVICE}.")

    def fn_autograd() -> None:
        autograd_forward_backward(model, inputs, loss_fn)

    def init_fn_autograd() -> None:
        torch.cuda.empty_cache()
        gc.collect()
        fn_autograd()

    def fn_autograd_gramian() -> None:
        autograd_gramian_forward_backward(model, inputs, loss_fn, W)

    def init_fn_autograd_gramian() -> None:
        torch.cuda.empty_cache()
        gc.collect()
        fn_autograd_gramian()

    def fn_autojac() -> None:
        autojac_forward_backward(model, inputs, loss_fn, A)

    def init_fn_autojac() -> None:
        torch.cuda.empty_cache()
        gc.collect()
        fn_autojac()

    def fn_autogram() -> None:
        autogram_forward_backward(model, inputs, loss_fn, engine, W)

    def init_fn_autogram() -> None:
        torch.cuda.empty_cache()
        gc.collect()
        fn_autogram()

    def optionally_cuda_sync() -> None:
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

    def pre_fn() -> None:
        model.zero_grad()
        optionally_cuda_sync()

    def post_fn() -> None:
        optionally_cuda_sync()

    n_runs = 10
    autograd_times = time_call(fn_autograd, init_fn_autograd, pre_fn, post_fn, n_runs)
    print_times("autograd", autograd_times)

    autograd_gramian_times = time_call(
        fn_autograd_gramian,
        init_fn_autograd_gramian,
        pre_fn,
        post_fn,
        n_runs,
    )
    print_times("autograd gramian", autograd_gramian_times)

    autojac_times = time_call(fn_autojac, init_fn_autojac, pre_fn, post_fn, n_runs)
    print_times("autojac", autojac_times)

    engine = Engine(model, batch_dim=0)
    autogram_times = time_call(fn_autogram, init_fn_autogram, pre_fn, post_fn, n_runs)
    print_times("autogram", autogram_times)


def noop() -> None:
    pass


def time_call(
    fn: Callable[[], None],
    init_fn: Callable[[], None] = noop,
    pre_fn: Callable[[], None] = noop,
    post_fn: Callable[[], None] = noop,
    n_runs: int = 10,
) -> Tensor:
    init_fn()

    times = []
    for _ in range(n_runs):
        pre_fn()
        start = time.perf_counter()
        fn()
        post_fn()
        elapsed_time = time.perf_counter() - start
        times.append(elapsed_time)

    return torch.tensor(times)


def print_times(name: str, times: Tensor) -> None:
    print(f"{name} times (avg = {times.mean():.5f}, std = {times.std():.5f})")
    print(times)
    print()


if __name__ == "__main__":
    # To test this on cuda, add the following environment variables when running this:
    # CUBLAS_WORKSPACE_CONFIG=:4096:8;PYTEST_TORCH_DEVICE=cuda:0
    main()
