import gc
from collections.abc import Callable

import torch
from settings import DEVICE
from torch.profiler import ProfilerActivity, profile
from utils.architectures import (
    AlexNet,
    Cifar10Model,
    GroupNormMobileNetV3Small,
    InstanceNormMobileNetV2,
    InstanceNormResNet18,
    ModuleFactory,
    SqueezeNet,
    WithTransformerLarge,
)
from utils.forward_backwards import (
    autogram_forward_backward,
    autojac_forward_backward,
    make_mse_loss_fn,
)
from utils.tensors import make_inputs_and_targets

from tests.paths import TRACES_DIR
from torchjd.aggregation import UPGrad, UPGradWeighting
from torchjd.autogram import Engine

PARAMETRIZATIONS = [
    (ModuleFactory(WithTransformerLarge), 4),
    (ModuleFactory(Cifar10Model), 64),
    (ModuleFactory(AlexNet), 4),
    (ModuleFactory(InstanceNormResNet18), 4),
    (ModuleFactory(GroupNormMobileNetV3Small), 8),
    (ModuleFactory(SqueezeNet), 4),
    (ModuleFactory(InstanceNormMobileNetV2), 2),
]


def profile_method(
    method_name: str,
    forward_backward_fn: Callable,
    factory: ModuleFactory,
    batch_size: int,
) -> None:
    """
    Profiles memory and computation time of a forward and backward pass.

    :param method_name: Name of the method being profiled (for output paths)
    :param forward_backward_fn: Function to execute the forward and backward pass.
    :param factory: A ModuleFactory that creates the model to profile.
    :param batch_size: The batch size to use for profiling.
    """
    print(f"{method_name}: {factory} with batch_size={batch_size} on {DEVICE}:")

    _clear_unused_memory()
    model = factory()
    inputs, targets = make_inputs_and_targets(model, batch_size)
    loss_fn = make_mse_loss_fn(targets)

    activities = _get_profiler_activities()

    # Warmup run
    forward_backward_fn(model, inputs, loss_fn)
    model.zero_grad()
    _clear_unused_memory()

    # Profiled run
    with profile(
        activities=activities,
        profile_memory=True,
        record_shapes=False,  # Otherwise some tensors may be referenced longer than normal
        with_stack=True,
    ) as prof:
        forward_backward_fn(model, inputs, loss_fn)

    _save_and_print_trace(prof, method_name, factory, batch_size)


def _clear_unused_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_profiler_activities() -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if DEVICE.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def _save_and_print_trace(
    prof: profile,
    method_name: str,
    factory: ModuleFactory,
    batch_size: int,
) -> None:
    filename = f"{factory}-bs{batch_size}-{DEVICE.type}.json"
    output_dir = TRACES_DIR / method_name
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / filename

    prof.export_chrome_trace(str(trace_path))
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))


def profile_autojac(factory: ModuleFactory, batch_size: int) -> None:
    def forward_backward_fn(model, inputs, loss_fn) -> None:
        aggregator = UPGrad()
        autojac_forward_backward(model, inputs, loss_fn, aggregator)

    profile_method("autojac", forward_backward_fn, factory, batch_size)


def profile_autogram(factory: ModuleFactory, batch_size: int) -> None:
    def forward_backward_fn(model, inputs, loss_fn) -> None:
        engine = Engine(model, batch_dim=0)
        weighting = UPGradWeighting()
        autogram_forward_backward(model, inputs, loss_fn, engine, weighting)

    profile_method("autogram", forward_backward_fn, factory, batch_size)


def main() -> None:
    for factory, batch_size in PARAMETRIZATIONS:
        profile_autojac(factory, batch_size)
        print("\n" + "=" * 80 + "\n")
        profile_autogram(factory, batch_size)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # To test this on cuda, add the following environment variables when running this:
    # CUBLAS_WORKSPACE_CONFIG=:4096:8;PYTEST_TORCH_DEVICE=cuda:0
    main()
