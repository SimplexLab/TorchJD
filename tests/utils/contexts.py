from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, TypeAlias

import torch
from settings import DEVICE

ExceptionContext: TypeAlias = AbstractContextManager[Exception | None]


@contextmanager
def fork_rng(seed: int = 0) -> Generator[Any, None, None]:
    devices = [] if DEVICE.type == "cpu" else [DEVICE]
    with torch.random.fork_rng(devices=devices, device_type=DEVICE.type) as ctx:
        torch.manual_seed(seed)
        yield ctx
