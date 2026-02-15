import os

import torch

_POSSIBLE_TEST_DEVICES = {"cpu", "cuda:0"}
_POSSIBLE_TEST_DTYPES = {"float32", "float64"}

try:
    _device_str = os.environ["PYTEST_TORCH_DEVICE"]
except KeyError:
    _device_str = "cpu"  # Default to cpu if environment variable not set

if _device_str not in _POSSIBLE_TEST_DEVICES:
    raise ValueError(
        f"Invalid value of environment variable PYTEST_TORCH_DEVICE: {_device_str}.\n"
        f"Possible values: {_POSSIBLE_TEST_DEVICES}.",
    )

if _device_str == "cuda:0" and not torch.cuda.is_available():
    raise ValueError('Requested device "cuda:0" but cuda is not available.')

DEVICE = torch.device(_device_str)

try:
    _dtype_str = os.environ["PYTEST_TORCH_DTYPE"]
except KeyError:
    _dtype_str = "float32"  # Default to float32 if environment variable not set

if _dtype_str not in _POSSIBLE_TEST_DTYPES:
    raise ValueError(
        f"Invalid value of environment variable PYTEST_TORCH_DTYPE: {_dtype_str}.\n"
        f"Possible values: {_POSSIBLE_TEST_DTYPES}.",
    )

DTYPE = getattr(torch, _dtype_str)  # "float32" => torch.float32
