"""
This file contains the test of the backward usage example, with a verification of the value of the
obtained `.jac` field.
"""

from utils.asserts import assert_jac_close


def test_backward() -> None:
    import torch

    from torchjd.autojac import backward

    param = torch.tensor([1.0, 2.0], requires_grad=True)
    # Compute arbitrary quantities that are function of param
    y1 = torch.tensor([-1.0, 1.0]) @ param
    y2 = (param**2).sum()

    backward([y1, y2])

    assert_jac_close(param, torch.tensor([[-1.0, 1.0], [2.0, 4.0]]), rtol=0.0, atol=1e-04)


def test_backward2() -> None:
    import torch

    from torchjd.autojac import backward

    param = torch.tensor([1.0, 2.0], requires_grad=True)
    # Compute arbitrary quantities that are function of param
    y1 = torch.tensor([-1.0, 1.0]) @ param
    y2 = (param**2).sum()

    J1 = torch.tensor([1.0, 0.0])
    J2 = torch.tensor([0.0, 1.0])

    backward([y1, y2], jac_tensors=[J1, J2])

    assert_jac_close(param, torch.tensor([[-1.0, 1.0], [2.0, 4.0]]), rtol=0.0, atol=1e-04)
