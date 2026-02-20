"""
This file contains the test of the jac_to_grad usage example, with a verification of the value of
the obtained `.grad` field.
"""

from utils.asserts import assert_grad_close


def test_jac_to_grad() -> None:
    import torch

    from torchjd.aggregation import UPGrad
    from torchjd.autojac import backward, jac_to_grad

    param = torch.tensor([1.0, 2.0], requires_grad=True)
    # Compute arbitrary quantities that are function of param
    y1 = torch.tensor([-1.0, 1.0]) @ param
    y2 = (param**2).sum()
    backward([y1, y2])  # param now has a .jac field
    jac_to_grad([param], aggregator=UPGrad())  # param now has a .grad field

    assert_grad_close(param, torch.tensor([0.5000, 2.5000]), rtol=0.0, atol=1e-04)
