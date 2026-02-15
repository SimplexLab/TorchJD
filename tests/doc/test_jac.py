"""
This file contains the test of the jac usage example, with a verification of the value of the obtained jacobians tuple.
"""

from torch.testing import assert_close


def test_jac():
    import torch

    from torchjd.autojac import jac

    param = torch.tensor([1.0, 2.0], requires_grad=True)
    # Compute arbitrary quantities that are function of param
    y1 = torch.tensor([-1.0, 1.0]) @ param
    y2 = (param**2).sum()
    jacobians = jac([y1, y2], [param])

    assert len(jacobians) == 1
    assert_close(jacobians[0], torch.tensor([[-1.0, 1.0], [2.0, 4.0]]), rtol=0.0, atol=1e-04)


def test_jac_2():
    import torch

    from torchjd.autojac import jac

    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # shape: [2, 2]
    bias = torch.tensor([0.5, -0.5], requires_grad=True)  # shape: [2]
    # Compute arbitrary quantities that are function of weight and bias
    input_vec = torch.tensor([1.0, -1.0])
    y1 = weight @ input_vec + bias  # shape: [2]
    y2 = (weight**2).sum() + (bias**2).sum()  # shape: [] (scalar)
    jacobians = jac([y1, y2], [weight, bias])  # shapes: [3, 2, 2], [3, 2]
    jacobian_matrices = tuple(J.flatten(1) for J in jacobians)  # shapes: [3, 4], [3, 2]
    combined_jacobian_matrix = torch.concat(jacobian_matrices, dim=1)  # shape: [3, 6]
    gramian = combined_jacobian_matrix @ combined_jacobian_matrix.T  # shape: [3, 3]

    assert_close(
        gramian,
        torch.tensor([[3.0, 0.0, -1.0], [0.0, 3.0, -3.0], [-1.0, -3.0, 122.0]]),
        rtol=0.0,
        atol=1e-04,
    )
