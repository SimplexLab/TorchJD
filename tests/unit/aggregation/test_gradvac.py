import torch
import torch.nn as nn
from pytest import mark, raises
from torch import Tensor, tensor
from torch.testing import assert_close
from utils.tensors import ones_, randn_

from torchjd.aggregation import DEFAULT_GRADVAC_EPS, GradVac

from ._asserts import assert_expected_structure, assert_non_differentiable
from ._inputs import scaled_matrices, typical_matrices

scaled_pairs = [(GradVac(), m) for m in scaled_matrices]
typical_pairs = [(GradVac(), m) for m in typical_matrices]
requires_grad_pairs = [(GradVac(), ones_(3, 5, requires_grad=True))]


def test_repr() -> None:
    g = GradVac()
    assert repr(g).startswith("GradVac(")
    assert "beta=" in repr(g)
    assert "group_type=" in repr(g)
    assert "encoder=" in repr(g)
    assert "eps=" in repr(g)


def test_beta_out_of_range() -> None:
    with raises(ValueError, match="beta"):
        GradVac(beta=-0.1)
    with raises(ValueError, match="beta"):
        GradVac(beta=1.1)


def test_eps_non_positive() -> None:
    with raises(ValueError, match="eps"):
        GradVac(eps=0.0)
    with raises(ValueError, match="eps"):
        GradVac(eps=-1e-9)


def test_eps_setter_rejects_non_positive() -> None:
    g = GradVac()
    with raises(ValueError, match="eps"):
        g.eps = 0.0


def test_default_eps_constant() -> None:
    assert DEFAULT_GRADVAC_EPS == 1e-8
    assert GradVac().eps == DEFAULT_GRADVAC_EPS


def test_eps_can_be_changed_between_steps() -> None:
    j = tensor([[1.0, 0.0], [0.0, 1.0]])
    agg = GradVac()
    agg.eps = 1e-6
    assert agg(j).isfinite().all()
    agg.reset()
    agg.eps = 1e-10
    assert agg(j).isfinite().all()


def test_group_type_invalid() -> None:
    with raises(ValueError, match="group_type"):
        GradVac(group_type=3)


def test_group_type_0_rejects_encoder() -> None:
    net = nn.Linear(1, 1)
    with raises(ValueError, match="encoder"):
        GradVac(encoder=net)


def test_group_type_0_rejects_shared_params() -> None:
    p = nn.Parameter(tensor([1.0]))
    with raises(ValueError, match="shared_params"):
        GradVac(shared_params=[p])


def test_group_type_1_requires_encoder() -> None:
    with raises(ValueError, match="encoder"):
        GradVac(group_type=1)


def test_group_type_1_rejects_shared_params() -> None:
    net = nn.Linear(1, 1)
    p = nn.Parameter(tensor([1.0]))
    with raises(ValueError, match="shared_params"):
        GradVac(group_type=1, encoder=net, shared_params=[p])


def test_group_type_2_requires_shared_params() -> None:
    with raises(ValueError, match="shared_params"):
        GradVac(group_type=2)


def test_group_type_2_rejects_encoder() -> None:
    net = nn.Linear(1, 1)
    with raises(ValueError, match="encoder"):
        GradVac(group_type=2, encoder=net, shared_params=list(net.parameters()))


def test_encoder_without_leaf_parameters() -> None:
    class Empty(nn.Module):
        pass

    with raises(ValueError, match="encoder"):
        GradVac(group_type=1, encoder=Empty())


def test_shared_params_empty() -> None:
    with raises(ValueError, match="shared_params"):
        GradVac(group_type=2, shared_params=())


def test_group_type_1_forward() -> None:
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    d = sum(p.numel() for p in net.parameters())
    j = randn_((2, d))
    torch.manual_seed(0)
    out = GradVac(group_type=1, encoder=net)(j)
    assert out.shape == (d,)
    assert out.isfinite().all()


def test_group_type_2_forward() -> None:
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    params = list(net.parameters())
    d = sum(p.numel() for p in params)
    j = randn_((2, d))
    torch.manual_seed(0)
    out = GradVac(group_type=2, shared_params=params)(j)
    assert out.shape == (d,)
    assert out.isfinite().all()


def test_jacobian_width_mismatch() -> None:
    net = nn.Linear(2, 2)
    d = sum(p.numel() for p in net.parameters())
    agg = GradVac(group_type=1, encoder=net)
    with raises(ValueError, match="Jacobian width"):
        agg(tensor([[1.0] * (d - 1), [2.0] * (d - 1)]))


def test_zero_rows_returns_zero_vector() -> None:
    out = GradVac()(tensor([]).reshape(0, 3))
    assert_close(out, tensor([0.0, 0.0, 0.0]))


def test_zero_columns_returns_zero_vector() -> None:
    """Handled inside forward before grouping validation."""

    out = GradVac()(tensor([]).reshape(2, 0))
    assert out.shape == (0,)


def test_reproducible_with_manual_seed() -> None:
    j = randn_((3, 8))
    torch.manual_seed(12345)
    a1 = GradVac(beta=0.3)
    out1 = a1(j)
    torch.manual_seed(12345)
    a2 = GradVac(beta=0.3)
    out2 = a2(j)
    assert_close(out1, out2)


def test_reset_restores_first_step_behavior() -> None:
    j = tensor([[1.0, 0.0], [0.0, 1.0]])
    torch.manual_seed(7)
    agg = GradVac(beta=0.5)
    first = agg(j)
    agg(j)
    agg.reset()
    torch.manual_seed(7)
    assert_close(first, agg(j))


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: GradVac, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: GradVac, matrix: Tensor) -> None:
    assert_non_differentiable(aggregator, matrix)
