import torch
from pytest import mark, raises
from torch import Tensor

from torchjd.aggregation import STCH, STCHWeighting

from ._asserts import assert_expected_structure, assert_permutation_invariant
from ._inputs import scaled_matrices, typical_matrices

aggregators = [
    STCH(),
    STCH(mu=0.1),
    STCH(mu=0.5),
    STCH(mu=2.0),
    STCH(mu=10.0),
]
scaled_pairs = [(aggregator, matrix) for aggregator in aggregators for matrix in scaled_matrices]
typical_pairs = [(aggregator, matrix) for aggregator in aggregators for matrix in typical_matrices]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: STCH, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: STCH, matrix: Tensor):
    assert_permutation_invariant(aggregator, matrix)


def test_representations():
    A = STCH(mu=1.0)
    assert repr(A) == "STCH(mu=1.0, warmup_steps=None, eps=1e-20)"
    assert str(A) == "STCH(mu=1)"

    A = STCH(mu=0.5)
    assert repr(A) == "STCH(mu=0.5, warmup_steps=None, eps=1e-20)"
    assert str(A) == "STCH(mu=0.5)"

    A = STCH(mu=1.0, warmup_steps=100)
    assert repr(A) == "STCH(mu=1.0, warmup_steps=100, eps=1e-20)"
    assert str(A) == "STCH(mu=1)"


def test_invalid_mu():
    with raises(ValueError, match=r"Parameter `mu` should be a positive float"):
        STCH(mu=0.0)

    with raises(ValueError, match=r"Parameter `mu` should be a positive float"):
        STCH(mu=-1.0)


def test_invalid_warmup_steps():
    with raises(ValueError, match=r"Parameter `warmup_steps` should be a positive integer or None"):
        STCH(warmup_steps=0)

    with raises(ValueError, match=r"Parameter `warmup_steps` should be a positive integer or None"):
        STCH(warmup_steps=-1)


def test_weights_sum_to_one():
    """Test that the weights computed by STCHWeighting sum to 1."""
    weighting = STCHWeighting(mu=1.0)
    gramian = torch.tensor([[4.0, 2.0], [2.0, 9.0]])  # Gradient norms are 2 and 3
    weights = weighting(gramian)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_small_mu_focuses_on_max():
    """Test that small mu focuses weights on the task with largest gradient norm."""
    gramian = torch.tensor([[1.0, 0.0], [0.0, 100.0]])  # Norms are 1 and 10

    weighting_small_mu = STCHWeighting(mu=0.01)
    weights = weighting_small_mu(gramian)

    # With very small mu, the weight should be almost entirely on the second task
    assert weights[1] > 0.99


def test_large_mu_approaches_uniform():
    """Test that large mu approaches uniform weighting."""
    gramian = torch.tensor([[1.0, 0.0], [0.0, 100.0]])  # Very different norms

    weighting_large_mu = STCHWeighting(mu=100.0)
    weights = weighting_large_mu(gramian)

    # With very large mu, weights should approach uniform [0.5, 0.5]
    assert torch.isclose(weights[0], torch.tensor(0.5), atol=0.1)
    assert torch.isclose(weights[1], torch.tensor(0.5), atol=0.1)


def test_weighting_invalid_mu():
    with raises(ValueError, match=r"Parameter `mu` should be a positive float"):
        STCHWeighting(mu=0.0)

    with raises(ValueError, match=r"Parameter `mu` should be a positive float"):
        STCHWeighting(mu=-1.0)


def test_weighting_invalid_warmup_steps():
    with raises(ValueError, match=r"Parameter `warmup_steps` should be a positive integer or None"):
        STCHWeighting(warmup_steps=0)

    with raises(ValueError, match=r"Parameter `warmup_steps` should be a positive integer or None"):
        STCHWeighting(warmup_steps=-1)


# Tests for warmup functionality


def test_warmup_returns_uniform_during_warmup():
    """Test that during warmup, uniform weights are returned."""
    weighting = STCHWeighting(mu=1.0, warmup_steps=3)
    gramian = torch.tensor([[1.0, 0.0], [0.0, 100.0]])  # Very different norms

    # During warmup, weights should be uniform regardless of gradient norms
    for _ in range(3):
        weights = weighting(gramian)
        assert torch.isclose(weights[0], torch.tensor(0.5), atol=1e-6)
        assert torch.isclose(weights[1], torch.tensor(0.5), atol=1e-6)


def test_warmup_uses_nadir_after_warmup():
    """Test that after warmup, the nadir vector is used for normalization."""
    weighting = STCHWeighting(mu=1.0, warmup_steps=2)

    # During warmup: accumulate gradient norms
    gramian1 = torch.tensor([[4.0, 0.0], [0.0, 16.0]])  # Norms: [2, 4]
    gramian2 = torch.tensor([[4.0, 0.0], [0.0, 16.0]])  # Norms: [2, 4]

    weights1 = weighting(gramian1)  # Step 1: warmup
    weights2 = weighting(gramian2)  # Step 2: warmup

    # During warmup, should return uniform weights
    assert torch.isclose(weights1[0], torch.tensor(0.5), atol=1e-6)
    assert torch.isclose(weights2[0], torch.tensor(0.5), atol=1e-6)

    # After warmup, nadir should be [2, 4] (average of accumulated norms)
    # Now with a gramian that has different norms
    gramian3 = torch.tensor([[4.0, 0.0], [0.0, 4.0]])  # Norms: [2, 2]
    weights3 = weighting(gramian3)  # Step 3: after warmup

    # Normalized: [2/2, 2/4] = [1, 0.5]
    # log: [0, -0.693], max=0, reg: [0, -0.693]
    # exp: [1, 0.5], weights should favor first task
    assert weights3[0] > weights3[1]


def test_reset_clears_state():
    """Test that reset() clears the warmup state."""
    weighting = STCHWeighting(mu=1.0, warmup_steps=2)

    gramian = torch.tensor([[4.0, 0.0], [0.0, 16.0]])

    # Go through warmup
    weighting(gramian)
    weighting(gramian)
    weighting(gramian)  # After warmup

    assert weighting.step == 3
    assert weighting.nadir_vector is not None

    # Reset
    weighting.reset()

    assert weighting.step == 0
    assert weighting.nadir_vector is None
    assert weighting.nadir_accumulator is None

    # Should be in warmup again
    weights = weighting(gramian)
    assert torch.isclose(weights[0], torch.tensor(0.5), atol=1e-6)


def test_aggregator_reset():
    """Test that STCH.reset() properly resets the weighting state."""
    A = STCH(mu=1.0, warmup_steps=2)

    matrix = torch.tensor([[2.0, 0.0], [0.0, 4.0]])

    # Go through warmup
    A(matrix)
    A(matrix)
    A(matrix)

    # Reset through aggregator
    A.reset()

    # Weighting should be reset
    assert A._stch_weighting.step == 0
    assert A._stch_weighting.nadir_vector is None


def test_no_warmup_when_warmup_steps_none():
    """Test that no warmup occurs when warmup_steps is None."""
    weighting = STCHWeighting(mu=1.0, warmup_steps=None)
    gramian = torch.tensor([[1.0, 0.0], [0.0, 100.0]])

    # Should immediately use STCH weighting (not uniform)
    weights = weighting(gramian)

    # With mu=1.0 and norms [1, 10], the second task should have higher weight
    assert weights[1] > weights[0]


def test_warmup_step_counter():
    """Test that the step counter increments correctly."""
    weighting = STCHWeighting(mu=1.0, warmup_steps=3)
    gramian = torch.tensor([[4.0, 0.0], [0.0, 16.0]])

    assert weighting.step == 0

    weighting(gramian)
    assert weighting.step == 1

    weighting(gramian)
    assert weighting.step == 2

    weighting(gramian)
    assert weighting.step == 3

    weighting(gramian)  # First step after warmup: nadir_vector gets computed
    assert weighting.step == 4

    weighting(gramian)  # Steady-state: nadir_vector is already set
    assert weighting.step == 5


def test_warmup_with_varying_gramians():
    """Test warmup with different gramians to verify accumulation."""
    weighting = STCHWeighting(mu=1.0, warmup_steps=2)

    gramian1 = torch.tensor([[1.0, 0.0], [0.0, 4.0]])  # Norms: [1, 2]
    gramian2 = torch.tensor([[9.0, 0.0], [0.0, 16.0]])  # Norms: [3, 4]

    weighting(gramian1)  # Step 1: warmup
    weighting(gramian2)  # Step 2: warmup (completes warmup)

    # nadir_vector is computed on first call AFTER warmup
    gramian3 = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    weighting(gramian3)  # Step 3: computes nadir_vector and uses it

    # Nadir should be average: [(1+3)/2, (2+4)/2] = [2, 3]
    expected_nadir = torch.tensor([2.0, 3.0])
    assert weighting.nadir_vector is not None
    assert torch.allclose(weighting.nadir_vector, expected_nadir, atol=1e-6)
