import pytest
import torch
from torchjd.aggregation import Composition
from torchjd.aggregation._aggregator import Aggregator

class DummyScale(Aggregator):
    """Scales input matrix by a factor."""
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def __call__(self, matrix: torch.Tensor) -> torch.Tensor:
        return matrix * self.factor

class DummySum(Aggregator):
    """Sums matrix rows."""
    def __call__(self, matrix: torch.Tensor) -> torch.Tensor:
        return matrix.sum(dim=0)

def test_composition_lshift_operator():
    scale = DummyScale(2.0)
    sum_agg = DummySum()

    # Test composition via operator
    composed = sum_agg << scale

    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # Forward pass: scale matrix by 2, then sum rows
    result = composed(matrix)
    expected = (matrix * 2.0).sum(dim=0)

    assert torch.allclose(result, expected)
    assert str(composed) == f"{sum_agg} << {scale}"

def test_composition_sequential_execution():
    J = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    comp = Composition([DummyScale(2.0), DummySum()])
    result = comp(J)
    expected = torch.tensor([10.0, 14.0, 18.0])
    assert torch.allclose(result, expected)

def test_lshift_operator_syntax():
    scale = DummyScale(2.0)
    sum_agg = DummySum()
    comp = sum_agg << scale
    assert isinstance(comp, Composition)
    assert len(comp.aggregators) == 2
