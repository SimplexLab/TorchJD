import torch
from torch import Tensor
from typing import Sequence
from torchjd.aggregation.abstract import Aggregator

class Composition(Aggregator):
    """
    Composes a sequence of aggregators/transforms into a single aggregator.
    Executes each aggregator sequentially on the Jacobian matrix.
    """
    def __init__(self, aggregators: Sequence[Aggregator]):
        super().__init__()
        if not aggregators:
            raise ValueError("Aggregators sequence cannot be empty.")
        self.aggregators = list(aggregators)

    def __call__(self, matrix: Tensor) -> Tensor:
        out = matrix
        for aggregator in self.aggregators:
            out = aggregator(out)
        return out

    def __repr__(self) -> str:
        names = [agg.__class__.__name__ for agg in self.aggregators]
        return f"Composition({', '.join(names)})"
