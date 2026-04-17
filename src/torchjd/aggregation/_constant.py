from torch import Tensor

from torchjd.aggregation._weighting_bases import FromNothingWeighting

from ._aggregator_bases import WeightedAggregator
from ._utils.str import vector_to_str
from ._weighting_bases import Weighting


class _ConstantWeighting(Weighting[None]):
    """
    :class:`~torchjd.aggregation._weighting_bases.Weighting` that returns constant, pre-determined
    weights.

    :param weights: The weights to return at each call.
    """

    def __init__(self, weights: Tensor) -> None:
        if weights.dim() != 1:
            raise ValueError(
                "Parameter `weights` should be a 1-dimensional tensor. Found `weights.shape = "
                f"{weights.shape}`.",
            )

        super().__init__()
        self.weights = weights

    def forward(self, _: None, /) -> Tensor:
        return self.weights


class ConstantWeighting(FromNothingWeighting):
    def __init__(self, weights: Tensor) -> None:
        super().__init__(_ConstantWeighting(weights))


class Constant(WeightedAggregator):
    """
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` that makes a linear combination of
    the rows of the provided matrix, with constant, pre-determined weights.

    :param weights: The weights associated to the rows of the input matrices.
    """

    weighting: ConstantWeighting

    def __init__(self, weights: Tensor) -> None:
        super().__init__(weighting=ConstantWeighting(weights=weights))
        self._weights = weights

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weights={repr(self._weights)})"

    def __str__(self) -> str:
        weights_str = vector_to_str(self._weights)
        return f"{self.__class__.__name__}([{weights_str}])"
