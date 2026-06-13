from torch import Tensor

from torchjd._vector_str import pref_vector_to_str_suffix

from ._scalarizer_base import Scalarizer


class Constant(Scalarizer):
    """
    :class:`~torchjd.scalarization.Scalarizer` that combines the input tensor of values with
    constant, pre-determined weights.

    :param weights: The weights to apply to the values. Must have the same shape as the values
        passed at call time.
    """

    def __init__(self, weights: Tensor) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, values: Tensor, /) -> Tensor:
        if values.shape != self.weights.shape:
            raise ValueError(
                f"Parameter `values` should have shape {tuple(self.weights.shape)} (matching the "
                f"shape of the weights). Found `values.shape = {tuple(values.shape)}`.",
            )
        return (self.weights * values).sum()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weights={repr(self.weights)})"

    def __str__(self) -> str:
        if self.weights.ndim == 1:
            return f"{self.__class__.__name__}{pref_vector_to_str_suffix(self.weights)}"
        return f"{self.__class__.__name__}(weights of shape {tuple(self.weights.shape)})"
