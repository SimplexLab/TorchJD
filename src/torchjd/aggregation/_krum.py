import torch
from torch import Tensor
from torch.nn import functional as F

from torchjd._linalg import PSDMatrix

from ._aggregator_bases import GramianWeightedAggregator
from ._weighting_bases import Weighting


class Krum(GramianWeightedAggregator):
    """
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` for adversarial federated learning,
    as defined in `Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent
    <https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf>`_.

    :param n_byzantine: The number of rows of the input matrix that can come from an adversarial
        source.
    :param n_selected: The number of selected rows in the context of Multi-Krum. Defaults to 1.
    """

    def __init__(self, n_byzantine: int, n_selected: int = 1) -> None:
        self._n_byzantine = n_byzantine
        self._n_selected = n_selected
        super().__init__(KrumWeighting(n_byzantine=n_byzantine, n_selected=n_selected))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_byzantine={self._n_byzantine}, n_selected="
            f"{self._n_selected})"
        )

    def __str__(self) -> str:
        return f"Krum{self._n_byzantine}-{self._n_selected}"


class KrumWeighting(Weighting[PSDMatrix]):
    """
    :class:`~torchjd.aggregation._weighting_bases.Weighting` giving the weights of
    :class:`~torchjd.aggregation.Krum`.

    :param n_byzantine: The number of rows of the input matrix that can come from an adversarial
        source.
    :param n_selected: The number of selected rows in the context of Multi-Krum. Defaults to 1.
    """

    def __init__(self, n_byzantine: int, n_selected: int = 1) -> None:
        super().__init__()
        if n_byzantine < 0:
            raise ValueError(
                "Parameter `n_byzantine` should be a non-negative integer. Found `n_byzantine = "
                f"{n_byzantine}`.",
            )

        if n_selected < 1:
            raise ValueError(
                "Parameter `n_selected` should be a positive integer. Found `n_selected = "
                f"{n_selected}`.",
            )

        self.n_byzantine = n_byzantine
        self.n_selected = n_selected

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        self._check_matrix_shape(gramian)
        gradient_norms_squared = torch.diagonal(gramian)
        distances_squared = (
            gradient_norms_squared.unsqueeze(0) + gradient_norms_squared.unsqueeze(1) - 2 * gramian
        )
        distances = torch.sqrt(distances_squared)

        n_closest = gramian.shape[0] - self.n_byzantine - 2
        smallest_distances, _ = torch.topk(distances, k=n_closest + 1, largest=False)
        smallest_distances_excluding_self = smallest_distances[:, 1:]
        scores = smallest_distances_excluding_self.sum(dim=1)

        _, selected_indices = torch.topk(scores, k=self.n_selected, largest=False)
        one_hot_selected_indices = F.one_hot(selected_indices, num_classes=gramian.shape[0])
        weights = one_hot_selected_indices.sum(dim=0).to(dtype=gramian.dtype) / self.n_selected

        return weights

    def _check_matrix_shape(self, gramian: PSDMatrix) -> None:
        min_rows = self.n_byzantine + 3
        if gramian.shape[0] < min_rows:
            raise ValueError(
                f"Parameter `gramian` should have at least {min_rows} rows (n_byzantine + 3). Found"
                f" `gramian` with {gramian.shape[0]} rows.",
            )

        if gramian.shape[0] < self.n_selected:
            raise ValueError(
                f"Parameter `gramian` should have at least {self.n_selected} rows (n_selected). "
                f"Found `gramian` with {gramian.shape[0]} rows.",
            )
