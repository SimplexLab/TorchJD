from pytest import mark
from torch.testing import assert_close
from utils.tensors import randn_

from torchjd._linalg import PSDMatrix, compute_gramian, flatten
from torchjd.aggregation import Flattening, MeanWeighting, SumWeighting, UPGradWeighting, Weighting


@mark.parametrize(
    "half_shape",
    [
        [1],
        [12],
        [4, 3],
        [2, 3, 2],
    ],
)
@mark.parametrize(
    "weighting",
    [
        SumWeighting(),
        MeanWeighting(),
        UPGradWeighting(),
    ],
)
def test_flattening(half_shape: list[int], weighting: Weighting[PSDMatrix]) -> None:
    matrix = randn_([*half_shape, 2])
    generalized_gramian = compute_gramian(matrix, 1)
    gramian = flatten(generalized_gramian)

    flattening = Flattening(weighting)
    weights = flattening(generalized_gramian)

    expected_weights = weighting(gramian).reshape(half_shape)
    assert_close(weights, expected_weights)
