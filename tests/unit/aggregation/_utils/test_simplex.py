from torch.testing import assert_close
from utils.tensors import tensor_

from torchjd.aggregation._utils.simplex import project_onto_simplex


def test_project_onto_simplex_known_values() -> None:
    """The simplex projection matches hand-computed Euclidean projections."""

    # Already-positive input: the deficit (1 - sum) is spread equally, no clamping.
    assert_close(
        project_onto_simplex(tensor_([0.5, 0.1, 0.1])),
        tensor_([0.6, 0.2, 0.2]),
    )
    # Input with a negative entry: it gets clamped to zero.
    assert_close(
        project_onto_simplex(tensor_([1.0, 0.0, -0.5])),
        tensor_([1.0, 0.0, 0.0]),
    )
