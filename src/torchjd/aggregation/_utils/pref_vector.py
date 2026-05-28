from torch import Tensor

from torchjd.aggregation._constant import ConstantWeighting
from torchjd.aggregation._weighting_bases import Weighting
from torchjd.linalg import Matrix


def pref_vector_to_weighting(
    pref_vector: Tensor | None,
    default: Weighting[Matrix],
) -> Weighting[Matrix]:
    """
    Returns the weighting associated to a given preference vector, with a fallback to a default
    weighting if the preference vector is None.
    """

    if pref_vector is None:
        return default
    if pref_vector.ndim != 1:
        raise ValueError(
            "Parameter `pref_vector` must be a vector (1D Tensor). Found `pref_vector.ndim = "
            f"{pref_vector.ndim}`.",
        )
    return ConstantWeighting(pref_vector)
