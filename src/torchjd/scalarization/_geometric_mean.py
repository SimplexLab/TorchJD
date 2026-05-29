import torch
from torch import Tensor

from ._scalarizer_base import Scalarizer


class GeometricMean(Scalarizer):
    """
    :class:`~torchjd.scalarization.Scalarizer` that returns the geometric mean of the input tensor
    of values, as studied in `MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss
    Strategy for Multi-Task Learning
    <https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf>`_.

    This method is also known as GLS (Geometric Loss Strategy).
    """

    def forward(self, values: Tensor, /) -> Tensor:
        if (values < 1e-12).any():
            raise ValueError(
                "GeometricMean is only defined for strictly positive values. Found a value "
                "below 1e-12 in the input."
            )
        return torch.exp(torch.log(values).mean())
