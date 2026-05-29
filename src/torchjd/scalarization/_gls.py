import torch
from torch import Tensor

from ._scalarizer_base import Scalarizer


class GLS(Scalarizer):
    """
    :class:`~torchjd.scalarization.Scalarizer` that returns the geometric mean of the input tensor
    of values, as studied in `MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss
    Strategy for Multi-Task Learning
    <https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf>`_.
    """

    def forward(self, values: Tensor, /) -> Tensor:
        return torch.exp(torch.log(values).mean())
