# The code of this file was partly adapted from
# https://github.com/SamsungLabs/MTL/tree/master/code/optim/aligned.
# It is therefore also subject to the following license.
#
# MIT License
#
# Copyright (c) 2022 Samsung
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Literal, TypeAlias

import torch
from torch import Tensor

from torchjd._linalg import PSDMatrix

from ._aggregator_bases import GramianWeightedAggregator
from ._mean import MeanWeighting
from ._utils.pref_vector import pref_vector_to_str_suffix, pref_vector_to_weighting
from ._weighting_bases import Weighting

SUPPORTED_SCALE_MODE: TypeAlias = Literal["min", "median", "rmse"]


class AlignedMTL(GramianWeightedAggregator):
    r"""
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` as defined in Algorithm 1 of
    `Independent Component Alignment for Multi-Task Learning
    <https://openaccess.thecvf.com/content/CVPR2023/papers/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.pdf>`_.

    :param pref_vector: The preference vector to use.  If not provided, defaults to
        :math:`\begin{bmatrix} \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^T \in \mathbb{R}^m`.
    :param scale_mode: The scaling mode used to build the balance transformation. ``"min"`` uses
        the smallest eigenvalue (default), ``"median"`` uses the median eigenvalue, and ``"rmse"``
        uses the mean eigenvalue (as in the original implementation).

    .. note::
        This implementation was adapted from the official implementation of SamsungLabs/MTL,
        which is not available anymore at the time of writing.
    """

    def __init__(
        self,
        pref_vector: Tensor | None = None,
        scale_mode: SUPPORTED_SCALE_MODE = "min",
    ):
        self._pref_vector = pref_vector
        self._scale_mode: SUPPORTED_SCALE_MODE = scale_mode
        super().__init__(AlignedMTLWeighting(pref_vector, scale_mode=scale_mode))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(pref_vector={repr(self._pref_vector)}, "
            f"scale_mode={repr(self._scale_mode)})"
        )

    def __str__(self) -> str:
        return f"AlignedMTL{pref_vector_to_str_suffix(self._pref_vector)}"


class AlignedMTLWeighting(Weighting[PSDMatrix]):
    r"""
    :class:`~torchjd.aggregation._weighting_bases.Weighting` giving the weights of
    :class:`~torchjd.aggregation.AlignedMTL`.

    :param pref_vector: The preference vector to use. If not provided, defaults to
        :math:`\begin{bmatrix} \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^T \in \mathbb{R}^m`.
    :param scale_mode: The scaling mode used to build the balance transformation. ``"min"`` uses
        the smallest eigenvalue (default), ``"median"`` uses the median eigenvalue, and ``"rmse"``
        uses the mean eigenvalue (as in the original implementation).
    """

    def __init__(
        self,
        pref_vector: Tensor | None = None,
        scale_mode: SUPPORTED_SCALE_MODE = "min",
    ):
        super().__init__()
        self._pref_vector = pref_vector
        self._scale_mode: SUPPORTED_SCALE_MODE = scale_mode
        self.weighting = pref_vector_to_weighting(pref_vector, default=MeanWeighting())

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        w = self.weighting(gramian)
        B = self._compute_balance_transformation(gramian, self._scale_mode)
        alpha = B @ w

        return alpha

    @staticmethod
    def _compute_balance_transformation(
        M: Tensor,
        scale_mode: SUPPORTED_SCALE_MODE = "min",
    ) -> Tensor:
        lambda_, V = torch.linalg.eigh(M, UPLO="U")  # More modern equivalent to torch.symeig
        tol = torch.max(lambda_) * len(M) * torch.finfo().eps
        rank = sum(lambda_ > tol)

        if rank == 0:
            identity = torch.eye(len(M), dtype=M.dtype, device=M.device)
            return identity

        order = torch.argsort(lambda_, dim=-1, descending=True)
        lambda_, V = lambda_[order][:rank], V[:, order][:, :rank]

        sigma_inv = torch.diag(1 / lambda_.sqrt())

        if scale_mode == "min":
            scale = lambda_[-1]
        elif scale_mode == "median":
            scale = torch.median(lambda_)
        elif scale_mode == "rmse":
            scale = lambda_.mean()
        else:
            raise ValueError(
                f"Invalid scale_mode={scale_mode!r}. Expected 'min', 'median', or 'rmse'.",
            )

        B = scale.sqrt() * V @ sigma_inv @ V.T
        return B
