# The code of this file was partly adapted from
# https://github.com/tum-pbs/ConFIG/tree/main/conflictfree.
# It is therefore also subject to the following license.
#
# MIT License
#
# Copyright (c) 2024 TUM Physics-based Simulation
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


import torch
from torch import Tensor

from torchjd._linalg import Matrix

from ._aggregator_bases import Aggregator
from ._sum import SumWeighting
from ._utils.non_differentiable import raise_non_differentiable_error
from ._utils.pref_vector import pref_vector_to_str_suffix, pref_vector_to_weighting


class ConFIG(Aggregator):
    """
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` as defined in Equation 2 of `ConFIG:
    Towards Conflict-free Training of Physics Informed Neural Networks
    <https://arxiv.org/pdf/2408.11104>`_.

    :param pref_vector: The preference vector used to weight the rows. If not provided, defaults to
        equal weights of 1.

    .. note::
        This implementation was adapted from the `official implementation
        <https://github.com/tum-pbs/ConFIG/tree/main/conflictfree>`_.
    """

    def __init__(self, pref_vector: Tensor | None = None):
        super().__init__()
        self.weighting = pref_vector_to_weighting(pref_vector, default=SumWeighting())
        self._pref_vector = pref_vector

        # This prevents computing gradients that can be very wrong.
        self.register_full_backward_pre_hook(raise_non_differentiable_error)

    def forward(self, matrix: Matrix, /) -> Tensor:
        weights = self.weighting(matrix)
        units = torch.nan_to_num((matrix / (matrix.norm(dim=1)).unsqueeze(1)), 0.0)
        best_direction = torch.linalg.pinv(units) @ weights

        unit_target_vector = torch.nn.functional.normalize(best_direction, dim=0)

        length = torch.sum(matrix @ unit_target_vector)

        return length * unit_target_vector

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pref_vector={repr(self._pref_vector)})"

    def __str__(self) -> str:
        return f"ConFIG{pref_vector_to_str_suffix(self._pref_vector)}"
