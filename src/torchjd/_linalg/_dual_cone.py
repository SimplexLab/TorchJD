from abc import ABC, abstractmethod

from torch import Tensor

from ._matrix import PSDMatrix


class DualConeProjector(ABC):
    @abstractmethod
    def project_weights(U: Tensor, G: PSDMatrix) -> Tensor:
        r"""
        Computes the weights `w` of the projection of `J^T u` onto the dual cone of
        the rows of `J`, provided `G = J J^T` and `u`. In other words, this computes the `w` that
        satisfies `\pi_J(J^T u) = J^T w`, with `\pi_J` defined in Equation 3 of [1].

        By Proposition 1 of [1], this is equivalent to solving for `v` the following quadratic
        program:
        minimize        v^T G v
        subject to      u \preceq v

        Reference:
        [1] `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

        :param U: The tensor of weights corresponding to the vectors to project, of shape `[..., m]`.
        :param G: The Gramian matrix of shape `[m, m]`. It must be symmetric and positive definite.
        :return: A tensor of projection weights with the same shape as `U`.
        """
