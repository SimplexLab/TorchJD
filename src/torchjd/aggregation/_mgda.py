from typing import Literal, cast

import torch
from torch import Tensor

from torchjd._linalg import PSDMatrix

from ._aggregator_bases import GramianWeightedAggregator
from ._weighting_bases import Weighting

_NormType = Literal["none", "l2", "loss", "loss+"]


class MGDA(GramianWeightedAggregator):
    r"""
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` performing the gradient aggregation
    step of `Multiple-gradient descent algorithm (MGDA) for multiobjective optimization
    <https://comptes-rendus.academie-sciences.fr/mathematique/articles/10.1016/j.crma.2012.03.014/>`_.
    The implementation is based on Algorithm 2 of `Multi-Task Learning as Multi-Objective
    Optimization
    <https://proceedings.neurips.cc/paper_files/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf>`_.

    :param norm_type: The type of gradient normalization to apply before solving:

        - ``"none"`` (default): No normalization.
        - ``"l2"``: Normalize each gradient by its L2 norm.
        - ``"loss"``: Normalize each gradient by its corresponding loss value. Requires calling
          :meth:`set_losses` before each aggregation.
        - ``"loss+"``: Normalize each gradient by (loss × L2 norm). Requires calling
          :meth:`set_losses` before each aggregation.

    :param epsilon: The value of :math:`\hat{\gamma}` below which we stop the optimization.
    :param max_iters: The maximum number of iterations of the optimization loop.

    .. warning::
        When using ``norm_type`` other than ``"none"``, the iterative solver may exhibit
        convergence sensitivity, potentially leading to slightly different solutions for equivalent
        inputs. Use with caution and consider increasing ``max_iters`` or decreasing ``epsilon`` if
        more consistent results are needed.

    .. note::
        When using ``norm_type="loss"`` or ``norm_type="loss+"``, you must call :meth:`set_losses`
        with the current loss values before each call to the aggregator.

    Examples
    --------

    **No normalization (default):**

    >>> from torch import tensor
    >>> from torchjd.aggregation import MGDA
    >>>
    >>> A = MGDA()  # norm_type="none" by default
    >>> J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    >>> A(J)
    tensor([0., 1., 1.])

    **L2 normalization** - normalizes each gradient by its L2 norm, helping to balance tasks with
    different gradient magnitudes:

    >>> A = MGDA(norm_type="l2")
    >>> J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    >>> A(J)
    tensor([1., 1., 1.])

    **Loss normalization** - normalizes each gradient by its corresponding loss value. Useful when
    tasks have different loss scales:

    >>> A = MGDA(norm_type="loss")
    >>> A.set_losses(tensor([0.5, 2.0]))  # Must set losses before aggregation
    >>> J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    >>> A(J)
    tensor([3.4900, 1.0000, 1.0000])

    **Loss+ normalization** - normalizes each gradient by (loss × L2 norm). Combines both loss and
    gradient magnitude balancing:

    >>> A = MGDA(norm_type="loss+")
    >>> A.set_losses(tensor([0.5, 2.0]))  # Must set losses before aggregation
    >>> J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    >>> A(J)
    tensor([4.1606, 1.0000, 1.0000])
    """

    def __init__(
        self,
        norm_type: _NormType = "none",
        epsilon: float = 1e-5,
        max_iters: int = 250,
    ):
        if norm_type not in ("none", "l2", "loss", "loss+"):
            raise ValueError(
                f"Parameter `norm_type` should be 'none', 'l2', 'loss', or 'loss+'. Found "
                f"`norm_type = {norm_type!r}`."
            )

        mgda_weighting = MGDAWeighting(norm_type=norm_type, epsilon=epsilon, max_iters=max_iters)
        super().__init__(mgda_weighting)
        self._mgda_weighting = mgda_weighting
        self._norm_type = norm_type
        self._epsilon = epsilon
        self._max_iters = max_iters

    def set_losses(self, losses: Tensor) -> None:
        """
        Set the loss values to use for normalization.

        This method must be called before each aggregation when using ``norm_type="loss"`` or
        ``norm_type="loss+"``.

        :param losses: A 1D tensor of loss values, one per task/row of the Jacobian matrix.
        """
        self._mgda_weighting.set_losses(losses)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(norm_type={self._norm_type!r}, "
            f"epsilon={self._epsilon}, max_iters={self._max_iters})"
        )


class MGDAWeighting(Weighting[PSDMatrix]):
    r"""
    :class:`~torchjd.aggregation._weighting_bases.Weighting` giving the weights of
    :class:`~torchjd.aggregation.MGDA`.

    :param norm_type: The type of gradient normalization to apply before solving:

        - ``"none"`` (default): No normalization.
        - ``"l2"``: Normalize each gradient by its L2 norm.
        - ``"loss"``: Normalize each gradient by its corresponding loss value. Requires calling
          :meth:`set_losses` before each forward pass.
        - ``"loss+"``: Normalize each gradient by (loss × L2 norm). Requires calling
          :meth:`set_losses` before each forward pass.

    :param epsilon: The value of :math:`\hat{\gamma}` below which we stop the optimization.
    :param max_iters: The maximum number of iterations of the optimization loop.

    .. warning::
        When using ``norm_type`` other than ``"none"``, the iterative solver may exhibit
        convergence sensitivity, potentially leading to slightly different solutions for equivalent
        inputs. Use with caution and consider increasing ``max_iters`` or decreasing ``epsilon`` if
        more consistent results are needed.
    """

    def __init__(
        self,
        norm_type: _NormType = "none",
        epsilon: float = 1e-5,
        max_iters: int = 250,
    ):
        super().__init__()

        if norm_type not in ("none", "l2", "loss", "loss+"):
            raise ValueError(
                f"Parameter `norm_type` should be 'none', 'l2', 'loss', or 'loss+'. Found "
                f"`norm_type = {norm_type!r}`."
            )

        self.norm_type = norm_type
        self.epsilon = epsilon
        self.max_iters = max_iters
        self._losses: Tensor | None = None

    def set_losses(self, losses: Tensor) -> None:
        """
        Set the loss values to use for normalization.

        This method must be called before each forward pass when using ``norm_type="loss"`` or
        ``norm_type="loss+"``.

        :param losses: A 1D tensor of loss values, one per task/row of the Gramian matrix.
        """
        if losses.dim() != 1:
            raise ValueError(
                f"Parameter `losses` should be a 1D tensor. Found `losses.shape = {losses.shape}`."
            )
        self._losses = losses.detach()

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        """
        This is the Frank-Wolfe solver in Algorithm 2 of `Multi-Task Learning as Multi-Objective
        Optimization
        <https://proceedings.neurips.cc/paper_files/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf>`_.
        """
        # Apply gradient normalization if requested
        if self.norm_type == "l2":
            gramian = self._normalize_gramian_l2(gramian)
        elif self.norm_type == "loss":
            gramian = self._normalize_gramian_loss(gramian)
        elif self.norm_type == "loss+":
            gramian = self._normalize_gramian_loss_plus(gramian)

        device = gramian.device
        dtype = gramian.dtype

        alpha = torch.ones(gramian.shape[0], device=device, dtype=dtype) / gramian.shape[0]
        for _ in range(self.max_iters):
            t = torch.argmin(gramian @ alpha)
            e_t = torch.zeros(gramian.shape[0], device=device, dtype=dtype)
            e_t[t] = 1.0
            a = alpha @ (gramian @ e_t)
            b = alpha @ (gramian @ alpha)
            c = e_t @ (gramian @ e_t)
            if c <= a:
                gamma = 1.0
            elif b <= a:
                gamma = 0.0
            else:
                gamma = (b - a) / (b + c - 2 * a)
            alpha = (1 - gamma) * alpha + gamma * e_t
            if gamma < self.epsilon:
                break
        return alpha

    @staticmethod
    def _normalize_gramian_l2(gramian: PSDMatrix) -> PSDMatrix:
        """
        Normalize the Gramian as if each gradient was normalized by its L2 norm.

        If G = J @ J.T, normalizing each row of J by its L2 norm gives:
        G_norm[i,j] = G[i,j] / (||J[i]|| * ||J[j]||)
        where ||J[i]|| = sqrt(G[i,i])
        """
        grad_norms = torch.sqrt(torch.diag(gramian).clamp(min=1e-20))
        norm_matrix = grad_norms.unsqueeze(1) * grad_norms.unsqueeze(0)
        return cast(PSDMatrix, gramian / norm_matrix)

    def _normalize_gramian_loss(self, gramian: PSDMatrix) -> PSDMatrix:
        """
        Normalize the Gramian as if each gradient was normalized by its loss value.

        If G = J @ J.T, normalizing each row of J by loss[i] gives:
        G_norm[i,j] = G[i,j] / (loss[i] * loss[j])
        """
        if self._losses is None:
            raise RuntimeError(
                "Losses must be set before calling forward() when using norm_type='loss'. "
                "Call set_losses() first."
            )

        n = gramian.shape[0]
        if self._losses.shape[0] != n:
            raise ValueError(
                f"Number of losses ({self._losses.shape[0]}) must match the number of rows in "
                f"the gramian ({n})."
            )

        losses = self._losses.to(device=gramian.device, dtype=gramian.dtype).clamp(min=1e-20)
        norm_matrix = losses.unsqueeze(1) * losses.unsqueeze(0)
        return cast(PSDMatrix, gramian / norm_matrix)

    def _normalize_gramian_loss_plus(self, gramian: PSDMatrix) -> PSDMatrix:
        """
        Normalize the Gramian as if each gradient was normalized by (loss × L2 norm).

        If G = J @ J.T, normalizing each row of J by (loss[i] * ||J[i]||) gives:
        G_norm[i,j] = G[i,j] / (loss[i] * ||J[i]|| * loss[j] * ||J[j]||)
        """
        if self._losses is None:
            raise RuntimeError(
                "Losses must be set before calling forward() when using norm_type='loss+'. "
                "Call set_losses() first."
            )

        n = gramian.shape[0]
        if self._losses.shape[0] != n:
            raise ValueError(
                f"Number of losses ({self._losses.shape[0]}) must match the number of rows in "
                f"the gramian ({n})."
            )

        losses = self._losses.to(device=gramian.device, dtype=gramian.dtype).clamp(min=1e-20)
        grad_norms = torch.sqrt(torch.diag(gramian).clamp(min=1e-20))
        combined_norms = losses * grad_norms
        norm_matrix = combined_norms.unsqueeze(1) * combined_norms.unsqueeze(0)
        return cast(PSDMatrix, gramian / norm_matrix)
