import torch
from torch import Tensor

from ._scalarizer_base import Scalarizer


class STCH(Scalarizer):
    r"""
    :class:`~torchjd.scalarization.Scalarizer` that combines the input tensor of values using smooth
    Tchebycheff scalarization, as defined in `Smooth Tchebycheff Scalarization for Multi-Objective
    Optimization <https://openreview.net/pdf?id=m4dO5L6eCp>`_.

    It returns

    .. math::
        \mu \log \sum_{i=1}^m \exp\left(\frac{\lambda_i (f_i - z_i^*)}{\mu}\right),

    a smooth approximation of the (non-differentiable) weighted maximum
    :math:`\max_i \lambda_i (f_i - z_i^*)` that becomes tighter as ``mu`` decreases.

    Following the paper's notation:

    - :math:`f_i` is the :math:`i`-th input value (the :math:`i`-th objective),
    - :math:`m` is the number of objectives (the number of elements of the input),
    - :math:`\lambda_i` is its preference weight (the ``weights`` parameter),
    - :math:`z_i^*` is the :math:`i`-th component of the ideal point (the ``reference`` parameter),
    - :math:`\mu` is the smoothing parameter (the ``mu`` parameter).

    :param mu: The smoothing parameter :math:`\mu`. Must be strictly positive. Smaller values make
        the scalarization closer to the maximum. The paper evaluates :math:`\mu \in \{0.01, 0.1,
        0.5, 1\}` and reports that a small :math:`\mu` works reasonably well, while no single value
        is best across all problems.
    :param weights: The preference vector :math:`\lambda` applied to the values (in the paper, on
        the probability simplex). If ``None``, a uniform preference summing to one is used. If
        provided, it must have the same shape as the values passed at call time.
    :param reference: The ideal point :math:`z^*` subtracted from the values. If ``None``, no shift
        is applied. If provided, it must have the same shape as the values passed at call time.
    """

    def __init__(
        self,
        mu: float,
        weights: Tensor | None = None,
        reference: Tensor | None = None,
    ) -> None:
        if mu <= 0.0:
            raise ValueError(f"Parameter `mu` should be strictly positive. Found `mu = {mu}`.")

        super().__init__()
        self.mu = mu
        self.weights = weights
        self.reference = reference

    def forward(self, values: Tensor, /) -> Tensor:
        if self.weights is not None and self.weights.shape != values.shape:
            raise ValueError(
                f"Parameter `weights` should have the same shape as `values`. Found "
                f"`weights.shape = {tuple(self.weights.shape)}` and `values.shape = "
                f"{tuple(values.shape)}`."
            )
        if self.reference is not None and self.reference.shape != values.shape:
            raise ValueError(
                f"Parameter `reference` should have the same shape as `values`. Found "
                f"`reference.shape = {tuple(self.reference.shape)}` and `values.shape = "
                f"{tuple(values.shape)}`."
            )

        if self.weights is None:
            weights = torch.full_like(values, 1.0 / values.numel())
        else:
            weights = self.weights

        shifted = values if self.reference is None else values - self.reference

        # Center the weighted values before dividing by mu (Appendix B.1 of the paper). This keeps
        # the largest exponent at 0 so the `/ mu` step never overflows for large values and small
        # mu. Adding `max_y` back makes it value-preserving: the result and its gradient are
        # mathematically identical to `mu * logsumexp(weights * shifted / mu)`.
        y = weights * shifted
        max_y = y.max()
        exponents = (y - max_y) / self.mu
        return self.mu * torch.logsumexp(exponents.flatten(), dim=-1) + max_y

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mu={self.mu}, weights={self.weights!r}, "
            f"reference={self.reference!r})"
        )
