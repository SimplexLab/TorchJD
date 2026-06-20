import torch
from torch import Tensor

from ._scalarizer_base import Scalarizer

_EPSILON = 1e-12


class PBI(Scalarizer):
    r"""
    :class:`~torchjd.scalarization.Scalarizer` that combines the input tensor of values using the
    Penalty-based Boundary Intersection (PBI) scalarization, proposed in `MOEA/D: A Multiobjective
    Evolutionary Algorithm Based on Decomposition <https://ieeexplore.ieee.org/document/4358754>`_.

    It decomposes the values, relative to a reference point, into a component along a preference
    direction and a component perpendicular to it, and penalizes the latter:

    .. math::
        d_1 = (L - z^*)^\top \hat w, \qquad
        d_2 = \lVert (L - z^*) - d_1 \hat w \rVert, \qquad
        d_1 + \theta\, d_2,

    where:

    - :math:`L_i` is the :math:`i`-th input value (the :math:`i`-th objective);
    - :math:`z^*` is the reference (ideal) point (the ``reference`` parameter);
    - :math:`\hat w = w / \lVert w \rVert` is the normalized preference direction (the ``weights``
      parameter);
    - :math:`d_1` is the distance along the preference direction and :math:`d_2` is the distance to
      it;
    - :math:`\theta` is the penalty coefficient applied to :math:`d_2` (the ``theta`` parameter).

    :param theta: The penalty coefficient :math:`\theta` applied to the perpendicular distance. Must
        be non-negative. A value of ``0`` reduces PBI to the projection onto the preference
        direction. The paper uses ``5`` in its experiments; there is no single best value, and the
        paper notes that a too large or too small value worsens the result.
    :param weights: The preference vector :math:`w`, giving the direction along which the values are
        decomposed. Its values should be non-negative. It must have the same shape as the values
        passed at call time. To approximate the whole Pareto front rather than a single trade-off, it
        should be re-sampled from a Dirichlet distribution and reassigned before every call, e.g. for
        ``m`` objectives ``pbi.weights = torch.distributions.Dirichlet(torch.ones(m)).sample()``.
    :param reference: The reference (ideal) point :math:`z^*` subtracted from the values. It should
        be a lower bound on the values. If ``None``, the origin is used, which assumes non-negative
        values. If provided, it must have the same shape as the values passed at call time.

    .. note::
        :math:`d_2` is a Euclidean norm, whose gradient is undefined when the values lie exactly on
        the preference direction (:math:`d_2 = 0`). To keep the gradient finite there, a small
        constant is added under the square root; this shifts the result by at most around
        :math:`10^{-6}` at that point and is negligible elsewhere.
    """

    def __init__(self, theta: float, weights: Tensor, reference: Tensor | None = None) -> None:
        if theta < 0.0:
            raise ValueError(f"Parameter `theta` should be non-negative. Found `theta = {theta}`.")

        super().__init__()
        self.theta = theta
        self.weights = weights
        self.reference = reference

    def forward(self, values: Tensor, /) -> Tensor:
        if self.weights.shape != values.shape:
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

        shifted = values if self.reference is None else values - self.reference
        f = shifted.flatten()
        direction = self.weights.flatten()
        direction = direction / direction.norm()

        d1 = f @ direction
        perpendicular = f - d1 * direction
        # `perpendicular` has a zero norm when the values lie exactly on the preference direction
        # (always the case for a single-objective input, which has no perpendicular component). The
        # norm's gradient is then undefined, so we add a small constant under the square root to keep
        # it finite; this shifts the result by at most around 1e-6 there and is negligible elsewhere.
        d2 = torch.sqrt(perpendicular @ perpendicular + _EPSILON)
        return d1 + self.theta * d2

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(theta={self.theta}, weights={self.weights!r}, "
            f"reference={self.reference!r})"
        )
