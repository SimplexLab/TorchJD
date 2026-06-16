import torch
from torch import Tensor

from ._scalarizer_base import Scalarizer


class COSMOS(Scalarizer):
    r"""
    :class:`~torchjd.scalarization.Scalarizer` that combines the input tensor of values using the
    COSMOS scalarization, proposed in `Scalable Pareto Front Approximation for Deep Multi-Objective
    Learning <https://arxiv.org/pdf/2103.13392>`_.

    It returns a linear scalarization penalized by the cosine similarity between the values and the
    preference vector:

    .. math::
        \sum_i r_i L_i - \lambda \frac{\sum_i r_i L_i}{\lVert r \rVert \, \lVert L \rVert},

    where:

    - :math:`L_i` is the :math:`i`-th input value (the :math:`i`-th objective);
    - :math:`r_i` is its preference weight (the ``weights`` parameter);
    - :math:`\lambda` is the cosine-similarity penalty coefficient (the ``lambda_`` parameter);
    - the subtracted term is :math:`\lambda \cos(r, L)`, which rewards aligning the vector of values
      with the preference direction and is what spreads the approximated Pareto front.

    :param lambda_: The cosine-similarity penalty coefficient :math:`\lambda`. Must be non-negative.
        A value of ``0`` reduces COSMOS to a plain linear scalarization. The paper uses values
        ranging from ``0.01`` to ``8`` depending on the dataset, with no single best value.
    :param weights: The preference vector :math:`r` applied to the values (in the paper, sampled on
        the probability simplex). If ``None``, a uniform preference summing to one is used. If
        provided, it must have the same shape as the values passed at call time.

    .. note::
        COSMOS divides by :math:`\lVert L \rVert`, so an all-zero vector of values produces ``nan``.
        This is not enforced.

    .. note::
        The full COSMOS method also conditions the model on the preference vector by concatenating it
        to the input; that is a modeling choice left to the user. This scalarizer only implements the
        objective. The `libmoon <https://github.com/xzhang2523/libmoon>`_ reference normalizes the
        linear term by :math:`\lVert r \rVert`; here the linear term is the raw weighted sum, as in
        the paper and the official implementation.
    """

    def __init__(self, lambda_: float, weights: Tensor | None = None) -> None:
        if lambda_ < 0.0:
            raise ValueError(
                f"Parameter `lambda_` should be non-negative. Found `lambda_ = {lambda_}`."
            )

        super().__init__()
        self.lambda_ = lambda_
        self.weights = weights

    def forward(self, values: Tensor, /) -> Tensor:
        if self.weights is not None and self.weights.shape != values.shape:
            raise ValueError(
                f"Parameter `weights` should have the same shape as `values`. Found "
                f"`weights.shape = {tuple(self.weights.shape)}` and `values.shape = "
                f"{tuple(values.shape)}`."
            )

        if self.weights is None:
            weights = torch.full_like(values, 1.0 / values.numel())
        else:
            weights = self.weights

        weighted_sum = (weights * values).sum()
        cosine_similarity = weighted_sum / (weights.norm() * values.norm())
        return weighted_sum - self.lambda_ * cosine_similarity

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lambda_={self.lambda_}, weights={self.weights!r})"
