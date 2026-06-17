from torch import Tensor
from torch.nn.functional import cosine_similarity

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
    :param weights: The preference vector :math:`r` applied to the values. It must have the same
        shape as the values passed at call time. To approximate the whole Pareto front rather than a
        single trade-off, it should be re-sampled from a Dirichlet distribution and reassigned before
        every call, as in the paper, e.g. for ``m`` objectives
        ``cosmos.weights = torch.distributions.Dirichlet(torch.ones(m)).sample()`` (a uniform
        distribution over the probability simplex; a concentration smaller than one spreads the
        samples toward the corners of the simplex).

    .. note::
        The full COSMOS method also conditions the model on the preference vector by concatenating it
        to the input; that is a modeling choice left to the user. This scalarizer only implements the
        objective.
    """

    def __init__(self, lambda_: float, weights: Tensor) -> None:
        if lambda_ < 0.0:
            raise ValueError(
                f"Parameter `lambda_` should be non-negative. Found `lambda_ = {lambda_}`."
            )

        super().__init__()
        self.lambda_ = lambda_
        self.weights = weights

    def forward(self, values: Tensor, /) -> Tensor:
        if self.weights.shape != values.shape:
            raise ValueError(
                f"Parameter `weights` should have the same shape as `values`. Found "
                f"`weights.shape = {tuple(self.weights.shape)}` and `values.shape = "
                f"{tuple(values.shape)}`."
            )

        weighted_sum = (self.weights * values).sum()
        cosine = cosine_similarity(self.weights.flatten(), values.flatten(), dim=0)
        return weighted_sum - self.lambda_ * cosine

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lambda_={self.lambda_}, weights={self.weights!r})"
