from collections.abc import Sequence

import torch
from torch import Tensor, nn

from torchjd._mixins import Stateful

from ._scalarizer_base import Scalarizer


class IMTLL(Scalarizer, Stateful):
    r"""
    :class:`~torchjd.Stateful`
    :class:`~torchjd.scalarization.Scalarizer` that combines the input tensor of values using learned
    per-task scales. ``IMTL-L`` is the loss-balancing variant of Impartial
    Multi-Task Learning, proposed in `Towards Impartial Multi-Task Learning
    <https://openreview.net/pdf?id=IMPnRXEWpvr>`_.

    Each value :math:`L_i` is assigned a learnable scale parameter :math:`s_i`, and the values are
    combined as

    .. math::
        \sum_i \left( e^{s_i} L_i - s_i \right)

    where:

    - :math:`L_i` is the :math:`i`-th value (typically the loss of task :math:`i`);
    - :math:`s_i` is the learnable scale parameter of task :math:`i`.

    The factor :math:`e^{s_i}` rescales each loss so that the scaled losses stay at a comparable
    magnitude across tasks, while the :math:`- s_i` term is a regularizer that prevents the trivial
    solution :math:`s_i \to -\infty`. The :math:`s_i` are stored as an ``nn.Parameter``, so the
    parameters of this scalarizer must be passed to the optimizer to be learned jointly with the
    model.

    Although it is derived without any distribution assumption (unlike
    :class:`~torchjd.scalarization.UW`, which is derived from Gaussian/Laplace likelihoods), IMTL-L
    is in fact almost equivalent to :class:`~torchjd.scalarization.UW`: this scalarization equals
    :math:`2\,\mathrm{UW}` evaluated at the negated parameter, so the two differ only by a constant
    factor of two and the sign convention of the learned parameter, and share the same per-task
    weighting and the same optima.

    The complementary gradient-balancing variant (IMTL-G) is provided as the
    :class:`~torchjd.aggregation.IMTLG` aggregator.

    :param shape: The shape of the values to scalarize, used to create one scale per value. An
        ``int`` ``n`` is interpreted as the shape ``(n,)``.

    The following example shows how to train a model with Impartial Multi-Task Learning (loss
    balance), as described in the paper.

        >>> import torch
        >>> from torch.nn import Linear
        >>>
        >>> from torchjd.scalarization import IMTLL
        >>>
        >>> model = Linear(3, 2)
        >>> scalarizer = IMTLL(2)  # Move to the right device with e.g. IMTLL(2).to(device="cuda")
        >>> optimizer = torch.optim.SGD([*model.parameters(), *scalarizer.parameters()], lr=0.1)
        >>>
        >>> features = torch.randn(8, 3)
        >>> # Compute some dummy losses just for the sake of the example
        >>> losses = model(features).pow(2).mean(dim=0)  # One loss per output dimension.
        >>> loss = scalarizer(losses)
        >>> loss.backward()
        >>> optimizer.step()

    .. note::
        The scales are initialized to ``0``, so at the start of training the scalarization reduces to
        the plain sum of the values (since :math:`e^0 = 1`). Following the paper, IMTL-L is designed
        to balance positive losses.
    """

    def __init__(self, shape: int | Sequence[int]) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(shape))

    def forward(self, values: Tensor, /) -> Tensor:
        if values.shape != self.log_scale.shape:
            raise ValueError(
                f"Parameter `values` should have shape {tuple(self.log_scale.shape)} (matching the "
                f"shape of the scales). Found `values.shape = {tuple(values.shape)}`.",
            )
        return (torch.exp(self.log_scale) * values - self.log_scale).sum()

    def reset(self) -> None:
        with torch.no_grad():
            self.log_scale.zero_()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={tuple(self.log_scale.shape)})"
