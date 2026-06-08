from collections.abc import Sequence

import torch
from torch import Tensor, nn

from torchjd._mixins import Stateful

from ._scalarizer_base import Scalarizer


class UW(Scalarizer, Stateful):
    r"""
    :class:`~torchjd.scalarization.Stateful`
    :class:`~torchjd.scalarization.Scalarizer` that combines the input tensor of values using
    learned per-task uncertainties. ``UW`` is short for Uncertainty Weighting, the method proposed
    in `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
    <https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf>`_.

    Each value :math:`L_i` is assigned a learnable log-variance :math:`s_i`, and the values are
    combined as

    .. math::
        \sum_i \left( \frac{1}{2} e^{-s_i} L_i + \frac{1}{2} s_i \right)

    where:

    - :math:`L_i` is the :math:`i`-th value (typically the loss of task :math:`i`);
    - :math:`s_i = \log \sigma_i^2` is the learnable log-variance of task :math:`i`.

    Following the paper, the log-variance :math:`s_i` is learned rather than the variance
    :math:`\sigma_i^2` directly: this is numerically more stable (the combination never divides by
    zero) and keeps :math:`s_i` unconstrained, since :math:`e^{-s_i}` is always positive. The
    :math:`s_i` are stored as an ``nn.Parameter``, so the parameters of this scalarizer must be
    passed to the optimizer to be learned jointly with the model.

    :param shape: The shape of the values to scalarize, used to create one log-variance per value.
        An ``int`` ``n`` is interpreted as the shape ``(n,)``.

    The following example shows train a model with Uncertainty Weighting, as described in the paper.

        >>> import torch
        >>> from torch.nn import Linear
        >>>
        >>> from torchjd.scalarization import UW
        >>>
        >>> model = Linear(3, 2)
        >>> scalarizer = UW(2)  # Move to the right device with e.g. UW(2).to(device="cuda")
        >>> optimizer = torch.optim.SGD([*model.parameters(), *scalarizer.parameters()], lr=0.1)
        >>>
        >>> features = torch.randn(8, 3)
        >>> # Compute some dummy losses just for the sake of the example
        >>> losses = model(features).pow(2).mean(dim=0)  # One loss per output dimension.
        >>> loss = scalarizer(losses)
        >>> loss.backward()
        >>> optimizer.step()

    .. note::
        The log-variances are initialized to ``0`` (i.e. :math:`\sigma_i^2 = 1`), which gives
        uniform weights at the start of training. The paper reports that the result is robust to
        this initialization. (`LibMTL <https://github.com/median-research-group/LibMTL>`_
        initializes them to ``-0.5`` instead.)
    """

    def __init__(self, shape: int | Sequence[int]) -> None:
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(shape))

    def forward(self, values: Tensor, /) -> Tensor:
        if values.shape != self.log_var.shape:
            raise ValueError(
                f"Parameter `values` should have shape {tuple(self.log_var.shape)} (matching the "
                f"shape of the log-variances). Found `values.shape = {tuple(values.shape)}`.",
            )
        return (0.5 * torch.exp(-self.log_var) * values + 0.5 * self.log_var).sum()

    def reset(self) -> None:
        with torch.no_grad():
            self.log_var.zero_()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={tuple(self.log_var.shape)})"
