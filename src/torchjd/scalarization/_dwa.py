import torch
from torch import Tensor
from torch.nn.functional import softmax

from torchjd._mixins import Stateful

from ._scalarizer_base import Scalarizer


class DWA(Scalarizer, Stateful):
    r"""
    :class:`~torchjd.Stateful`
    :class:`~torchjd.scalarization.Scalarizer` that combines the input tensor of values using Dynamic
    Weight Average (DWA), proposed in `End-to-End Multi-Task Learning with Attention
    <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_.

    DWA weights each value by how quickly its loss has been decreasing relative to the others. At
    epoch :math:`t`, the current batch's values are combined as

    .. math::
        \sum_k \lambda_k(t)\, \ell_k, \qquad
        \lambda_k(t) = \frac{K \exp(w_k(t-1) / T)}{\sum_i \exp(w_i(t-1) / T)}, \qquad
        w_k(t-1) = \frac{L_k(t-1)}{L_k(t-2)}

    where:

    - :math:`\ell_k` is the :math:`k`-th value being scalarized (typically the current batch's loss
      for task k);
    - :math:`L_k(t)` is the :math:`k`-th value averaged over epoch :math:`t` (used only for the
      weights);
    - :math:`w_k(t-1)` is the relative descending rate: the ratio of average losses over the two
      previous epochs;
    - :math:`T` is the temperature; a larger :math:`T` makes the weights more uniform;
    - :math:`K` is the number of values (e.g. the number of tasks); the factor :math:`K` keeps
      :math:`\sum_k \lambda_k = K`.

    The weights use only the two previous epochs' average losses, so they need no gradient. At each
    call, the scalarization is returned and the current batch's losses are summed to the current
    epoch's loss sums. :meth:`step` must then be called once at the end of each epoch to finalize
    that epoch's average loss and roll the history forward. During the first two epochs, before two
    averages are available, the weights are uniform.

    :param temperature: The temperature :math:`T`. Must be strictly positive. Larger values make the
        weights more uniform. The paper uses ``2.0``.

    The following example shows how to train a model with DWA. The scalarizer is called on every
    batch, and :meth:`step` is called once at the end of each epoch.

        >>> import torch
        >>> from torch.nn import Linear
        >>>
        >>> from torchjd.scalarization import DWA
        >>>
        >>> model = Linear(3, 2)
        >>> scalarizer = DWA()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>>
        >>> for epoch in range(3):
        ...     for _ in range(4):  # Iterate over the batches of the epoch.
        ...         features = torch.randn(8, 3)
        ...         losses = model(features).pow(2).mean(dim=0)  # One loss per output dimension.
        ...         loss = scalarizer(losses)
        ...         optimizer.zero_grad()
        ...         loss.backward()
        ...         optimizer.step()
        ...     scalarizer.step()  # Roll the epoch history once, at the end of the epoch.

    .. note::
        DWA weights each value by the ratio of its losses over consecutive epochs, which the paper
        defines as a descending rate in the range :math:`(0, +\infty)`. The losses are therefore
        expected to keep a consistent, nonzero sign across epochs (they need not be positive).
    """

    def __init__(self, temperature: float = 2.0) -> None:
        if temperature <= 0.0:
            raise ValueError(
                f"Parameter `temperature` should be strictly positive. Found `temperature = "
                f"{temperature}`."
            )

        super().__init__()
        self.temperature = temperature
        self._loss_sum: Tensor | None = None
        self._n_batches: int = 0
        self._previous_averages: list[Tensor] = []

    def forward(self, values: Tensor, /) -> Tensor:
        weights = self._compute_weights(values)

        detached = values.detach()
        if self._loss_sum is None:
            self._loss_sum = detached.clone()
        elif self._loss_sum.shape != detached.shape:
            raise ValueError(
                f"The shape of `values` changed from {tuple(self._loss_sum.shape)} to "
                f"{tuple(detached.shape)} within an epoch. Call `reset()` before changing it."
            )
        else:
            self._loss_sum = self._loss_sum + detached
        self._n_batches += 1

        return (weights * values).sum()

    def step(self) -> None:
        """
        Finalizes the current epoch's average loss and rolls the history forward, discarding the
        average from two epochs ago. Should be called once at the end of each epoch.
        """

        if self._loss_sum is None:
            return
        average = self._loss_sum / self._n_batches
        self._previous_averages = [*self._previous_averages, average][-2:]
        self._loss_sum = None
        self._n_batches = 0

    def reset(self) -> None:
        self._loss_sum = None
        self._n_batches = 0
        self._previous_averages = []

    def _compute_weights(self, values: Tensor) -> Tensor:
        if len(self._previous_averages) < 2:
            return torch.ones_like(values)
        older = self._previous_averages[0]
        newer = self._previous_averages[1]
        if older.shape != values.shape:
            raise ValueError(
                f"The shape of `values` changed from {tuple(older.shape)} to "
                f"{tuple(values.shape)}. Call `reset()` before changing it."
            )
        rates = (newer / older).flatten()
        weights = softmax(rates / self.temperature, dim=0)
        return values.numel() * weights.reshape(values.shape)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(temperature={self.temperature})"
