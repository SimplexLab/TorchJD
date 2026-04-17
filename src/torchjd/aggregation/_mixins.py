from abc import ABC, abstractmethod

import torch


class Stateful(ABC):
    r"""
    Mixin for stateful mappings.

    A maping implements `Stateful` **if and only if** its behavior depends on an internal
    state.

    Formally, a stateless mapping is a function :math:`f : x \mapsto y` whereas a stateful
    maping is a transition map :math:`A : (x, s) \mapsto (y, s')` where :math:`s` is the
    internal state, :math:`s'` the updated state, and :math:`y` the output.
    There exists an initial state :math:`s_0`, and the method `reset()` restores the state to
    :math:`s_0`. A `Stateful` mapping must be constructed with the intial state :math:`s_0`.
    """

    @abstractmethod
    def reset(self) -> None:
        """Resets the internal state :math:`s_0`."""


class StochasticState(Stateful):
    r"""
    State respresenting stochasticity.

    Internally, a ``StochasticState`` mapping holds a :class:`torch.Generator` that serves as an
    independent random number stream.

    :param seed: Seed for the internal :class:`torch.Generator`. If ``None``, a seed is drawn
        from the global PyTorch RNG to fork an independent stream.
    :param generator: An existing :class:`torch.Generator` to share, typically from a companion
        :class:`StochasticState` instance. Mutually exclusive with ``seed``.
    """

    def __init__(self, seed: int | None = None, generator: torch.Generator | None = None) -> None:
        if generator is not None and seed is not None:
            raise ValueError("Parameters `seed` and `generator` are mutually exclusive.")
        if generator is not None:
            self.generator = generator
        else:
            self.generator = torch.Generator()
            if seed is None:
                seed = int(torch.randint(0, 2**62, size=(1,), dtype=torch.int64).item())
            self.generator.manual_seed(seed)
        self._initial_rng_state = self.generator.get_state()

    def reset(self) -> None:
        """Resets the random number generator to its initial state."""
        self.generator.set_state(self._initial_rng_state)
