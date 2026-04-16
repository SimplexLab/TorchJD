from abc import ABC, abstractmethod


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
