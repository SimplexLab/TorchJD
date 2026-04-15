from abc import ABC, abstractmethod


class Stateful(ABC):
    """Mixin adding a reset method."""

    @abstractmethod
    def reset(self) -> None:
        """Resets the internal state."""
