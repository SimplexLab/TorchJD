from abc import ABC, abstractmethod


class Resettable(ABC):
    """Mixin adding a reset method."""

    @abstractmethod
    def reset(self) -> None:
        """Resets the internal state."""
