from abc import ABC, abstractmethod


class ResettableMixin(ABC):
    """Class implementing a reset method."""

    @abstractmethod
    def reset(self) -> None:
        """Resets the internal state."""
