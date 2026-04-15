from abc import ABC, abstractmethod


class ResettableMixin(ABC):
    """Mixin that resettable classes should inherit from."""

    @abstractmethod
    def reset(self) -> None:
        """Resets the internal state."""
