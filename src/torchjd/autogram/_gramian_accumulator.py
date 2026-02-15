from torchjd._linalg import PSDMatrix


class GramianAccumulator:
    """
    Efficiently accumulates the Gramian of the Jacobian during reverse-mode differentiation.

    Jacobians from multiple graph paths to the same parameter are first summed to obtain the full
    Jacobian w.r.t. a parameter, then its Gramian is computed and accumulated, over parameters, into
    the total Gramian matrix. Intermediate matrices are discarded immediately to save memory.
    """

    def __init__(self) -> None:
        self._gramian: PSDMatrix | None = None

    def reset(self) -> None:
        self._gramian = None

    def accumulate_gramian(self, gramian: PSDMatrix) -> None:
        if self._gramian is not None:
            self._gramian.add_(gramian)
        else:
            self._gramian = gramian

    @property
    def gramian(self) -> PSDMatrix | None:
        """
        Get the Gramian matrix accumulated so far.

        :returns: Accumulated Gramian matrix of shape (batch_size, batch_size) or None if nothing
            was accumulated yet.
        """

        return self._gramian
