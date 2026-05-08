from typing import TypeGuard

from torch import Tensor

# Note: we're using classes and inherittance instead of NewType because it's possible to have
# multiple inherittance but there is no type intersection. However, these classes should never be
# instantiated: they're only used for static type checking.


class Matrix(Tensor):
    """
    Tensor with exactly 2 dimensions.

    Common examples include the Jacobian matrix J of shape ``[m, n]``, where m is the number of
    objectives and n is the number of model parameters, and the Gramian of the Jacobian
    G = J J^T of shape ``[m, m]``.

    .. note::
        This class should never be instantiated. It is only used for static type checking.
    """


class PSDTensor(Tensor):
    """
    Tensor representing a quadratic form. The first half of its dimensions matches the reversed
    second half of its dimensions (e.g. shape=[4, 3, 3, 4]), and its reshaping into a matrix should
    be positive semi-definite.
    """


class PSDMatrix(PSDTensor, Matrix):
    """
    Positive semi-definite matrix.

    A common example is the Gramian of the Jacobian G = J J^T of shape ``[m, m]``, where J is a
    Jacobian matrix of shape ``[m, n]``.

    .. note::
        This class should never be instantiated. It is only used for static type checking.
    """


def is_matrix(t: Tensor) -> TypeGuard[Matrix]:
    return t.ndim == 2


def is_psd_tensor(t: Tensor) -> TypeGuard[PSDTensor]:
    half_dim = t.ndim // 2
    return t.ndim % 2 == 0 and t.shape[:half_dim] == t.shape[: half_dim - 1 : -1]
    # We do not check that t is PSD as it is expensive, but this must be checked in the tests of
    # every function that uses this TypeGuard by using `assert_is_psd_tensor`.


def is_psd_matrix(t: Tensor) -> TypeGuard[PSDMatrix]:
    return t.ndim == 2 and t.shape[0] == t.shape[1]
    # We do not check that t is PSD as it is expensive, but this must be checked in the tests of
    # every function that uses this TypeGuard, by using `assert_is_psd_matrix`.
