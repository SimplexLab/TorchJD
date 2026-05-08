"""
This module provides type annotation classes representing tensors with specific structural
properties.

:class:`Matrix` represents any 2D tensor. A common example in the context of Jacobian descent
is the Jacobian matrix J of shape ``[m, n]``, where m is the number of objectives and n is the
number of model parameters.

:class:`PSDMatrix` represents a symmetric positive semi-definite square matrix. A common
example is the Gramian of the Jacobian G = J J^T of shape ``[m, m]``.

.. note::
    :class:`Matrix` and :class:`PSDMatrix` extend :class:`~torch.Tensor` for type-checking
    purposes only and should never be directly instantiated.

>>> import torch
>>> # Jacobian matrix of shape [m, n] = [2, 3]
>>> J = torch.tensor([[-4., 1., 1.], [6., 1., 1.]])
>>> J.ndim
2
>>> # Gramian of the Jacobian, of shape [m, m] = [2, 2]
>>> G = J @ J.T
>>> G.shape
torch.Size([2, 2])
"""

from torchjd._linalg._matrix import Matrix, PSDMatrix

__all__ = ["Matrix", "PSDMatrix"]
