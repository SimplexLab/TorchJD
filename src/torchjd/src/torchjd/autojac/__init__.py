"""
This package enables Jacobian descent, through the ``backward`` and ``mtl_backward`` functions, which
are meant to replace the call to ``torch.backward`` or ``loss.backward`` in gradient descent. To
combine the information of the Jacobian, an aggregator from the ``aggregation`` package has to be
used.
"""

from ._backward import backward
from ._jac import jac
from ._jac_to_grad import jac_to_grad
from ._mtl_backward import mtl_backward

__all__ = ["backward", "jac", "jac_to_grad", "mtl_backward"]
