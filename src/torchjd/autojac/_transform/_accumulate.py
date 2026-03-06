from torch import Tensor

from torchjd.autojac._accumulation import accumulate_grads, accumulate_jacs

from ._base import TensorDict, Transform


class AccumulateGrad(Transform):
    """
    Transform from Gradients to {} that accumulates gradients with respect to keys into their
    ``grad`` field.

    The Gradients are not cloned and may be modified in-place by subsequent accumulations, so they
    should not be used elsewhere.
    """

    def __call__(self, gradients: TensorDict, /) -> TensorDict:
        accumulate_grads(gradients.keys(), gradients.values())
        return {}

    def check_keys(self, _input_keys: set[Tensor], /) -> set[Tensor]:
        return set()


class AccumulateJac(Transform):
    """
    Transform from Jacobians to {} that accumulates jacobians with respect to keys into their
    ``jac`` field.

    The Jacobians are not cloned and may be modified in-place by subsequent accumulations, so they
    should not be used elsewhere.
    """

    def __call__(self, jacobians: TensorDict, /) -> TensorDict:
        accumulate_jacs(jacobians.keys(), jacobians.values())
        return {}

    def check_keys(self, _input_keys: set[Tensor], /) -> set[Tensor]:
        return set()
