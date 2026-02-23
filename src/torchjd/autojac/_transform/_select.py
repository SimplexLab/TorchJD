from collections.abc import Set as AbstractSet

from torch import Tensor

from ._base import RequirementError, TensorDict, Transform


class Select(Transform):
    """
    Transform returning a subset of the provided TensorDict.

    :param keys: The keys that should be included in the returned subset.
    """

    def __init__(self, keys: AbstractSet[Tensor]) -> None:
        self.keys = keys

    def __call__(self, tensor_dict: TensorDict, /) -> TensorDict:
        output = {key: tensor_dict[key] for key in self.keys}
        return type(tensor_dict)(output)

    def check_keys(self, input_keys: set[Tensor], /) -> set[Tensor]:
        keys = set(self.keys)
        if not keys.issubset(input_keys):
            raise RequirementError(
                f"The input_keys should be a super set of the keys to select. Found input_keys "
                f"{input_keys} and keys to select {keys}.",
            )
        return keys
