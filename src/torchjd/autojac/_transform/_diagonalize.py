import torch
from torch import Tensor

from ._base import RequirementError, TensorDict, Transform
from ._ordered_set import OrderedSet


class Diagonalize(Transform):
    """
    Transform diagonalizing Gradients into Jacobians.

    The first dimension of the returned Jacobians will be equal to the total number of elements in
    the tensors of the input tensor dict. The exact behavior of the diagonalization is best
    explained by some examples.

    Example 1:
        The input is one tensor of shape [3] and of value [1 2 3].
        The output Jacobian will be:
        [[1 0 0]
         [0 2 0]
         [0 0 3]]

    Example 2:
        The input is one tensor of shape [2, 2] and of value [[4 5] [6 7]].
        The output Jacobian will be:
        [[[4 0] [0 0]]
         [[0 5] [0 0]]
         [[0 0] [6 0]]
         [[0 0] [0 7]]]

    Example 3:
        The input is two tensors, of shapes [3] and [2, 2] and of values [1 2 3] and [[4 5] [6 7]].
        If the key_order has the tensor of shape [3] appear first and the one of shape [2, 2] appear
        second, the output Jacobians will be:
        [[1 0 0]
         [0 2 0]
         [0 0 3]
         [0 0 0]
         [0 0 0]
         [0 0 0]
         [0 0 0]] and
        [[[0 0] [0 0]]
         [[0 0] [0 0]]
         [[0 0] [0 0]]
         [[4 0] [0 0]]
         [[0 5] [0 0]]
         [[0 0] [6 0]]
         [[0 0] [0 7]]]

    :param key_order: The order in which the keys are represented in the rows of the output
        Jacobians.
    """

    def __init__(self, key_order: OrderedSet[Tensor]):
        self.key_order = key_order
        self.indices: list[tuple[int, int]] = []
        begin = 0
        for tensor in self.key_order:
            end = begin + tensor.numel()
            self.indices.append((begin, end))
            begin = end

    def __call__(self, tensors: TensorDict, /) -> TensorDict:
        flattened_considered_values = [tensors[key].reshape([-1]) for key in self.key_order]
        diagonal_matrix = torch.cat(flattened_considered_values).diag()
        diagonalized_tensors = {
            key: diagonal_matrix[:, begin:end].reshape((-1, *key.shape))
            for (begin, end), key in zip(self.indices, self.key_order, strict=True)
        }
        return diagonalized_tensors

    def check_keys(self, input_keys: set[Tensor], /) -> set[Tensor]:
        if not set(self.key_order) == input_keys:
            raise RequirementError(
                f"The input_keys must match the key_order. Found input_keys {input_keys} and"
                f"key_order {self.key_order}.",
            )
        return input_keys
