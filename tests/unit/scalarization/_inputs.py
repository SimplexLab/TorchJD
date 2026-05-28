from torch import Tensor
from utils.tensors import randn_

shapes: list[list[int]] = [[], [5], [3, 4], [2, 3, 4]]
all_inputs: list[Tensor] = [randn_(shape) for shape in shapes]
