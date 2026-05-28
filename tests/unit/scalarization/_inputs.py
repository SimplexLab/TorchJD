from torch import Tensor
from utils.tensors import randn_, tensor_

scalar_input: Tensor = tensor_(7.0)
vector_input: Tensor = randn_(5)
matrix_input: Tensor = randn_(3, 4)
tensor_3d_input: Tensor = randn_(2, 3, 4)

all_inputs: list[Tensor] = [scalar_input, vector_input, matrix_input, tensor_3d_input]
