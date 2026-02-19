from functools import partial
from typing import Generic, TypeVar

import torch
import torchvision
from settings import DEVICE, DTYPE
from torch import Tensor, nn
from torch.nn import Flatten, ReLU
from torch.utils._pytree import PyTree

from utils.contexts import fork_rng

_T = TypeVar("_T", bound=nn.Module)


class ModuleFactory(Generic[_T]):
    def __init__(self, architecture: type[_T], *args, **kwargs) -> None:
        self.architecture: type[_T] = architecture
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> _T:
        with fork_rng(seed=0):
            return self.architecture(*self.args, **self.kwargs).to(device=DEVICE, dtype=DTYPE)

    def __str__(self) -> str:
        args_string = ", ".join([str(arg) for arg in self.args])
        kwargs_string = ", ".join([f"{key}={value}" for key, value in self.kwargs.items()])
        optional_comma = "" if args_string == "" or kwargs_string == "" else ", "
        return f"{self.architecture.__name__}({args_string}{optional_comma}{kwargs_string})"


class ShapedModule(nn.Module):
    """Module class that guarantees subclasses to have INPUT_SHAPES and OUTPUT_SHAPES attributes."""

    INPUT_SHAPES: PyTree  # meant to be overridden
    OUTPUT_SHAPES: PyTree  # meant to be overridden

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if getattr(cls, "INPUT_SHAPES", None) is None:
            raise TypeError(f"{cls.__name__} must define INPUT_SHAPES")
        if getattr(cls, "OUTPUT_SHAPES", None) is None:
            raise TypeError(f"{cls.__name__} must define OUTPUT_SHAPES")


def get_in_out_shapes(module: nn.Module) -> tuple[PyTree, PyTree]:
    if isinstance(module, ShapedModule):
        return module.INPUT_SHAPES, module.OUTPUT_SHAPES

    if isinstance(module, nn.BatchNorm2d | nn.InstanceNorm2d):
        HEIGHT = 6  # Arbitrary choice
        WIDTH = 6  # Arbitrary choice
        shape = (module.num_features, HEIGHT, WIDTH)
        return shape, shape

    raise ValueError("Unknown input / output shapes of module", module)


class OverlyNested(ShapedModule):
    """Model that contains many unnecessary levels of nested modules."""

    INPUT_SHAPES = (9,)
    OUTPUT_SHAPES = (14,)

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Sequential(
                nn.Sequential(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Linear(9, 10),
                            ReLU(),
                        ),
                        nn.Linear(10, 11),
                    ),
                    ReLU(),
                    nn.Linear(11, 12),
                ),
                ReLU(),
                nn.Linear(12, 13),
                ReLU(),
            ),
            nn.Linear(13, 14),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.seq(input)


class MultiInputSingleOutput(ShapedModule):
    """Module that takes two inputs and returns one output."""

    INPUT_SHAPES = ((50,), (50,))
    OUTPUT_SHAPES = (60,)

    def __init__(self) -> None:
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn(50, 60))
        self.matrix2 = nn.Parameter(torch.randn(50, 60))

    def forward(self, inputs: tuple[Tensor, Tensor]) -> Tensor:
        input1, input2 = inputs
        output = input1 @ self.matrix1 + input2 @ self.matrix2
        return output


class MultiInputMultiOutput(ShapedModule):
    """Module that takes two inputs and returns two outputs that each depend on both inputs."""

    INPUT_SHAPES = ((50,), (50,))
    OUTPUT_SHAPES = ((60,), (70,))

    def __init__(self) -> None:
        super().__init__()
        self.matrix1_1 = nn.Parameter(torch.randn(50, 60))
        self.matrix2_1 = nn.Parameter(torch.randn(50, 60))
        self.matrix1_2 = nn.Parameter(torch.randn(50, 70))
        self.matrix2_2 = nn.Parameter(torch.randn(50, 70))

    def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        input1, input2 = inputs
        output1 = input1 @ self.matrix1_1 + input2 @ self.matrix2_1
        output2 = input1 @ self.matrix1_2 + input2 @ self.matrix2_2
        return output1, output2


class SingleInputPyTreeOutput(ShapedModule):
    """Module taking a single input and returning a complex PyTree of tensors as output."""

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = {
        "first": ((50,), [(60,), (70,)]),
        "second": (80,),
        "third": ([((90,),)],),
    }

    def __init__(self) -> None:
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn(50, 50))
        self.matrix2 = nn.Parameter(torch.randn(50, 60))
        self.matrix3 = nn.Parameter(torch.randn(50, 70))
        self.matrix4 = nn.Parameter(torch.randn(50, 80))
        self.matrix5 = nn.Parameter(torch.randn(50, 90))

    def forward(self, input: Tensor) -> PyTree:
        return {
            "first": (input @ self.matrix1, [input @ self.matrix2, input @ self.matrix3]),
            "second": input @ self.matrix4,
            "third": ([(input @ self.matrix5,)],),
        }


class PyTreeInputSingleOutput(ShapedModule):
    """Module taking a complex PyTree of tensors as input and returning a single output."""

    INPUT_SHAPES = {
        "one": [((10,), [(20,), (30,)]), (12,)],
        "two": (14,),
    }
    OUTPUT_SHAPES = (350,)

    def __init__(self) -> None:
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn(10, 50))
        self.matrix2 = nn.Parameter(torch.randn(20, 60))
        self.matrix3 = nn.Parameter(torch.randn(30, 70))
        self.matrix4 = nn.Parameter(torch.randn(12, 80))
        self.matrix5 = nn.Parameter(torch.randn(14, 90))

    def forward(self, inputs: PyTree) -> Tensor:
        input1 = inputs["one"][0][0]
        input2 = inputs["one"][0][1][0]
        input3 = inputs["one"][0][1][1]
        input4 = inputs["one"][1]
        input5 = inputs["two"]

        output1 = input1 @ self.matrix1
        output2 = input2 @ self.matrix2
        output3 = input3 @ self.matrix3
        output4 = input4 @ self.matrix4
        output5 = input5 @ self.matrix5
        output = torch.concatenate([output1, output2, output3, output4, output5], dim=1)

        return output


class PyTreeInputPyTreeOutput(ShapedModule):
    """
    Module taking a complex PyTree of tensors as input and returning a complex PyTree of tensors as
    output.
    """

    INPUT_SHAPES = {
        "one": [((10,), [(20,), (30,)]), (12,)],
        "two": (14,),
    }

    OUTPUT_SHAPES = {
        "first": ((50,), [(60,), (70,)]),
        "second": (80,),
        "third": ([((90,),)],),
    }

    def __init__(self) -> None:
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn(10, 50))
        self.matrix2 = nn.Parameter(torch.randn(20, 60))
        self.matrix3 = nn.Parameter(torch.randn(30, 70))
        self.matrix4 = nn.Parameter(torch.randn(12, 80))
        self.matrix5 = nn.Parameter(torch.randn(14, 90))

    def forward(self, inputs: PyTree) -> PyTree:
        input1 = inputs["one"][0][0]
        input2 = inputs["one"][0][1][0]
        input3 = inputs["one"][0][1][1]
        input4 = inputs["one"][1]
        input5 = inputs["two"]

        return {
            "first": (input1 @ self.matrix1, [input2 @ self.matrix2, input3 @ self.matrix3]),
            "second": input4 @ self.matrix4,
            "third": ([(input5 @ self.matrix5,)],),
        }


class SimpleBranched(ShapedModule):
    """Model with one input and two branches that rejoin into one output."""

    INPUT_SHAPES = (9,)
    OUTPUT_SHAPES = (16,)

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(9, 13)
        self.fc1 = nn.Linear(13, 14)
        self.fc2 = nn.Linear(14, 15)
        self.fc3 = nn.Linear(13, 15)
        self.fc4 = nn.Linear(15, 16)

    def forward(self, input: Tensor) -> Tensor:
        common_input = self.relu(self.fc0(input))
        branch1 = self.fc2(self.relu(self.fc1(common_input)))
        branch2 = self.fc3(common_input)
        output = self.fc4(self.relu(branch1 + branch2))
        return output


class MISOBranched(ShapedModule):
    """
    Model taking a single input, branching it using a MultiInputSingleOutputModule, and returning
    its output.
    """

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = MultiInputSingleOutput.OUTPUT_SHAPES

    def __init__(self) -> None:
        super().__init__()
        self.miso = MultiInputSingleOutput()

    def forward(self, input: Tensor) -> Tensor:
        return self.miso((input, input))


class MIMOBranched(ShapedModule):
    """
    Model taking a single input, branching it using a MultiInputMultiOutputModule, and returning
    the concatenation of its outputs.
    """

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (130,)

    def __init__(self) -> None:
        super().__init__()
        self.mimo = MultiInputMultiOutput()

    def forward(self, input: Tensor) -> Tensor:
        return torch.concatenate(list(self.mimo((input, input))), dim=1)


class SIPOBranched(ShapedModule):
    """
    Model taking a single input, branching it using a SingleInputPyTreeOutput, and returning the
    concatenation of its outputs.
    """

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (350,)

    def __init__(self) -> None:
        super().__init__()
        self.sipo = SingleInputPyTreeOutput()

    def forward(self, input: Tensor) -> Tensor:
        first, second, third = self.sipo(input).values()
        output1, output23 = first
        output2, output3 = output23
        output4 = second
        output5 = third[0][0][0]

        return torch.concatenate([output1, output2, output3, output4, output5], dim=1)


class PISOBranched(ShapedModule):
    """
    Model taking a single input, splitting it, branching it using a PyTreeInputSingleOutput, and
    returning its output.
    """

    INPUT_SHAPES = (86,)
    OUTPUT_SHAPES = (350,)

    def __init__(self) -> None:
        super().__init__()
        self.piso = PyTreeInputSingleOutput()

    def forward(self, input: Tensor) -> Tensor:
        input1 = input[:, 0:10]
        input2 = input[:, 10:30]
        input3 = input[:, 30:60]
        input4 = input[:, 60:72]
        input5 = input[:, 72:86]

        pytree_input = {
            "one": [(input1, [input2, input3]), input4],
            "two": input5,
        }

        return self.piso(pytree_input)


class PIPOBranched(ShapedModule):
    """
    Model taking a single input, splitting it, branching it using a PyTreeInputPyTreeOutput, and
    returning the concatenation of its outputs.
    """

    INPUT_SHAPES = (86,)
    OUTPUT_SHAPES = (350,)

    def __init__(self) -> None:
        super().__init__()
        self.pipo = PyTreeInputPyTreeOutput()

    def forward(self, input: Tensor) -> Tensor:
        input1 = input[:, 0:10]
        input2 = input[:, 10:30]
        input3 = input[:, 30:60]
        input4 = input[:, 60:72]
        input5 = input[:, 72:86]

        pytree_input = {
            "one": [(input1, [input2, input3]), input4],
            "two": input5,
        }

        pytree_output = self.pipo(pytree_input)

        first, second, third = pytree_output.values()
        output1, output23 = first
        output2, output3 = output23
        output4 = second
        output5 = third[0][0][0]

        return torch.concatenate([output1, output2, output3, output4, output5], dim=1)


class WithNoTensorOutput(ShapedModule):
    """
    Model that has modules that return no tensor. In reality, such a module could be used to gather
    some stats or print something, even though this is not standard.
    """

    INPUT_SHAPES = (27,)
    OUTPUT_SHAPES = (10,)

    class _NoneOutput(nn.Module):
        def __init__(self, shape: tuple[int, ...]) -> None:
            super().__init__()
            self.matrix = nn.Parameter(torch.randn(shape))

        def forward(self, _: PyTree) -> None:
            pass

    class _NonePyTreeOutput(nn.Module):
        def __init__(self, shape: tuple[int, ...]) -> None:
            super().__init__()
            self.matrix = nn.Parameter(torch.randn(shape))

        def forward(self, _: PyTree) -> PyTree:
            return {"one": [None, ()], "two": None}

    class _EmptyTupleOutput(nn.Module):
        def __init__(self, shape: tuple[int, ...]) -> None:
            super().__init__()
            self.matrix = nn.Parameter(torch.randn(shape))

        def forward(self, _: PyTree) -> tuple:
            return ()

    class _EmptyPytreeOutput(nn.Module):
        def __init__(self, shape: tuple[int, ...]) -> None:
            super().__init__()
            self.matrix = nn.Parameter(torch.randn(shape))

        def forward(self, _: PyTree) -> PyTree:
            return {"one": [(), ()], "two": [[], []]}

    def __init__(self) -> None:
        super().__init__()
        self.none_output = self._NoneOutput((27, 10))
        self.none_pytree_output = self._NonePyTreeOutput((27, 10))
        self.empty_tuple_output = self._EmptyTupleOutput((27, 10))
        self.empty_pytree_output = self._EmptyPytreeOutput((27, 10))
        self.linear = nn.Linear(27, 10)

    def forward(self, input: Tensor) -> Tensor:
        _ = self.none_output(input)
        _ = self.none_pytree_output(input)
        _ = self.empty_tuple_output(input)
        _ = self.empty_pytree_output(input)
        return self.linear(input)


class IntraModuleParamReuse(ShapedModule):
    """Module that reuses the same nn.Parameter for two computations directly inside of it."""

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (10,)

    def __init__(self) -> None:
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(50, 10))

    def forward(self, input: Tensor) -> Tensor:
        return input @ self.matrix + (input**2 / 5.0) @ self.matrix


class InterModuleParamReuse(ShapedModule):
    """Model that has two modules that both use the same nn.Parameter."""

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (10,)

    class _MatMulModule(nn.Module):
        """
        Simple module for matrix multiplication, that takes the parameter in its __init__ method so
        that this parameter can be used in other modules too.
        """

        def __init__(self, matrix: nn.Parameter) -> None:
            super().__init__()
            self.matrix = matrix

        def forward(self, input: Tensor) -> Tensor:
            return input @ self.matrix

    def __init__(self) -> None:
        super().__init__()
        matrix = nn.Parameter(torch.randn(50, 10))
        self.module1 = self._MatMulModule(matrix)
        self.module2 = self._MatMulModule(matrix)

    def forward(self, input: Tensor) -> Tensor:
        return self.module1(input) + self.module2(input**2 / 5.0)


class ModuleReuse(ShapedModule):
    """Model that uses the same module for two computations."""

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (10,)

    def __init__(self) -> None:
        super().__init__()
        self.module = nn.Linear(50, 10)

    def forward(self, input: Tensor) -> Tensor:
        return self.module(input) + self.module(input**2 / 5.0)


class SomeUnusedParam(ShapedModule):
    """Module that has an unused param and a used param."""

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (10,)

    def __init__(self) -> None:
        super().__init__()
        self.unused_param = nn.Parameter(torch.randn(50, 10))
        self.matrix = nn.Parameter(torch.randn(50, 10))

    def forward(self, input: Tensor) -> Tensor:
        return input @ self.matrix


class SomeFrozenParam(ShapedModule):
    """
    Module that has a frozen param (requires_grad=False) and a non-frozen param
    (requires_grad=True).
    """

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (10,)

    def __init__(self) -> None:
        super().__init__()
        self.frozen_param = nn.Parameter(torch.randn(50, 10), requires_grad=False)
        self.matrix = nn.Parameter(torch.randn(50, 10))

    def forward(self, input: Tensor) -> Tensor:
        return input @ self.matrix + (input**2 / 5.0) @ self.frozen_param


class WithSomeFrozenModule(ShapedModule):
    """
    Model that has a module whose params are all frozen, and a module whose params are not frozen.
    """

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (10,)

    def __init__(self) -> None:
        super().__init__()
        self.non_frozen = nn.Linear(50, 10)
        self.all_frozen = nn.Linear(50, 10)
        self.all_frozen.requires_grad_(False)

    def forward(self, input: Tensor) -> Tensor:
        return self.all_frozen(input) + self.non_frozen(input**2 / 5.0)


class RequiresGradOfSchrodinger(ShapedModule):
    """
    Model that contains a module whose output will not require grad despite containing a param that
    requires grad (so it will be hooked). The final output of the model will require grad, though,
    because another normal module is used on the output of the first module.
    """

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (3,)

    class SomeFrozenParamAndUnusedTrainableParam(ShapedModule):
        """
        Module that has a frozen param (requires_grad=False) and a non-frozen param (requires_grad=
        True), but the non-frozen param is also unused.
        """

        INPUT_SHAPES = (50,)
        OUTPUT_SHAPES = (10,)

        def __init__(self) -> None:
            super().__init__()
            self.frozen_param = nn.Parameter(torch.randn(50, 10), requires_grad=False)
            self.non_frozen_param = nn.Parameter(torch.randn(50, 10))

        def forward(self, input: Tensor) -> Tensor:
            return input @ self.frozen_param

    def __init__(self) -> None:
        super().__init__()
        self.weird_module = self.SomeFrozenParamAndUnusedTrainableParam()
        self.normal_module = nn.Linear(10, 3)

    def forward(self, input: Tensor) -> Tensor:
        return self.normal_module(self.weird_module(input))


class MultiOutputWithFrozenBranch(ShapedModule):
    """
    Module that has two outputs: one comes from a frozen parameter, so it will only require grad
    if the input requires grad, and one comes from a non-frozen parameter.
    """

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = ((10,), (10,))

    def __init__(self) -> None:
        super().__init__()
        self.frozen_param = nn.Parameter(torch.randn(50, 10), requires_grad=False)
        self.matrix = nn.Parameter(torch.randn(50, 10))

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        return (input**2 / 5.0) @ self.frozen_param, input @ self.matrix


class WithBuffered(ShapedModule):
    """Model using a module that has a buffer."""

    INPUT_SHAPES = (27,)
    OUTPUT_SHAPES = (10,)

    class _Buffered(nn.Module):
        buffer: Tensor

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("buffer", torch.tensor(1.5))

        def forward(self, input: Tensor) -> Tensor:
            return input * self.buffer

    def __init__(self) -> None:
        super().__init__()
        self.module_with_buffer = self._Buffered()
        self.linear = nn.Linear(27, 10)

    def forward(self, input: Tensor) -> Tensor:
        return self.linear(self.module_with_buffer(input))


class Randomness(ShapedModule):
    """Module with some randomness and a direct parameter."""

    INPUT_SHAPES = (9,)
    OUTPUT_SHAPES = (10,)

    def __init__(self) -> None:
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(9, 10))

    def forward(self, input: Tensor) -> Tensor:
        noise = torch.zeros_like(input)
        noise.normal_()
        return (input * noise) @ self.matrix


class WithSideEffect(ShapedModule):
    """Module with a side-effect during the forward."""

    INPUT_SHAPES = (9,)
    OUTPUT_SHAPES = (10,)

    def __init__(self) -> None:
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(9, 10))
        self.register_buffer("buffer", torch.zeros((9,)))

    def forward(self, input: Tensor) -> Tensor:
        self.buffer = self.buffer + 1.0
        return (input + self.buffer) @ self.matrix


class SomeUnusedOutput(ShapedModule):
    """
    Model that contains a module whose output is not used in the computation graph leading to the
    model outputs.
    """

    INPUT_SHAPES = (9,)
    OUTPUT_SHAPES = (10,)

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(9, 12)
        self.linear2 = nn.Linear(9, 10)

    def forward(self, input: Tensor) -> Tensor:
        _ = self.linear1(input)
        output = self.linear2(input)
        return output


class Ndim0Output(ShapedModule):
    """Simple model whose output is a scalar."""

    INPUT_SHAPES = (5,)
    OUTPUT_SHAPES = ()

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(5, 1)

    def forward(self, input: Tensor) -> Tensor:
        return self.linear(input).squeeze(dim=1) / 5.0


class Ndim1Output(ShapedModule):
    """Simple model whose output is a vector."""

    INPUT_SHAPES = (5,)
    OUTPUT_SHAPES = (3,)

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(5, 3)

    def forward(self, input: Tensor) -> Tensor:
        return self.linear(input) / 5.0


class Ndim2Output(ShapedModule):
    """Simple model whose output is a matrix."""

    INPUT_SHAPES = (5,)
    OUTPUT_SHAPES = (2, 3)

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(5, 3)
        self.linear2 = nn.Linear(5, 3)

    def forward(self, input: Tensor) -> Tensor:
        return torch.stack([self.linear1(input), self.linear2(input)], dim=1)


class Ndim3Output(ShapedModule):
    """Simple model whose output is a tensor of 3 dimensions."""

    INPUT_SHAPES = (6,)
    OUTPUT_SHAPES = (2, 3, 4)

    def __init__(self) -> None:
        super().__init__()
        self.tensor = nn.Parameter(torch.randn(6, 2, 3, 4))

    def forward(self, input: Tensor) -> Tensor:
        return torch.einsum("bi,icde->bcde", input, self.tensor)


class Ndim4Output(ShapedModule):
    """Simple model whose output is a tensor of 4 dimensions."""

    INPUT_SHAPES = (6,)
    OUTPUT_SHAPES = (2, 3, 4, 5)

    def __init__(self) -> None:
        super().__init__()
        self.tensor = nn.Parameter(torch.randn(6, 2, 3, 4, 5))

    def forward(self, input: Tensor) -> Tensor:
        return torch.einsum("bi,icdef->bcdef", input, self.tensor)


class WithRNN(ShapedModule):
    """Simple model containing an RNN module."""

    INPUT_SHAPES = (20, 8)  # Size 20, dim input_size (8)
    OUTPUT_SHAPES = (20, 5)  # Size 20, dim hidden_size (5)

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=8, hidden_size=5, batch_first=True)

    def forward(self, input: Tensor) -> Tensor:
        output, _ = self.rnn(input)
        return output


class WithDropout(ShapedModule):
    """Simple model containing Dropout layers."""

    INPUT_SHAPES = (3, 6, 6)
    OUTPUT_SHAPES = (3, 4, 4)

    def __init__(self) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, input: Tensor) -> Tensor:
        return self.dropout(self.conv2d(self.dropout(input)))


class ModelUsingSubmoduleParamsDirectly(ShapedModule):
    """
    Model that uses its submodule's parameters directly and that does not call its submodule's
    forward.
    """

    INPUT_SHAPES = (2,)
    OUTPUT_SHAPES = (3,)

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3)

    def forward(self, input: Tensor) -> Tensor:
        return input @ self.linear.weight.T + self.linear.bias


class ModelAlsoUsingSubmoduleParamsDirectly(ShapedModule):
    """
    Model that uses its submodule's parameters directly but that also calls its submodule's forward.
    """

    INPUT_SHAPES = (2,)
    OUTPUT_SHAPES = (3,)

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3)

    def forward(self, input: Tensor) -> Tensor:
        return input @ self.linear.weight.T + self.linear.bias + self.linear(input)


class _WithStringArg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(2, 3))

    def forward(self, s: str, input: Tensor) -> Tensor:
        if s == "two":
            return input @ self.matrix * 2.0
        return input @ self.matrix


class WithModuleWithStringArg(ShapedModule):
    """Model containing a module that has a string argument."""

    INPUT_SHAPES = (2,)
    OUTPUT_SHAPES = (3,)

    def __init__(self) -> None:
        super().__init__()
        self.with_string_arg = _WithStringArg()

    def forward(self, input: Tensor) -> Tensor:
        return self.with_string_arg("two", input)


class WithModuleWithStringKwarg(ShapedModule):
    """Model calling its submodule's forward with a string and a tensor as keyword arguments."""

    INPUT_SHAPES = (2,)
    OUTPUT_SHAPES = (3,)

    def __init__(self) -> None:
        super().__init__()
        self.with_string_arg = _WithStringArg()

    def forward(self, input: Tensor) -> Tensor:
        return self.with_string_arg(s="two", input=input)


class _WithHybridPyTreeArg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m0 = nn.Parameter(torch.randn(3, 3))
        self.m1 = nn.Parameter(torch.randn(4, 3))
        self.m2 = nn.Parameter(torch.randn(5, 3))
        self.m3 = nn.Parameter(torch.randn(6, 3))

    def forward(self, input: PyTree) -> Tensor:
        t0 = input["one"][0][0]
        t1 = input["one"][0][1]
        t2 = input["one"][1]
        t3 = input["two"]

        c0 = input["one"][0][3]
        c1 = input["one"][0][4][0]
        c2 = input["one"][2]
        c3 = input["three"]

        return c0 * t0 @ self.m0 + c1 * t1 @ self.m1 + c2 * t2 @ self.m2 + c3 * t3 @ self.m3


class WithModuleWithHybridPyTreeArg(ShapedModule):
    """
    Model containing a module that has a PyTree argument containing a mix of tensor and non-tensor
    leaves.
    """

    INPUT_SHAPES = (10,)
    OUTPUT_SHAPES = (3,)

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 18)
        self.with_string_arg = _WithHybridPyTreeArg()

    def forward(self, input: Tensor) -> Tensor:
        input = self.linear(input)

        t0, t1, t2, t3 = input[:, 0:3], input[:, 3:7], input[:, 7:12], input[:, 12:18]

        tree = {
            "zero": "unused",
            "one": [(t0, t1, "unused", 0.2, [0.3, "unused"]), t2, 0.4, "unused"],
            "two": t3,
            "three": 0.5,
        }

        return self.with_string_arg(tree)


class WithModuleWithHybridPyTreeKwarg(ShapedModule):
    """
    Model calling its submodule's forward with a PyTree keyword argument containing a mix of tensors
    and non-tensor values.
    """

    INPUT_SHAPES = (10,)
    OUTPUT_SHAPES = (3,)

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 18)
        self.with_string_arg = _WithHybridPyTreeArg()

    def forward(self, input: Tensor) -> Tensor:
        input = self.linear(input)

        t0, t1, t2, t3 = input[:, 0:3], input[:, 3:7], input[:, 7:12], input[:, 12:18]

        tree = {
            "zero": "unused",
            "one": [(t0, t1, "unused", 0.2, [0.3, "unused"]), t2, 0.4, "unused"],
            "two": t3,
            "three": 0.5,
        }

        return self.with_string_arg(input=tree)


class WithModuleWithStringOutput(ShapedModule):
    """Model containing a module that has a string output."""

    INPUT_SHAPES = (2,)
    OUTPUT_SHAPES = (3,)

    class WithStringOutput(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.matrix = nn.Parameter(torch.randn(2, 3))

        def forward(self, input: Tensor) -> tuple[str, Tensor]:
            return "test", input @ self.matrix

    def __init__(self) -> None:
        super().__init__()
        self.with_string_output = self.WithStringOutput()

    def forward(self, input: Tensor) -> Tensor:
        _, output = self.with_string_output(input)
        return output


class WithMultiHeadAttention(ShapedModule):
    """Module containing a MultiheadAttention layer."""

    INPUT_SHAPES = ((20, 8), (10, 9), (10, 11))
    OUTPUT_SHAPES = (20, 8)

    def __init__(self) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=8,
            num_heads=2,
            dropout=0.0,
            batch_first=True,
            kdim=9,
            vdim=11,
        )

    def forward(self, input: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        query, key, value = input
        attn_output, _ = self.mha(query, key, value)
        return attn_output


class WithTransformer(ShapedModule):
    """Module containing a Transformer."""

    INPUT_SHAPES = ((10, 8), (20, 8))
    OUTPUT_SHAPES = (20, 8)

    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=8,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=32,
            batch_first=True,
            dropout=0.0,
        )

    def forward(self, input: tuple[Tensor, Tensor]) -> Tensor:
        src, tgt = input
        return self.transformer(src, tgt)


class WithTransformerLarge(ShapedModule):
    """Module containing a large Transformer."""

    INPUT_SHAPES = ((10, 512), (20, 512))
    OUTPUT_SHAPES = (20, 512)

    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Transformer(
            batch_first=True,
            dropout=0.0,
        )

    def forward(self, input: tuple[Tensor, Tensor]) -> Tensor:
        src, tgt = input
        return self.transformer(src, tgt)


class FreeParam(ShapedModule):
    """
    Model that contains a free (i.e. not contained in a submodule) parameter, that is used at the
    beginning of the forward pass.
    """

    INPUT_SHAPES = (15,)
    OUTPUT_SHAPES = (80,)

    def __init__(self) -> None:
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(15, 16))  # Free parameter
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(16, 50)
        self.linear2 = nn.Linear(50, 60)
        self.linear3 = nn.Linear(60, 70)
        self.linear4 = nn.Linear(70, 80)

    def forward(self, input: Tensor) -> Tensor:
        output = self.relu(input @ self.matrix)
        output = self.relu(self.linear1(output))
        output = self.relu(self.linear2(output))
        output = self.relu(self.linear3(output))
        output = self.linear4(output)
        return output


class NoFreeParam(ShapedModule):
    """
    Same model as FreeParam but with the free parameter contained inside a submodule. Useful for
    speed comparison with FreeParam.
    """

    INPUT_SHAPES = (15,)
    OUTPUT_SHAPES = (80,)

    def __init__(self) -> None:
        super().__init__()
        self.linear0 = nn.Linear(15, 16, bias=False)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(16, 50)
        self.linear2 = nn.Linear(50, 60)
        self.linear3 = nn.Linear(60, 70)
        self.linear4 = nn.Linear(70, 80)

    def forward(self, input: Tensor) -> Tensor:
        output = self.relu(self.linear0(input))
        output = self.relu(self.linear1(output))
        output = self.relu(self.linear2(output))
        output = self.relu(self.linear3(output))
        output = self.linear4(output)
        return output


class Cifar10Model(ShapedModule):
    """
    Architecture for image classification on the CIFAR-10 dataset, similar to what we used in
    https://arxiv.org/pdf/2406.16232.
    """

    class Body(ShapedModule):
        """Convolutional feature extractor coming at the beginning of the Cifar10 model."""

        INPUT_SHAPES = (3, 32, 32)
        OUTPUT_SHAPES = (1024,)

        def __init__(self) -> None:
            super().__init__()
            layers = [
                nn.Conv2d(3, 32, 3),
                ReLU(),
                nn.Conv2d(32, 64, 3, groups=32),
                nn.Sequential(nn.MaxPool2d(2), ReLU()),
                nn.Conv2d(64, 64, 3, groups=64),
                nn.Sequential(nn.MaxPool2d(3), ReLU(), Flatten()),
            ]
            self.seq = nn.Sequential(*layers)

        def forward(self, input: Tensor) -> Tensor:
            return self.seq(input)

    class Head(ShapedModule):
        """Multi-Layer Perceptron classifier coming at the end of the Cifar10 model."""

        INPUT_SHAPES = (1024,)
        OUTPUT_SHAPES = (10,)

        def __init__(self) -> None:
            super().__init__()
            layers = [
                nn.Linear(1024, 128),
                ReLU(),
                nn.Linear(128, 10),
            ]
            self.seq = nn.Sequential(*layers)

        def forward(self, input: Tensor) -> Tensor:
            return self.seq(input)

    INPUT_SHAPES = Body.INPUT_SHAPES
    OUTPUT_SHAPES = Head.OUTPUT_SHAPES

    def __init__(self) -> None:
        super().__init__()
        self.body = self.Body()
        self.head = self.Head()

    def forward(self, input: Tensor) -> Tensor:
        features = self.body(input)
        output = self.head(features)
        return output


class AlexNet(ShapedModule):
    """
    AlexNet. Note that most of its parameters are in the last two linear layers (4096 * 4096 and
    4096 * 1000, not counting the biases). Also note that AlexNet has some (properly isolated)
    randomness due to having dropout layers.
    """

    INPUT_SHAPES = (3, 224, 224)
    OUTPUT_SHAPES = (1000,)

    def __init__(self) -> None:
        super().__init__()
        self.alexnet = torchvision.models.alexnet()

    def forward(self, input: Tensor) -> Tensor:
        return self.alexnet(input)


class InstanceNormResNet18(ShapedModule):
    """
    ResNet18 with BatchNorm2d layers replaced by InstanceNorm2d without tracking of running
    stats.
    """

    INPUT_SHAPES = (3, 224, 224)
    OUTPUT_SHAPES = (1000,)

    def __init__(self) -> None:
        super().__init__()
        self.resnet18 = torchvision.models.resnet18(
            norm_layer=partial(nn.InstanceNorm2d, track_running_stats=False, affine=True),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.resnet18(input) / 5.0


class GroupNormMobileNetV3Small(ShapedModule):
    """MobileNetV3Small with BatchNorm2d layers replaced by GroupNorm layers."""

    INPUT_SHAPES = (3, 224, 224)
    OUTPUT_SHAPES = (1000,)

    def __init__(self) -> None:
        super().__init__()
        self.mobile_net = torchvision.models.mobilenet_v3_small(
            norm_layer=partial(nn.GroupNorm, 2, affine=True),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.mobile_net(input)


class SqueezeNet(ShapedModule):
    """SqueezeNet."""

    INPUT_SHAPES = (3, 224, 224)
    OUTPUT_SHAPES = (1000,)

    def __init__(self) -> None:
        super().__init__()
        self.squeezenet = torchvision.models.squeezenet1_0()

    def forward(self, input: Tensor) -> Tensor:
        return self.squeezenet(input)


class InstanceNormMobileNetV2(ShapedModule):
    """MobileNetV3Small with BatchNorm2d layers replaced by InstanceNorm layers."""

    INPUT_SHAPES = (3, 224, 224)
    OUTPUT_SHAPES = (1000,)

    def __init__(self) -> None:
        super().__init__()
        self.mobilenet = torchvision.models.mobilenet_v2(
            norm_layer=partial(nn.InstanceNorm2d, track_running_stats=False, affine=True),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.mobilenet(input) / 10.0


# Other torchvision.models were not added for the following reasons:
# - VGG16: Sometimes takes to much memory on autojac even with bs=2, but autogram seems ok.
# - DenseNet: no way to easily replace the BatchNorms (no norm_layer param)
# - InceptionV3: no way to easily replace the BatchNorms (no norm_layer param)
# - GoogleNet: no way to easily replace the BatchNorms (no norm_layer param)
# - ShuffleNetV2: no way to easily replace the BatchNorms (no norm_layer param)
# - ResNeXt: Sometimes takes to much memory on autojac even with bs=2, but autogram seems ok.
# - WideResNet50: Sometimes takes to much memory on autojac even with bs=2, but autogram seems ok.
# - MNASNet: no way to easily replace the BatchNorms (no norm_layer param)
