import torch
from pytest import mark, raises
from settings import DTYPE
from torch.autograd import grad
from utils.asserts import (
    assert_grad_close,
    assert_has_grad,
    assert_has_jac,
    assert_has_no_grad,
    assert_has_no_jac,
    assert_jac_close,
)
from utils.tensors import arange_, ones_, rand_, randn_, tensor_

from torchjd.autojac import mtl_backward
from torchjd.autojac._mtl_backward import _create_grad_tensors_dict, _create_transform
from torchjd.autojac._transform import OrderedSet


@mark.parametrize("default_grad_tensors", [True, False])
def test_check_create_transform(default_grad_tensors: bool):
    """Tests that _create_transform creates a valid Transform."""

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_([1.0, 2.0], requires_grad=True)
    p2 = tensor_([3.0, 4.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    optional_grad_tensors = None if default_grad_tensors else [tensor_(1.0)] * 2

    tensors = OrderedSet([y1, y2])
    grad_tensors_dict = _create_grad_tensors_dict(tensors, optional_grad_tensors)

    transform = _create_transform(
        tensors=tensors,
        features=OrderedSet([f1, f2]),
        tasks_params=[OrderedSet([p1]), OrderedSet([p2])],
        shared_params=OrderedSet([p0]),
        retain_graph=False,
        parallel_chunk_size=None,
    )

    output_keys = transform.check_keys(set(grad_tensors_dict.keys()))
    assert output_keys == set()


def test_shape_is_correct():
    """Tests that mtl_backward works correctly."""

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_([1.0, 2.0], requires_grad=True)
    p2 = tensor_([3.0, 4.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    mtl_backward(tensors=[y1, y2], features=[f1, f2])

    assert_has_jac(p0)
    for p in [p1, p2]:
        assert_has_grad(p)


@mark.parametrize("shape", [(1, 3), (2, 3), (2, 6), (5, 8), (20, 55)])
@mark.parametrize("manually_specify_shared_params", [True, False])
@mark.parametrize("manually_specify_tasks_params", [True, False])
@mark.parametrize("chunk_size", [1, 2, None])
def test_value_is_correct(
    shape: tuple[int, int],
    manually_specify_shared_params: bool,
    manually_specify_tasks_params: bool,
    chunk_size: int | None,
):
    """
    Tests that the .jac value filled by mtl_backward is correct in a simple example of
    matrix-vector product for three tasks whose loss are given by a simple inner product of the
    shared features with the task parameter.

    This test should work with or without manually specifying the parameters.
    """

    p0 = randn_([shape[1]], requires_grad=True)
    p1 = randn_(shape[0], requires_grad=True)
    p2 = randn_(shape[0], requires_grad=True)
    p3 = randn_(shape[0], requires_grad=True)

    J = randn_(shape)
    f = J @ p0
    y1 = p1 @ f
    y2 = p2 @ f
    y3 = p3 @ f

    shared_params = [p0] if manually_specify_shared_params else None

    tasks_params = [[p1], [p2], [p3]] if manually_specify_tasks_params else None

    mtl_backward(
        tensors=[y1, y2, y3],
        features=f,
        tasks_params=tasks_params,
        shared_params=shared_params,
        parallel_chunk_size=chunk_size,
    )

    assert_grad_close(p1, f)
    assert_grad_close(p2, f)
    assert_grad_close(p3, f)

    expected_jacobian = torch.stack((p1, p2, p3)) @ J
    assert_jac_close(p0, expected_jacobian)


def test_empty_tasks_fails():
    """Tests that mtl_backward raises an error when called with an empty list of tasks."""

    p0 = tensor_([1.0, 2.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()

    with raises(ValueError):
        mtl_backward(tensors=[], features=[f1, f2])


def test_single_task():
    """Tests that mtl_backward works correctly with a single task."""

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_([3.0, 4.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]

    mtl_backward(tensors=[y1], features=[f1, f2])

    assert_has_jac(p0)
    assert_has_grad(p1)


def test_incoherent_task_number_fails():
    """
    Tests that mtl_backward raises an error when called with the number of tasks losses different
    from the number of tasks parameters.
    """

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_([1.0, 2.0], requires_grad=True)
    p2 = tensor_([3.0, 4.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    with raises(ValueError):
        mtl_backward(
            tensors=[y1, y2],
            features=[f1, f2],
            tasks_params=[[p1]],  # Wrong
            shared_params=[p0],
        )
    with raises(ValueError):
        mtl_backward(
            tensors=[y1],  # Wrong
            features=[f1, f2],
            tasks_params=[[p1], [p2]],
            shared_params=[p0],
        )


def test_empty_params():
    """Tests that mtl_backward does not fill the .jac/.grad values if no parameter is specified."""

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_([1.0, 2.0], requires_grad=True)
    p2 = tensor_([3.0, 4.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    mtl_backward(
        tensors=[y1, y2],
        features=[f1, f2],
        tasks_params=[[], []],
        shared_params=[],
    )

    assert_has_no_jac(p0)
    for p in [p1, p2]:
        assert_has_no_grad(p)


def test_multiple_params_per_task():
    """Tests that mtl_backward works correctly when the tasks each have several parameters."""

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1_a = tensor_(1.0, requires_grad=True)
    p1_b = tensor_([2.0, 3.0], requires_grad=True)
    p1_c = tensor_([[4.0, 5.0], [6.0, 7.0]], requires_grad=True)
    p2_a = tensor_(8.0, requires_grad=True)
    p2_b = tensor_([9.0, 10.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1_a + (f2 * p1_b).sum() + (f1 * p1_c).sum()
    y2 = f1 * p2_a * (f2 * p2_b).sum()

    mtl_backward(tensors=[y1, y2], features=[f1, f2])

    assert_has_jac(p0)
    for p in [p1_a, p1_b, p1_c, p2_a, p2_b]:
        assert_has_grad(p)


@mark.parametrize(
    "shared_params_shapes",
    [
        [()],
        [(2,)],
        [(3, 2)],
        [(4, 3, 2)],
        [(), (2,)],
        [(3, 2), (2,)],
        [(4, 3, 2), (3, 2), ()],
        [(5, 4, 3, 2), (5, 4, 3, 2)],
    ],
)
def test_various_shared_params(shared_params_shapes: list[tuple[int]]):
    """Tests that mtl_backward works correctly with various kinds of shared_params."""

    shared_params = [rand_(shape, requires_grad=True) for shape in shared_params_shapes]
    p1 = tensor_([1.0, 2.0], requires_grad=True)
    p2 = tensor_([3.0, 4.0], requires_grad=True)

    features = [shared_param.sum(dim=-1) for shared_param in shared_params]
    y1 = torch.stack([f.sum() for f in features]).sum()
    y2 = torch.stack([f.sum() ** 2 for f in features]).sum()

    mtl_backward(
        tensors=[y1, y2],
        features=features,
        tasks_params=[[p1], [p2]],  # Enforce differentiation w.r.t. params that haven't been used
        shared_params=shared_params,
    )

    for p in shared_params:
        assert_has_jac(p)
    for p in [p1, p2]:
        assert_has_grad(p)


def test_partial_params():
    """
    Tests that mtl_backward fills the right .jac/.grad values when only a subset of the parameters
    are specified as inputs.
    """

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_([1.0, 2.0], requires_grad=True)
    p2 = tensor_([3.0, 4.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    mtl_backward(
        tensors=[y1, y2],
        features=[f1, f2],
        tasks_params=[[p1], []],
        shared_params=[p0],
    )

    assert_has_jac(p0)
    assert_has_grad(p1)
    assert_has_no_grad(p2)


def test_empty_features_fails():
    """Tests that mtl_backward expectedly raises an error when no there is no feature."""

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_([1.0, 2.0], requires_grad=True)
    p2 = tensor_([3.0, 4.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    with raises(ValueError):
        mtl_backward(tensors=[y1, y2], features=[])


@mark.parametrize(
    "shape",
    [
        (2,),
        (3, 2),
        (4, 3, 2),
        (5, 4, 3, 2),
    ],
)
def test_various_single_features(shape: tuple[int, ...]):
    """Tests that mtl_backward works correctly with various kinds of feature tensors."""

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_([3.0, 4.0], requires_grad=True)
    p2 = tensor_([5.0, 6.0], requires_grad=True)

    f = rand_(shape) @ p0

    y1 = (f * p1[0]).sum() + (f * p1[1]).sum()
    y2 = (f * p2[0]).sum() * (f * p2[1]).sum()

    mtl_backward(tensors=[y1, y2], features=f)

    assert_has_jac(p0)
    for p in [p1, p2]:
        assert_has_grad(p)


@mark.parametrize(
    "shapes",
    [
        [(2,)],
        [(3, 2)],
        [(4, 3, 2)],
        [(5, 4, 3, 2)],
        [(2,), (2,)],
        [(3, 2), (2,)],
        [(4, 3, 2), (3, 2), (2,)],
        [(5, 4, 3, 2), (5, 4, 3, 2)],
    ],
)
def test_various_feature_lists(shapes: list[tuple[int]]):
    """Tests that mtl_backward works correctly with various kinds of feature lists."""

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = arange_(len(shapes), dtype=DTYPE, requires_grad=True)
    p2 = tensor_(5.0, requires_grad=True)

    features = [rand_(shape) @ p0 for shape in shapes]

    y1 = sum([(f * p).sum() for f, p in zip(features, p1, strict=True)])
    y2 = (features[0] * p2).sum()

    mtl_backward(tensors=[y1, y2], features=features)

    assert_has_jac(p0)
    for p in [p1, p2]:
        assert_has_grad(p)


def test_non_scalar_loss_fails():
    """Tests that mtl_backward raises an error when used with a non-scalar loss."""

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_([1.0, 2.0], requires_grad=True)
    p2 = tensor_([3.0, 4.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = torch.stack([f1 * p1[0], f2 * p1[1]])  # Non-scalar
    y2 = f1 * p2[0] + f2 * p2[1]

    with raises(ValueError):
        mtl_backward(tensors=[y1, y2], features=[f1, f2])


@mark.parametrize("chunk_size", [None, 1, 2, 4])
def test_various_valid_chunk_sizes(chunk_size):
    """Tests that mtl_backward works for various valid values of parallel_chunk_size."""

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_([1.0, 2.0], requires_grad=True)
    p2 = tensor_([3.0, 4.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    mtl_backward(
        tensors=[y1, y2],
        features=[f1, f2],
        parallel_chunk_size=chunk_size,
    )

    assert_has_jac(p0)
    for p in [p1, p2]:
        assert_has_grad(p)


@mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_fails(chunk_size: int):
    """Tests that mtl_backward raises an error when using invalid chunk sizes."""

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_([1.0, 2.0], requires_grad=True)
    p2 = tensor_([3.0, 4.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    with raises(ValueError):
        mtl_backward(
            tensors=[y1, y2],
            features=[f1, f2],
            parallel_chunk_size=chunk_size,
        )


def test_shared_param_retaining_grad_fails():
    """
    Tests that mtl_backward fails to fill a valid `.grad` when some shared param in the computation
    graph of the ``features`` parameter retains grad and vmap has to be used.
    """

    p0 = tensor_(1.0, requires_grad=True)
    p1 = tensor_(2.0, requires_grad=True)
    p2 = tensor_(3.0, requires_grad=True)

    a = 2 * p0
    a.retain_grad()
    f = 3 * a
    y1 = p1 * f
    y2 = p2 * f

    # mtl_backward itself doesn't raise the error, but it fills a.grad with a BatchedTensor
    mtl_backward(
        tensors=[y1, y2],
        features=[f],
        tasks_params=[[p1], [p2]],
        shared_params=[a, p0],
    )

    with raises(RuntimeError):
        # Using such a BatchedTensor should result in an error
        _ = -a.grad  # type: ignore[unsupported-operator]


def test_shared_activation_retaining_grad_fails():
    """
    Tests that mtl_backward fails to fill a valid `.grad` when some tensor in the computation graph
    of the ``features`` parameter retains grad and vmap has to be used.
    """

    p0 = tensor_(1.0, requires_grad=True)
    p1 = tensor_(2.0, requires_grad=True)
    p2 = tensor_(3.0, requires_grad=True)

    a = 2 * p0
    a.retain_grad()
    f = 3 * a
    y1 = p1 * f
    y2 = p2 * f

    # mtl_backward itself doesn't raise the error, but it fills a.grad with a BatchedTensor
    mtl_backward(
        tensors=[y1, y2],
        features=[f],
        tasks_params=[[p1], [p2]],
        shared_params=[p0],
    )

    with raises(RuntimeError):
        # Using such a BatchedTensor should result in an error
        _ = -a.grad  # type: ignore[unsupported-operator]


def test_tasks_params_overlap():
    """Tests that mtl_backward works correctly when the tasks' parameters have some overlap."""

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_(2.0, requires_grad=True)
    p2 = tensor_(3.0, requires_grad=True)
    p12 = tensor_(4.0, requires_grad=True)

    f = tensor_([-1.0, 1.0]) @ p0
    y1 = f * p1 * p12
    y2 = f * p2 * p12

    mtl_backward(tensors=[y1, y2], features=[f])

    assert_grad_close(p2, f * p12)
    assert_grad_close(p1, f * p12)
    assert_grad_close(p12, f * p1 + f * p2)

    J = tensor_([[-8.0, 8.0], [-12.0, 12.0]])
    assert_jac_close(p0, J)


def test_tasks_params_are_the_same():
    """Tests that mtl_backward works correctly when the tasks have the same params."""

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_(2.0, requires_grad=True)

    f = tensor_([-1.0, 1.0]) @ p0
    y1 = f * p1
    y2 = f + p1

    mtl_backward(tensors=[y1, y2], features=[f])

    assert_grad_close(p1, f + 1)

    J = tensor_([[-2.0, 2.0], [-1.0, 1.0]])
    assert_jac_close(p0, J)


def test_task_params_is_subset_of_other_task_params():
    """
    Tests that mtl_backward works correctly when one task's params is a subset of another task's
    params.
    """

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_(2.0, requires_grad=True)
    p2 = tensor_(3.0, requires_grad=True)

    f = tensor_([-1.0, 1.0]) @ p0
    y1 = f * p1
    y2 = y1 * p2

    mtl_backward(tensors=[y1, y2], features=[f], retain_graph=True)

    assert_grad_close(p2, y1)
    assert_grad_close(p1, p2 * f + f)

    J = tensor_([[-2.0, 2.0], [-6.0, 6.0]])
    assert_jac_close(p0, J)


def test_shared_params_overlapping_with_tasks_params_fails():
    """
    Tests that mtl_backward raises an error when the set of shared params overlaps with the set of
    task-specific params.
    """

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_(2.0, requires_grad=True)
    p2 = tensor_(3.0, requires_grad=True)

    f = tensor_([-1.0, 1.0]) @ p0
    y1 = f * p1
    y2 = p0.sum() * f * p2

    with raises(ValueError):
        mtl_backward(
            tensors=[y1, y2],
            features=[f],
            tasks_params=[[p1], [p0, p2]],  # Problem: p0 is also shared
            shared_params=[p0],
        )


def test_default_shared_params_overlapping_with_default_tasks_params_fails():
    """
    Tests that mtl_backward raises an error when the set of shared params obtained by default
    overlaps with the set of task-specific params obtained by default.
    """

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_(2.0, requires_grad=True)
    p2 = tensor_(3.0, requires_grad=True)

    f = tensor_([-1.0, 1.0]) @ p0
    y1 = f * p1
    y2 = p0.sum() * f * p2

    with raises(ValueError):
        mtl_backward(
            tensors=[y1, y2],
            features=[f],
        )


def test_repeated_losses():
    """
    Tests that mtl_backward does not allow repeating losses.

    This behavior is different from torch.autograd.backward which would sum the gradients of the
    repeated losses, but it simplifies a lot the implementation of autojac and there are alternative
    ways of producing Jacobians with repeated rows anyway.
    """

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_([1.0, 2.0], requires_grad=True)
    p2 = tensor_([3.0, 4.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    with raises(ValueError):
        losses = [y1, y1, y2]
        mtl_backward(tensors=losses, features=[f1, f2], retain_graph=True)


def test_repeated_features():
    """
    Tests that mtl_backward does not allow repeating features.

    Repeated features are a bit more tricky, because we differentiate with respect to them (in which
    case it shouldn't matter that they are repeated) and we also differentiate them (in which case
    repetition would be very confusing and should thus be forbidden).
    """

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_([1.0, 2.0], requires_grad=True)
    p2 = tensor_([3.0, 4.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    with raises(ValueError):
        features = [f1, f1, f2]
        mtl_backward(tensors=[y1, y2], features=features)


def test_repeated_shared_params():
    """
    Tests that mtl_backward correctly works when some shared are repeated. Since these are tensors
    with respect to which we differentiate, to match the behavior of torch.autograd.backward, this
    repetition should not affect the result.
    """

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_([1.0, 2.0], requires_grad=True)
    p2 = tensor_([3.0, 4.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    J0 = tensor_([[3.0, 9.0], [5.0, 19.0]])
    g1 = grad([y1], [p1], retain_graph=True)[0]
    g2 = grad([y2], [p2], retain_graph=True)[0]

    shared_params = [p0, p0]
    mtl_backward(tensors=[y1, y2], features=[f1, f2], shared_params=shared_params)

    assert_jac_close(p0, J0)
    assert_grad_close(p1, g1)
    assert_grad_close(p2, g2)


def test_repeated_task_params():
    """
    Tests that mtl_backward correctly works when some task-specific params are repeated for some
    task. Since these are tensors with respect to which we differentiate, to match the behavior of
    torch.autograd.backward, this repetition should not affect the result.
    """

    p0 = tensor_([1.0, 2.0], requires_grad=True)
    p1 = tensor_([1.0, 2.0], requires_grad=True)
    p2 = tensor_([3.0, 4.0], requires_grad=True)

    f1 = tensor_([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    J0 = tensor_([[3.0, 9.0], [5.0, 19.0]])
    g1 = grad([y1], [p1], retain_graph=True)[0]
    g2 = grad([y2], [p2], retain_graph=True)[0]

    tasks_params = [[p1, p1], [p2]]
    mtl_backward(tensors=[y1, y2], features=[f1, f2], tasks_params=tasks_params)

    assert_jac_close(p0, J0)
    assert_grad_close(p1, g1)
    assert_grad_close(p2, g2)


def test_grad_tensors_value_is_correct():
    """
    Tests that backward correctly computes the element-wise product of grad_tensors and the tensors.
    """

    p0 = randn_(3, requires_grad=True)
    p1 = randn_(2, requires_grad=True)
    p2 = randn_(2, requires_grad=True)
    p3 = randn_(4, requires_grad=True)

    J = randn_(2, 3)
    f = J @ p0  # shape: [2]
    y1 = p1 @ f  # shape: []
    y2 = p2 * f  # shape: [2]
    y3 = torch.outer(p3, f)  # shape: [4, 2]

    grad_y1 = torch.randn_like(y1)
    grad_y2 = torch.randn_like(y2)
    grad_y3 = torch.randn_like(y3)

    mtl_backward(
        tensors=[y1, y2, y3],
        features=f,
        grad_tensors=[grad_y1, grad_y2, grad_y3],
        tasks_params=None,
        shared_params=None,
        parallel_chunk_size=None,
    )

    assert_grad_close(p1, grad_y1 * f)
    assert_grad_close(p2, grad_y2 * f)
    assert_grad_close(p3, grad_y3 @ f)

    expected_jacobian = torch.stack([grad_y1 * p1, grad_y2 * p2, grad_y3.T @ p3]) @ J
    assert_jac_close(p0, expected_jacobian)


def test_grad_tensors_length_mismatch():
    """Tests that mtl_backward raises a ValueError early if len(grad_tensors) != len(tensors)."""

    p0 = randn_(3, requires_grad=True)
    p1 = randn_(2, requires_grad=True)
    p2 = randn_(2, requires_grad=True)

    J = randn_(2, 3)
    f = J @ p0
    y1 = p1 @ f
    y2 = p2 @ f

    with raises(
        ValueError,
        match=r"`grad_tensors` should have the same length as `tensors`\. \(got 1 and 2\)",
    ):
        mtl_backward(
            tensors=[y1, y2],
            features=f,
            grad_tensors=[torch.ones_like(y1)],
        )


def test_grad_tensors_shape_mismatch():
    """
    Tests that mtl_backward raises a ValueError early if the shape of a tensor in grad_tensors is
    incompatible with the corresponding tensor.
    """

    p0 = randn_(3, requires_grad=True)
    p1 = randn_(2, requires_grad=True)
    p2 = randn_(2, requires_grad=True)

    J = randn_(2, 3)
    f = J @ p0
    y1 = p1 @ f  # scalar, shape ()
    y2 = p2 @ f  # scalar, shape ()

    with raises(
        ValueError,
        match=r"Shape mismatch: `grad_tensors\[0\]` has shape .* but `tensors\[0\]` has shape .*\.",
    ):
        mtl_backward(
            tensors=[y1, y2],
            features=f,
            grad_tensors=[ones_(2), torch.ones_like(y2)],  # shape (2,) != shape () of y1
        )
