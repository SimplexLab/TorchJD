from torch.autograd.graph import get_gradient_edge
from utils.tensors import randn_

from torchjd.autogram._edge_registry import EdgeRegistry


def test_all_edges_are_leaves1() -> None:
    """Tests that get_leaf_edges works correctly when all edges are already leaves."""

    a = randn_([3, 4], requires_grad=True)
    b = randn_([4], requires_grad=True)
    c = randn_([3], requires_grad=True)

    d = (a @ b) + c

    edge_registry = EdgeRegistry()
    for tensor in [a, b, c]:
        edge_registry.register(get_gradient_edge(tensor))

    expected_leaves = {get_gradient_edge(tensor) for tensor in [a, b, c]}
    leaves = edge_registry.get_leaf_edges({get_gradient_edge(d)})
    assert leaves == expected_leaves


def test_all_edges_are_leaves2() -> None:
    """
    Tests that get_leaf_edges works correctly when all edges are already leaves of the graph of
    edges leading to them, but are not leaves of the autograd graph.
    """

    a = randn_([3, 4], requires_grad=True)
    b = randn_([4], requires_grad=True)
    c = randn_([4], requires_grad=True)
    d = randn_([4], requires_grad=True)

    e = a * b
    f = e + c
    g = f + d

    edge_registry = EdgeRegistry()
    for tensor in [e, g]:
        edge_registry.register(get_gradient_edge(tensor))

    expected_leaves = {get_gradient_edge(tensor) for tensor in [e, g]}
    leaves = edge_registry.get_leaf_edges({get_gradient_edge(e), get_gradient_edge(g)})
    assert leaves == expected_leaves


def test_some_edges_are_not_leaves1() -> None:
    """Tests that get_leaf_edges works correctly when some edges are leaves and some are not."""

    a = randn_([3, 4], requires_grad=True)
    b = randn_([4], requires_grad=True)
    c = randn_([4], requires_grad=True)
    d = randn_([4], requires_grad=True)

    e = a * b
    f = e + c
    g = f + d

    edge_registry = EdgeRegistry()
    for tensor in [a, b, c, d, e, f, g]:
        edge_registry.register(get_gradient_edge(tensor))

    expected_leaves = {get_gradient_edge(tensor) for tensor in [a, b, c, d]}
    leaves = edge_registry.get_leaf_edges({get_gradient_edge(g)})
    assert leaves == expected_leaves


def test_some_edges_are_not_leaves2() -> None:
    """
    Tests that get_leaf_edges works correctly when some edges are leaves and some are not. This
    time, not all tensors in the graph are registered so not all leavese in the graph have to be
    returned.
    """

    a = randn_([3, 4], requires_grad=True)
    b = randn_([4], requires_grad=True)
    c = randn_([4], requires_grad=True)
    d = randn_([4], requires_grad=True)

    e = a * b
    f = e + c
    g = f + d

    edge_registry = EdgeRegistry()
    for tensor in [a, c, d, e, g]:
        edge_registry.register(get_gradient_edge(tensor))

    expected_leaves = {get_gradient_edge(tensor) for tensor in [a, c, d]}
    leaves = edge_registry.get_leaf_edges({get_gradient_edge(g)})
    assert leaves == expected_leaves
