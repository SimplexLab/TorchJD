import torch
from pytest import mark, raises
from torch import Tensor
from utils.tensors import ones_, tensor_

from torchjd.scalarization import DWA

from ._asserts import assert_grad_flow, assert_returns_scalar
from ._inputs import all_inputs


def test_uniform_weights_for_first_two_epochs() -> None:
    dwa = DWA(temperature=2.0)
    # Epoch 1: no completed epoch yet, so weights are uniform (sum).
    torch.testing.assert_close(dwa(tensor_([1.0, 3.0])), tensor_(4.0))
    dwa.step()
    # Epoch 2: only one completed epoch, so weights are still uniform (sum).
    torch.testing.assert_close(dwa(tensor_([2.0, 5.0])), tensor_(7.0))
    dwa.step()


def test_weights_from_previous_two_epochs() -> None:
    dwa = DWA(temperature=2.0)
    dwa(tensor_([1.0, 1.0]))
    dwa.step()  # Epoch 1 average = [1, 1].
    dwa(tensor_([1.0, 4.0]))
    dwa.step()  # Epoch 2 average = [1, 4].
    # Epoch 3: rates = [1, 4] / [1, 1] = [1, 4].
    losses = tensor_([3.0, 5.0])
    result = dwa(losses)
    expected_weights = 2.0 * torch.softmax(tensor_([1.0, 4.0]) / 2.0, dim=0)
    torch.testing.assert_close(result, (expected_weights * losses).sum())


def test_uses_per_epoch_average() -> None:
    # The weights use the average loss over each epoch's batches, not just the last batch.
    dwa = DWA(temperature=2.0)
    dwa(tensor_([2.0, 2.0]))
    dwa(tensor_([0.0, 0.0]))
    dwa.step()  # Epoch 1 average = [1, 1].
    dwa(tensor_([2.0, 6.0]))
    dwa(tensor_([0.0, 2.0]))
    dwa.step()  # Epoch 2 average = [1, 4].
    losses = tensor_([3.0, 5.0])
    result = dwa(losses)
    expected_weights = 2.0 * torch.softmax(tensor_([1.0, 4.0]) / 2.0, dim=0)
    torch.testing.assert_close(result, (expected_weights * losses).sum())


def test_step_discards_oldest_epoch() -> None:
    dwa = DWA(temperature=2.0)
    dwa(tensor_([9.0, 9.0]))
    dwa.step()  # Epoch 1 average = [9, 9]; should be discarded after epoch 3.
    dwa(tensor_([1.0, 1.0]))
    dwa.step()  # Epoch 2 average = [1, 1].
    dwa(tensor_([1.0, 4.0]))
    dwa.step()  # Epoch 3 average = [1, 4].
    # Epoch 4 uses only epochs 2 and 3: rates = [1, 4] / [1, 1] = [1, 4].
    losses = tensor_([3.0, 5.0])
    result = dwa(losses)
    expected_weights = 2.0 * torch.softmax(tensor_([1.0, 4.0]) / 2.0, dim=0)
    torch.testing.assert_close(result, (expected_weights * losses).sum())


def test_weights_sum_to_numel() -> None:
    dwa = DWA()
    dwa(tensor_([1.0, 2.0]))
    dwa.step()
    dwa(tensor_([2.0, 1.0]))
    dwa.step()
    # The weights sum to the number of elements, so weighting a vector of ones gives that count.
    torch.testing.assert_close(dwa(ones_((2,))), tensor_(2.0))


@mark.parametrize("values", all_inputs)
def test_expected_structure(values: Tensor) -> None:
    assert_returns_scalar(DWA(), values)


@mark.parametrize("values", all_inputs)
def test_grad_flow(values: Tensor) -> None:
    assert_grad_flow(DWA(), values)


def test_grad_flows_with_computed_weights() -> None:
    # After two epochs the weights are computed from the (detached) loss history; gradients must
    # still flow to the current values.
    dwa = DWA(temperature=2.0)
    dwa(tensor_([1.0, 1.0]))
    dwa.step()
    dwa(tensor_([1.0, 4.0]))
    dwa.step()
    assert_grad_flow(dwa, tensor_([3.0, 5.0]))


def test_reset() -> None:
    dwa = DWA()
    dwa(tensor_([1.0, 2.0]))
    dwa.step()
    dwa(tensor_([3.0, 4.0]))
    dwa.reset()
    assert dwa._previous_averages == []
    assert dwa._loss_sum is None
    assert dwa._n_batches == 0


def test_step_without_forward_is_noop() -> None:
    dwa = DWA()
    dwa.step()  # No losses accumulated yet.
    assert dwa._previous_averages == []


def test_supports_consistently_negative_losses() -> None:
    # DWA works on negative losses too, as long as each value keeps a consistent sign: the ratio of
    # same-sign losses is positive, so the weights match those of the equivalent positive case.
    dwa = DWA(temperature=2.0)
    dwa(tensor_([-2.0, -2.0]))
    dwa.step()  # Epoch 1 average = [-2, -2].
    dwa(tensor_([-2.0, -8.0]))
    dwa.step()  # Epoch 2 average = [-2, -8]; rates = [-2, -8] / [-2, -2] = [1, 4].
    losses = tensor_([3.0, 5.0])
    result = dwa(losses)
    expected_weights = 2.0 * torch.softmax(tensor_([1.0, 4.0]) / 2.0, dim=0)
    torch.testing.assert_close(result, (expected_weights * losses).sum())


def test_raises_on_shape_change_within_epoch() -> None:
    dwa = DWA()
    dwa(tensor_([1.0, 2.0]))
    with raises(ValueError):
        dwa(tensor_([1.0, 2.0, 3.0]))


def test_raises_on_shape_change_between_epochs() -> None:
    dwa = DWA()
    dwa(tensor_([1.0, 2.0]))
    dwa.step()
    dwa(tensor_([2.0, 1.0]))
    dwa.step()
    with raises(ValueError):
        dwa(tensor_([1.0, 2.0, 3.0]))


@mark.parametrize("temperature", [0.0, -1.0])
def test_raises_on_non_positive_temperature(temperature: float) -> None:
    with raises(ValueError):
        DWA(temperature=temperature)


def test_representations() -> None:
    assert repr(DWA()) == "DWA(temperature=2.0)"
    assert repr(DWA(temperature=1.5)) == "DWA(temperature=1.5)"
    assert str(DWA()) == "DWA"
