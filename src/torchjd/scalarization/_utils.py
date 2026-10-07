from torch import Tensor


def check_same_shape_as_values(tensor: Tensor | None, values: Tensor, name: str) -> None:
    """
    Checks that ``tensor`` has the same shape as ``values``. Nothing is checked if ``tensor`` is
    ``None``.

    :param tensor: The tensor to check.
    :param values: The tensor of values to scalarize.
    :param name: The name of the parameter corresponding to ``tensor``, used in the error message.
    """

    if tensor is not None and tensor.shape != values.shape:
        raise ValueError(
            f"Parameter `{name}` should have the same shape as `values`. Found "
            f"`{name}.shape = {tuple(tensor.shape)}` and `values.shape = {tuple(values.shape)}`."
        )
