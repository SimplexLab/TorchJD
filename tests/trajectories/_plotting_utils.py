"""Utility functions for plotting."""


def map_orders_to_indices(
    aggregator_keys: list[str], aggregator_order: dict[str, int]
) -> dict[str, int]:
    """Map aggregator keys to their indices based on sorted order.

    This function takes the available aggregators and maps them to sequential indices
    (0, 1, 2, ...) based on their order values. This ensures that subplots are positioned
    correctly regardless of which aggregators are actually present.

    :param aggregator_keys: List of aggregator keys to map.
    :param aggregator_order: Dictionary mapping aggregator keys to their order values.

    Example: if ``aggregator_keys = ["mean", "dualproj", "aligned_mtl"]`` with orders
    ``[0, 2, 8]``, this returns ``{"mean": 0, "dualproj": 1, "aligned_mtl": 2}``.
    """
    sorted_keys = sorted(aggregator_keys, key=lambda k: aggregator_order[k])
    return {key: idx for idx, key in enumerate(sorted_keys)}


def compute_subplot_layout(n_aggregators: int) -> tuple[int, int]:
    """Compute subplot layout (n_rows, n_cols) based on number of aggregators.

    :param n_aggregators: Number of aggregators to plot.
    :raises ValueError: If n_aggregators is not between 1 and 10.
    """
    if n_aggregators <= 5:
        return 1, n_aggregators
    if n_aggregators == 6:
        return 2, 3
    if n_aggregators == 7 or n_aggregators == 8:
        return 2, 4
    if n_aggregators == 9 or n_aggregators == 10:
        return 2, 5
    raise ValueError(f"Unsupported number of aggregators: {n_aggregators}")


def get_subplot_position(
    order: int, n_aggregators: int, n_rows: int, n_cols: int
) -> tuple[int, int]:
    """Convert order index to (row, col) position.

    :param order: The order index of the aggregator.
    :param n_aggregators: Total number of aggregators.
    :param n_rows: Number of rows in the subplot grid.
    :param n_cols: Number of columns in the subplot grid.
    """
    if n_rows == 1:
        return 0, order
    # For 2 rows: n_cols equals both n_aggregators//2 (even split) and (n_aggregators+1)//2 (odd).
    if n_aggregators in [6, 8, 10]:
        return order // n_cols, order % n_cols
    if n_aggregators in [7, 9]:
        if order < n_cols:
            return 0, order
        return 1, order - n_cols
    raise ValueError(f"Unsupported combination of n_aggregators={n_aggregators}, n_rows={n_rows}")


def get_unused_subplot_positions(
    n_aggregators: int, n_rows: int, n_cols: int
) -> list[tuple[int, int]]:
    """Get list of unused subplot positions.

    For layouts where not all subplot positions are used (e.g., 7 aggregators in a 2x4 grid),
    this function returns the positions that should be hidden.

    :param n_aggregators: Total number of aggregators.
    :param n_rows: Number of rows in the subplot grid.
    :param n_cols: Number of columns in the subplot grid.
    """
    if n_rows == 1:
        return []

    used_positions = set()
    for idx in range(n_aggregators):
        pos = get_subplot_position(idx, n_aggregators, n_rows, n_cols)
        used_positions.add(pos)

    all_positions = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    return [pos for pos in all_positions if pos not in used_positions]
