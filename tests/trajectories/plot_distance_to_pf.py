"""
Plot the evolution of the distance to the Pareto front of an objective function over time.

Usage:
    uv run python tests/trajectories/plot_distance_to_pf.py <objective>

Arguments:
    <objective>    The key of the objective function (e.g., EWQ, CQF, HQF).
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from trajectories._constants import AGGREGATOR_ORDER, AGGREGATORS, LATEX_NAMES, OBJECTIVES
from trajectories._pareto_utils import make_2d_pf_distance_fn
from trajectories._paths import RESULTS_DIR, get_distance_to_pf_plots_dir, get_values_dir
from trajectories._plotters import (
    MultiEvolutionPlotter,
    SquareBoxAspectSetter,
    TitleSetter,
    XAxisLabeller,
    XTicksClearer,
    YAxisLabeller,
    YTicksClearer,
)
from trajectories._plotting_utils import (
    compute_subplot_layout,
    get_subplot_position,
    get_unused_subplot_positions,
    map_orders_to_indices,
)


def main() -> None:
    print("Plotting distance to Pareto front...")

    parser = argparse.ArgumentParser(
        description=(
            "Plot the evolution of the distance to the Pareto front of an objective function"
            " over time."
        )
    )
    parser.add_argument(
        "objective",
        help=f"Key of the objective function. Choices: {list(OBJECTIVES)}",
    )
    args = parser.parse_args()
    objective_key = args.objective

    with open(RESULTS_DIR / objective_key / "metadata.json") as f:
        metadata = json.load(f)

    values_dir = get_values_dir(objective_key)
    distance_to_pf_plots_dir = get_distance_to_pf_plots_dir(objective_key)
    distance_to_pf_plots_dir.mkdir(parents=True, exist_ok=True)

    # This seems to be the only way to make the font be Type1, which is the only font type supported
    # by ICML.
    plt.rcParams.update({"text.usetex": True})
    objective_key = metadata["objective_key"]
    objective = OBJECTIVES[objective_key]

    common_plotter = SquareBoxAspectSetter()

    aggregator_keys = metadata["aggregator_keys"]
    aggregator_to_Y = {key: np.load(values_dir / f"{key}.npy") for key in aggregator_keys}

    n_aggregators = len(aggregator_keys)
    n_rows, n_cols = compute_subplot_layout(n_aggregators)
    key_to_index = map_orders_to_indices(aggregator_keys, AGGREGATOR_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.5), sharey="all")
    if n_rows == n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    unused_positions = get_unused_subplot_positions(n_aggregators, n_rows, n_cols)
    for i, j in unused_positions:
        axes[i][j].axis("off")

    save_path = distance_to_pf_plots_dir / "all.pdf"
    pf_distance_fn = make_2d_pf_distance_fn(objective)

    for aggregator_key, Y in aggregator_to_Y.items():
        aggregator = AGGREGATORS[aggregator_key]
        print(aggregator)

        # Y has shape [n_initial_points, n_iter, n_values]
        pfd = torch.vmap(torch.vmap(pf_distance_fn))(torch.from_numpy(Y).to(dtype=torch.float64))

        index = key_to_index[aggregator_key]
        i, j = get_subplot_position(index, n_aggregators, n_rows, n_cols)

        plotter = (
            common_plotter
            + MultiEvolutionPlotter(pfd.numpy())
            + TitleSetter(LATEX_NAMES[aggregator_key])
        )
        plotter += XAxisLabeller("Iteration") if i == n_rows - 1 else XTicksClearer()
        plotter += YAxisLabeller("Distance to Pareto front") if j == 0 else YTicksClearer()

        plotter(axes[i][j])

    fig.tight_layout(h_pad=-2.5)

    print("Saving figure")
    plt.savefig(save_path, bbox_inches="tight")
    print()


if __name__ == "__main__":
    main()
