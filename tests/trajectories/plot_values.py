"""
Plot the trajectories of an objective function in the value space.

Usage:
    uv run python tests/trajectories/plot_values.py <objective>

Arguments:
    <objective>    The key of the objective function (e.g., EWQ, CQF, HQF).
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from trajectories._constants import (
    AGGREGATOR_ORDER,
    AGGREGATORS,
    LATEX_NAMES,
    OBJECTIVES,
    PLOT_VALUES_LIMS,
)
from trajectories._objectives import WithSPSMappingMixin
from trajectories._pareto_utils import compute_normalized_2d_pf_distances, sample_2d_pf
from trajectories._paths import RESULTS_DIR, get_value_plots_dir, get_values_dir
from trajectories._plotters import (
    ContentLimAdjuster,
    HeatmapPlotter,
    LimAdjuster,
    MultiTrajPlotter,
    PFPlotter,
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
    print("Plotting in value space...")

    parser = argparse.ArgumentParser(
        description="Plot the trajectories of an objective function in the value space."
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
    value_plots_dir = get_value_plots_dir(objective_key)
    value_plots_dir.mkdir(parents=True, exist_ok=True)

    # This seems to be the only way to make the font be Type1, which is the only font type supported
    # by ICML.
    plt.rcParams.update({"text.usetex": True})
    objective_key = metadata["objective_key"]
    objective = OBJECTIVES[objective_key]

    if objective.n_values != 2:
        raise ValueError("Can only plot values trajectories for objectives with 2 values.")

    common_plotter = SquareBoxAspectSetter()
    aggregator_keys = metadata["aggregator_keys"]
    aggregator_to_Y = {key: np.load(values_dir / f"{key}.npy") for key in aggregator_keys}
    first_agg_Y = next(iter(aggregator_to_Y.values()))
    initial_values = first_agg_Y[:, 0, :]
    main_content = initial_values

    if isinstance(objective, WithSPSMappingMixin):
        pf_points_array = sample_2d_pf(objective).numpy()
        common_plotter += PFPlotter(pf_points_array)
        main_content = np.concatenate([main_content, pf_points_array])

    if objective_key in PLOT_VALUES_LIMS:
        lims = PLOT_VALUES_LIMS[objective_key]
        xlim = lims["xlim"]
        ylim = lims["ylim"]
        common_plotter += LimAdjuster(xlim=xlim, ylim=ylim)
    else:
        adjust_plotter = ContentLimAdjuster(main_content)
        common_plotter += adjust_plotter
        xlim = adjust_plotter.xlim
        ylim = adjust_plotter.ylim

    if isinstance(objective, WithSPSMappingMixin):
        distances = compute_normalized_2d_pf_distances(
            objective,
            y0_min=xlim[0],
            y0_max=xlim[1],
            y1_min=ylim[0],
            y1_max=ylim[1],
            n=200,
        )
        common_plotter += HeatmapPlotter(
            values=distances.numpy(),
            x_min=xlim[0],
            x_max=xlim[1],
            y_min=ylim[0],
            y_max=ylim[1],
            vmin=0,
            vmax=1,
            cmap="Reds",
        )

    n_aggregators = len(aggregator_keys)
    n_rows, n_cols = compute_subplot_layout(n_aggregators)
    key_to_index = map_orders_to_indices(aggregator_keys, AGGREGATOR_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.5))
    if n_rows == n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    unused_positions = get_unused_subplot_positions(n_aggregators, n_rows, n_cols)
    for i, j in unused_positions:
        axes[i][j].axis("off")

    save_path = value_plots_dir / "all.pdf"

    for aggregator_key, Y in aggregator_to_Y.items():
        aggregator = AGGREGATORS[aggregator_key]
        print(aggregator)

        index = key_to_index[aggregator_key]
        i, j = get_subplot_position(index, n_aggregators, n_rows, n_cols)

        plotter = common_plotter + MultiTrajPlotter(Y) + TitleSetter(LATEX_NAMES[aggregator_key])
        plotter += XAxisLabeller("Objective $1$") if i == n_rows - 1 else XTicksClearer()
        plotter += YAxisLabeller("Objective $2$") if j == 0 else YTicksClearer()

        plotter(axes[i][j])

    fig.tight_layout(h_pad=-2.5)
    print("Saving figure")
    plt.savefig(save_path, bbox_inches="tight")
    print()


if __name__ == "__main__":
    main()
