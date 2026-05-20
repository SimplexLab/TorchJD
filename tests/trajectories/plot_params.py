"""
Plot the trajectories of an objective function in the parameter space.

Usage:
    uv run python tests/trajectories/plot_params.py <objective>

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
    INITIAL_POINTS,
    LATEX_NAMES,
    OBJECTIVES,
)
from trajectories._objectives import ElementWiseQuadratic, WithSPSMappingMixin
from trajectories._optimization import compute_gradient_cosine_similarities
from trajectories._pareto_utils import sample_2d_spss
from trajectories._paths import RESULTS_DIR, get_param_plots_dir, get_params_dir
from trajectories._plotters import (
    AxesPlotter,
    ContentLimAdjuster,
    ContourCirclesPlotter,
    HeatmapPlotter,
    LimAdjuster,
    MultiTrajPlotter,
    SPSPlotter,
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
    print("Plotting in parameter space...")

    parser = argparse.ArgumentParser(
        description="Plot the trajectories of an objective function in the parameter space."
    )
    parser.add_argument(
        "objective",
        help=f"Key of the objective function. Choices: {list(OBJECTIVES)}",
    )
    args = parser.parse_args()
    objective_key = args.objective

    with open(RESULTS_DIR / objective_key / "metadata.json") as f:
        metadata = json.load(f)

    params_dir = get_params_dir(objective_key)
    param_plots_dir = get_param_plots_dir(objective_key)
    param_plots_dir.mkdir(parents=True, exist_ok=True)

    # This seems to be the only way to make the font be Type1, which is the only font type supported
    # by ICML.
    plt.rcParams.update({"text.usetex": True})
    objective_key = metadata["objective_key"]
    objective = OBJECTIVES[objective_key]

    if objective.n_params != 2:
        raise ValueError("Can only plot param trajectories for objectives with 2 params.")

    initial_points = INITIAL_POINTS[objective_key]
    initial_points_array = np.stack([np.array(point) for point in initial_points])
    main_content = initial_points_array

    common_plotter = SquareBoxAspectSetter()

    if objective.n_values == 2 and isinstance(objective, WithSPSMappingMixin):
        sps_points = sample_2d_spss(objective).numpy()
        main_content = np.concatenate([main_content, sps_points])
        common_plotter += SPSPlotter(sps_points)

    if isinstance(objective, ElementWiseQuadratic):
        common_plotter += AxesPlotter()
        common_plotter += ContourCirclesPlotter()
        common_plotter += LimAdjuster(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
    else:
        adjust_plotter = ContentLimAdjuster(main_content)
        common_plotter += adjust_plotter

        if objective.n_values == 2:
            similarities = compute_gradient_cosine_similarities(
                objective,
                x0_min=adjust_plotter.xlim[0],
                x0_max=adjust_plotter.xlim[1],
                x1_min=adjust_plotter.ylim[0],
                x1_max=adjust_plotter.ylim[1],
                n=200,
            )
            common_plotter += HeatmapPlotter(
                values=similarities.numpy() ** 3,
                x_min=adjust_plotter.xlim[0],
                x_max=adjust_plotter.xlim[1],
                y_min=adjust_plotter.ylim[0],
                y_max=adjust_plotter.ylim[1],
                vmin=-1,
                vmax=1,
                cmap="PiYG",
            )

    aggregator_keys = metadata["aggregator_keys"]
    aggregator_to_X = {key: np.load(params_dir / f"{key}.npy") for key in aggregator_keys}

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

    save_path = param_plots_dir / "all.pdf"

    for aggregator_key, X in aggregator_to_X.items():
        aggregator = AGGREGATORS[aggregator_key]
        print(aggregator)

        index = key_to_index[aggregator_key]
        i, j = get_subplot_position(index, n_aggregators, n_rows, n_cols)

        plotter = common_plotter + MultiTrajPlotter(X) + TitleSetter(LATEX_NAMES[aggregator_key])
        plotter += XAxisLabeller("$x_1$") if i == n_rows - 1 else XTicksClearer()
        plotter += YAxisLabeller("$x_2$") if j == 0 else YTicksClearer()

        plotter(axes[i][j])

    fig.tight_layout(h_pad=-2.5)
    print("Saving figure")
    plt.savefig(save_path, bbox_inches="tight")
    print()


if __name__ == "__main__":
    main()
