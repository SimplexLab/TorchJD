"""
Optimize the objective using various aggregators. Save the trajectories in the parameter and value
spaces.

Usage:
    uv run python tests/trajectories/optimize.py <objective> <aggregator>...

Arguments:
    <objective>        The key of the objective function (e.g., EWQ, CQF, HQF).
    <aggregator>...    The keys of the aggregators to use (e.g., upgrad, mean, mgda).
"""

import argparse
import json
import random
import warnings

import numpy as np
import torch

from torchjd.aggregation import Stateful
from trajectories._constants import (
    AGGREGATORS,
    BASE_LEARNING_RATES,
    INITIAL_POINTS,
    LR_MULTIPLIER_OVERRIDES,
    LR_MULTIPLIERS,
    N_ITERS,
    OBJECTIVES,
)
from trajectories._optimization import optimize
from trajectories._paths import RESULTS_DIR, get_params_dir, get_values_dir

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the objective using various aggregators. Save the trajectories in the"
            " parameter and value spaces."
        )
    )
    parser.add_argument(
        "objective",
        help=f"Key of the objective function. Choices: {list(OBJECTIVES)}",
    )
    parser.add_argument(
        "aggregators",
        nargs="+",
        metavar="aggregator",
        help=f"Keys of the aggregators to use. Choices: {list(AGGREGATORS)}",
    )
    args = parser.parse_args()

    objective_key = args.objective
    if objective_key not in OBJECTIVES:
        raise ValueError(f"Unknown objective key: {objective_key}")

    aggregator_keys = args.aggregators
    for aggregator_key in aggregator_keys:
        if aggregator_key not in AGGREGATORS:
            raise ValueError(f"Unknown aggregator key: {aggregator_key}")

    objective = OBJECTIVES[objective_key]
    initial_points = INITIAL_POINTS[objective_key]

    learning_rates = {}
    lr_multiplier_overrides = LR_MULTIPLIER_OVERRIDES.get(objective_key, {})
    base_lr = BASE_LEARNING_RATES[objective_key]
    for key in LR_MULTIPLIERS:
        mult = lr_multiplier_overrides.get(key, LR_MULTIPLIERS[key])
        learning_rates[key] = mult * base_lr
    n_iters = N_ITERS[objective_key]

    torch.use_deterministic_algorithms(True)

    params_dir = get_params_dir(objective_key)
    values_dir = get_values_dir(objective_key)
    params_dir.mkdir(exist_ok=True, parents=True)
    values_dir.mkdir(exist_ok=True, parents=True)

    metadata = {
        "objective_key": objective_key,
        "objective_repr": repr(objective),
        "aggregator_keys": aggregator_keys,
        "aggregator_reprs": {key: repr(AGGREGATORS[key]) for key in aggregator_keys},
        "learning_rates": learning_rates,
        "initial_points": initial_points,
    }
    with open(RESULTS_DIR / objective_key / "metadata.json", "w") as f:
        json.dump(metadata, f)

    for aggregator_key in aggregator_keys:
        aggregator = AGGREGATORS[aggregator_key]
        lr = learning_rates[aggregator_key]
        print(aggregator)
        xs_list = []
        ys_list = []
        for initial_point in initial_points:
            print(initial_point)

            if isinstance(aggregator, Stateful):
                aggregator.reset()
            _reset_seed()

            initial_x = torch.tensor(initial_point)
            xs, ys = optimize(
                objective, initial_x=initial_x, aggregator=aggregator, lr=lr, n_iters=n_iters
            )

            xs_list.append(torch.stack(xs))
            ys_list.append(torch.stack(ys))

        X = torch.stack(xs_list).numpy()
        Y = torch.stack(ys_list).numpy()
        np.save(params_dir / f"{aggregator_key}.npy", X)
        np.save(values_dir / f"{aggregator_key}.npy", Y)
        print()


def _reset_seed() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


if __name__ == "__main__":
    main()
