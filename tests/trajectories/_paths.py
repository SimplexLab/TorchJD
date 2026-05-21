from pathlib import Path

from tests.paths import TRAJECTORIES_RESULTS_DIR


def get_params_dir(objective_key: str) -> Path:
    return TRAJECTORIES_RESULTS_DIR / objective_key / "X"


def get_values_dir(objective_key: str) -> Path:
    return TRAJECTORIES_RESULTS_DIR / objective_key / "Y"


def get_param_plots_dir(objective_key: str) -> Path:
    return TRAJECTORIES_RESULTS_DIR / objective_key / "param_plots"


def get_value_plots_dir(objective_key: str) -> Path:
    return TRAJECTORIES_RESULTS_DIR / objective_key / "value_plots"


def get_distance_to_pf_plots_dir(objective_key: str) -> Path:
    return TRAJECTORIES_RESULTS_DIR / objective_key / "distance_to_pf"
