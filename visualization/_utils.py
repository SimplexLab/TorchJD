from collections.abc import Callable

import numpy as np
import torch
from plotly import graph_objects as go
from plotly.graph_objs import Figure, Scatter

from torchjd.aggregation import Aggregator


class Plotter:
    def __init__(
        self,
        aggregator_factories: dict[str, Callable[[], Aggregator]],
        selected_keys: list[str],
        matrix: torch.Tensor,
        seed: int = 0,
    ) -> None:
        self._aggregator_factories = aggregator_factories
        self.selected_keys = selected_keys
        self.matrix = matrix
        self.seed = seed

    def make_fig(self) -> Figure:
        torch.random.manual_seed(self.seed)
        aggregators = [self._aggregator_factories[key]() for key in self.selected_keys]
        results = [agg(self.matrix) for agg in aggregators]

        fig = go.Figure()

        start_angle, opening = compute_2d_dual_cone(self.matrix.numpy())
        cone = make_cone_scatter(start_angle, opening, label="Dual cone")
        fig.add_trace(cone)

        for i in range(len(self.matrix)):
            scatter = make_vector_scatter(
                self.matrix[i],
                "blue",
                f"g{i + 1}",
                textposition="top right",
            )
            fig.add_trace(scatter)

        for i in range(len(results)):
            scatter = make_vector_scatter(
                results[i],
                "black",
                self.selected_keys[i],
                showlegend=True,
                dash=True,
            )
            fig.add_trace(scatter)

        all_x = [0] + self.matrix.numpy()[:, 0].tolist() + [res[0].item() for res in results]
        all_y = [0] + self.matrix.numpy()[:, 1].tolist() + [res[1].item() for res in results]
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        len_x = max_x - min_x
        len_y = max_y - min_y
        margin_prop = 0.05
        range_x = [min_x - len_x * margin_prop, max_x + len_x * margin_prop]
        range_y = [min_y - len_y * margin_prop, max_y + len_y * margin_prop]

        fig.update_layout(hovermode=False, width=1000, height=900)
        fig.update_xaxes(scaleanchor="y", scaleratio=1, range=range_x)
        fig.update_yaxes(range=range_y)

        return fig


def make_vector_scatter(
    gradient: torch.Tensor,
    color: str,
    label: str,
    showlegend: bool = False,
    dash: bool = False,
    textposition: str = "bottom center",
    line_width: float = 2.5,
    text_size: float = 12,
    marker_size: float = 12,
) -> Scatter:
    line = {"color": color, "width": line_width}
    if dash:
        line["dash"] = "dash"

    scatter = go.Scatter(
        x=[0, gradient[0]],
        y=[0, gradient[1]],
        mode="lines+markers+text",
        line=line,
        marker={
            "symbol": "arrow",
            "color": color,
            "size": marker_size,
            "angleref": "previous",
        },
        name=label,
        text=["", label],
        textposition=textposition,
        textfont={"color": color, "size": text_size},
        showlegend=showlegend,
    )
    return scatter


def make_cone_scatter(
    start_angle: float,
    opening: float,
    label: str,
    scale: float = 100.0,
    printable: bool = False,
) -> Scatter:
    if opening < -1e-8:
        cone_outline = np.zeros([0, 2])
    else:
        middle_angle = start_angle + opening / 2
        end_angle = start_angle + opening

        start_vec = angle_to_coord(start_angle, scale)
        end_vec = angle_to_coord(end_angle, scale)

        if np.abs(opening - 2 * np.pi) <= 0.01:
            cone_outline = np.array(
                [
                    [0, 0],
                    start_vec,
                    end_vec,
                    [0, 0],
                ],
            )
        else:
            middle_point = angle_to_coord(middle_angle, scale)

            cone_outline = np.array(
                [
                    [0, 0],
                    start_vec,
                    middle_point,
                    end_vec,
                    [0, 0],
                ],
            )

    if printable:
        fillpattern = {
            "bgcolor": "white",
            "shape": "\\",
            "fgcolor": "rgba(0, 220, 0, 0.5)",
            "size": 30,
            "solidity": 0.15,
        }
    else:
        fillpattern = None

    cone = go.Scatter(
        x=cone_outline[:, 0],
        y=cone_outline[:, 1],
        fill="toself",
        mode="lines",
        fillcolor="rgba(0, 255, 0, 0.07)",
        line={"color": "rgb(0, 220, 0)", "width": 2},
        name=label,
        fillpattern=fillpattern,
    )

    return cone


def compute_2d_dual_cone(matrix: np.ndarray) -> tuple[float, float]:
    row_angles = [coord_to_angle(*row)[0] for row in matrix]
    start_angles = [(angle - np.pi / 2) % (2 * np.pi) for angle in row_angles]

    cone_start_angle = start_angles[0]
    opening = np.pi
    for hs_start_angle in start_angles[1:]:
        cone_start_angle, opening = combine_bounds(cone_start_angle, opening, hs_start_angle)

    return cone_start_angle, opening


def combine_bounds(
    cone_start_angle: float,
    opening: float,
    hs_start_angle: float,
) -> tuple[float, float]:
    angle_between_starts = (hs_start_angle - cone_start_angle) % (2 * np.pi)
    if angle_between_starts < np.pi:
        cone_start_angle = hs_start_angle
        opening = opening - angle_between_starts
    else:
        opening = min(opening, (hs_start_angle + np.pi - cone_start_angle) % (2 * np.pi))

    return cone_start_angle, opening


def coord_to_angle(x: float, y: float) -> tuple[float, float]:
    r = np.sqrt(x**2 + y**2)
    if r == 0:
        raise ValueError("No angle")
    angle = np.arccos(x / r) if y >= 0 else 2 * np.pi - np.arccos(x / r)
    return angle, r


def angle_to_coord(angle: float, r: float = 1.0) -> tuple[float, float]:
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return x, y
