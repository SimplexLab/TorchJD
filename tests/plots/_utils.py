import numpy as np
import torch
from plotly import graph_objects as go
from plotly.graph_objs import Figure, Scatter

from torchjd.aggregation import Aggregator


class Plotter:
    def __init__(self, aggregators: list[Aggregator], matrix: torch.Tensor, seed: int = 0):
        self.aggregators = aggregators
        self.matrix = matrix
        self.seed = seed

    def make_fig(self) -> Figure:
        torch.random.manual_seed(self.seed)
        results = [agg(self.matrix) for agg in self.aggregators]

        fig = go.Figure()

        start_angle, opening = compute_2d_dual_cone(self.matrix.numpy())
        cone = make_cone_scatter(start_angle, opening, label="Dual cone")
        fig.add_trace(cone)

        for i in range(len(self.matrix)):
            scatter = make_vector_scatter(self.matrix[i], "blue", f"g{i + 1}")
            fig.add_trace(scatter)

        for i in range(len(results)):
            scatter = make_vector_scatter(
                results[i],
                "black",
                str(self.aggregators[i]),
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
                    [0, 0],  # Origin
                    start_vec,  # Tip of the first vector
                    end_vec,  # Tip of the second vector
                    [0, 0],  # Back to the origin to close the cone
                ],
            )
        else:
            middle_point = angle_to_coord(middle_angle, scale)

            cone_outline = np.array(
                [
                    [0, 0],  # Origin
                    start_vec,  # Tip of the first vector
                    middle_point,  # Tip of the vector in-between
                    end_vec,  # Tip of the second vector
                    [0, 0],  # Back to the origin to close the cone
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
        fill="toself",  # Fill the area inside the polygon
        mode="lines",
        fillcolor="rgba(0, 255, 0, 0.07)",
        line={"color": "rgb(0, 220, 0)", "width": 2},
        name=label,
        fillpattern=fillpattern,
    )

    return cone


def make_segment_scatter(start: torch.Tensor, end: torch.Tensor) -> Scatter:
    segment = go.Scatter(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        mode="lines",
        line={"color": "rgb(150, 150, 150)", "width": 2.5, "dash": "longdash"},
    )

    return segment


def make_polygon_scatter(points: list[torch.Tensor]) -> Scatter:
    polygon = go.Scatter(
        x=[point[0] for point in points],
        y=[point[1] for point in points],
        mode="lines",
        line={"color": "rgb(100, 100, 100)", "width": 1.5},
    )
    return polygon


def make_right_angle(
    vector: torch.Tensor,
    size: float,
    positive_para: bool = True,
    positive_orth: bool = True,
) -> list[torch.Tensor]:
    vec_para = vector / torch.linalg.norm(vector) * size
    vec_orth = torch.tensor([-vec_para[1], vec_para[0]])

    para_mult = 1.0 if positive_para else -1.0
    orth_mult = 1.0 if positive_orth else -1.0

    p0 = vector
    p1 = p0 + para_mult * vec_para
    p2 = p1 + orth_mult * vec_orth
    p3 = p2 - para_mult * vec_para

    return [p1, p2, p3]


def compute_2d_dual_cone(matrix: np.ndarray) -> tuple[float, float]:
    """
    Computes the frontier of the dual cone from a matrix of 2-dimensional rows.
    Returns the result as an angle in [0, 2pi[ corresponding to the start of the cone, and an
    opening angle, that is <= pi and that can be negative if the cone is empty.

    This method currently does not handle the case where the cone is a straight line passing by the
    origin (when matrix is for instance [[1, 0],[-1, 0]]).

    :param matrix: Any real-valued [m, 2] matrix.
    """

    row_angles = [coord_to_angle(*row)[0] for row in matrix]

    # Compute the start of the dual half-space of each individual row.
    start_angles = [(angle - np.pi / 2) % (2 * np.pi) for angle in row_angles]

    # Combine these dual half-spaces to obtain the global dual cone.
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
    """
    Computes the intersection between a cone, defined by a start angle and an opening, and a
    half-space, defined by a start angle.
    """

    angle_between_starts = (hs_start_angle - cone_start_angle) % (2 * np.pi)
    if angle_between_starts < np.pi:
        cone_start_angle = hs_start_angle
        opening = opening - angle_between_starts
    else:
        opening = min(opening, (hs_start_angle + np.pi - cone_start_angle) % (2 * np.pi))

    return cone_start_angle, opening


def coord_to_angle(x: float, y: float) -> tuple[float, float]:
    """
    Converts an (x, y) pair into its angle from the (1, 0) vector, as a value in [0, 2pi[, and its
    length.
    """

    r = np.sqrt(x**2 + y**2)

    if r == 0:
        raise ValueError("No angle")
    angle = np.arccos(x / r) if y >= 0 else 2 * np.pi - np.arccos(x / r)

    return angle, r


def angle_to_coord(angle: float, r: float = 1.0) -> tuple[float, float]:
    """
    Converts an angle in [0, 2pi[ from the (1, 0) vector, and a radius, into an (x, y) pair.
    """

    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return x, y


def compute_2d_non_conflicting_cone(matrix: np.ndarray) -> tuple[float, float]:
    """
    Computes the frontier of the non-conflicting cone from a matrix of 2-dimensional rows.
    Returns the result as an angle in [0, 2pi[ corresponding to the start of the cone, and an
    opening angle, that is <= pi and that can be negative if the cone is empty.

    This method currently does not handle the case where the cone is a straight line passing by the
    origin (when matrix is for instance [[1, 0],[-1, 0]]).

    :param matrix: Any real-valued [m, 2] matrix.
    """

    row_angles = [coord_to_angle(*row)[0] for row in matrix]

    # Compute the start of the non-conflicting half-space of each individual row.
    start_angles = [(angle - np.pi / 2) % (2 * np.pi) for angle in row_angles]

    # Combine these non-conflicting half-spaces to obtain the global non-conflicting cone.
    cone_start_angle = start_angles[0]
    opening = np.pi
    for hs_start_angle in start_angles[1:]:
        cone_start_angle, opening = combine_bounds(cone_start_angle, opening, hs_start_angle)

    return cone_start_angle, opening


def project(vector: torch.Tensor, onto: torch.Tensor) -> torch.Tensor:
    onto_normalized = onto / torch.linalg.norm(onto)
    projection = vector @ onto_normalized * onto_normalized
    return projection
