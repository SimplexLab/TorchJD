import logging
import os
import webbrowser
from collections.abc import Callable
from threading import Timer

import numpy as np
import torch
from dash import Dash, Input, Output, callback, dcc, html
from plotly.graph_objs import Figure

from plots._utils import Plotter, angle_to_coord, coord_to_angle
from torchjd.aggregation import (
    IMTLG,
    MGDA,
    Aggregator,
    AlignedMTL,
    CAGrad,
    ConFIG,
    DualProj,
    GradDrop,
    GradVac,
    Mean,
    NashMTL,
    PCGrad,
    Random,
    Sum,
    TrimmedMean,
    UPGrad,
)

MIN_LENGTH = 0.01
MAX_LENGTH = 25.0


def _format_angle_display(angle: float) -> str:
    return f"{angle:.4f} rad ({np.degrees(angle):.1f}°)"


def _format_length_display(r: float) -> str:
    return f"{r:.4f}"


def main() -> None:
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.CRITICAL)

    matrix = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, -1.0],
            [1.0, 0.0],
        ],
    )

    n_tasks = matrix.shape[0]
    aggregator_factories: dict[str, Callable[[], Aggregator]] = {
        "AlignedMTL-min": lambda: AlignedMTL(scale_mode="min"),
        "AlignedMTL-median": lambda: AlignedMTL(scale_mode="median"),
        "AlignedMTL-RMSE": lambda: AlignedMTL(scale_mode="rmse"),
        str(CAGrad(c=0.5)): lambda: CAGrad(c=0.5),
        str(ConFIG()): lambda: ConFIG(),
        str(DualProj()): lambda: DualProj(),
        str(GradDrop()): lambda: GradDrop(),
        str(GradVac()): lambda: GradVac(),
        str(IMTLG()): lambda: IMTLG(),
        str(Mean()): lambda: Mean(),
        str(MGDA()): lambda: MGDA(),
        str(NashMTL(n_tasks=n_tasks)): lambda: NashMTL(n_tasks=n_tasks),
        str(PCGrad()): lambda: PCGrad(),
        str(Random()): lambda: Random(),
        str(Sum()): lambda: Sum(),
        str(TrimmedMean(trim_number=1)): lambda: TrimmedMean(trim_number=1),
        str(UPGrad()): lambda: UPGrad(),
    }

    aggregator_strings = list(aggregator_factories.keys())

    plotter = Plotter(aggregator_factories, [], matrix)

    app = Dash(__name__)

    fig = plotter.make_fig()

    figure_div = html.Div(
        children=[dcc.Graph(id="aggregations-fig", figure=fig)],
        style={"display": "inline-block"},
    )

    seed_div = html.Div(
        [
            html.P("Seed", style={"display": "inline-block", "margin-right": 20}),
            dcc.Input(
                id="seed-selector",
                type="number",
                placeholder="",
                value=0,
                style={"display": "inline-block", "border": "1px solid black", "width": "25%"},
            ),
        ],
        style={"display": "inline-block", "width": "100%"},
    )

    gradient_divs = []
    gradient_slider_inputs = []
    for i in range(len(matrix)):
        initial_gradient = matrix[i]
        div, angle_input, r_input = make_gradient_div(i, initial_gradient)
        gradient_divs.append(div)

        gradient_slider_inputs.append(Input(angle_input, "value"))
        gradient_slider_inputs.append(Input(r_input, "value"))

    checklist = dcc.Checklist(aggregator_strings, [], id="aggregator-checklist")

    control_div = html.Div(
        children=[seed_div, *gradient_divs, checklist],
        style={"display": "inline-block", "vertical-align": "top"},
    )

    app.layout = html.Div([figure_div, control_div])

    @callback(
        Output("aggregations-fig", "figure", allow_duplicate=True),
        Input("seed-selector", "value"),
        prevent_initial_call=True,
    )
    def update_seed(value: int) -> Figure:
        plotter.seed = value
        return plotter.make_fig()

    n_gradients = len(matrix)
    gradient_value_outputs: list[Output] = []
    for i in range(n_gradients):
        gradient_value_outputs.append(Output(f"g{i + 1}-angle-value", "children"))
        gradient_value_outputs.append(Output(f"g{i + 1}-length-value", "children"))

    @callback(
        Output("aggregations-fig", "figure", allow_duplicate=True),
        *gradient_value_outputs,
        *gradient_slider_inputs,
        prevent_initial_call=True,
    )
    def update_gradient_coordinate(*values: str) -> tuple[Figure, ...]:
        values_ = [float(value) for value in values]

        display_parts: list[str] = []
        for j in range(len(values_) // 2):
            angle = values_[2 * j]
            r = values_[2 * j + 1]
            x, y = angle_to_coord(angle, r)
            plotter.matrix[j, 0] = x
            plotter.matrix[j, 1] = y
            display_parts.append(_format_angle_display(angle))
            display_parts.append(_format_length_display(r))

        return (plotter.make_fig(), *display_parts)

    @callback(
        Output("aggregations-fig", "figure", allow_duplicate=True),
        Input("aggregator-checklist", "value"),
        prevent_initial_call=True,
    )
    def update_aggregators(value: list[str]) -> Figure:
        plotter.selected_keys = list(value)
        return plotter.make_fig()

    Timer(1, open_browser).start()
    app.run(debug=False, port=1222)


def make_gradient_div(
    i: int,
    initial_gradient: torch.Tensor,
) -> tuple[html.Div, dcc.Input, dcc.Input]:
    x = initial_gradient[0].item()
    y = initial_gradient[1].item()
    angle, r = coord_to_angle(x, y)

    angle_input = dcc.Input(
        id=f"g{i + 1}-angle-range",
        type="range",
        value=angle,
        min=0,
        max=2 * np.pi,
        style={"width": "250px"},
    )

    r_input = dcc.Input(
        id=f"g{i + 1}-r-range",
        type="range",
        value=r,
        min=MIN_LENGTH,
        max=MAX_LENGTH,
        style={"width": "250px"},
    )

    label_style: dict[str, str | int] = {
        "display": "inline-block",
        "width": "52px",
        "margin-right": "8px",
        "vertical-align": "middle",
    }
    value_style: dict[str, str] = {
        "display": "inline-block",
        "margin-left": "10px",
        "min-width": "140px",
        "font-family": "monospace",
        "font-size": "13px",
        "vertical-align": "middle",
    }
    row_style: dict[str, str] = {"display": "block", "margin-bottom": "6px"}
    div = html.Div(
        [
            dcc.Markdown(
                f"$g_{{{i + 1}}}$",
                mathjax=True,
                style={
                    "margin": "0 0 6px 0",
                    "font-weight": "bold",
                    "display": "block",
                },
            ),
            html.Div(
                [
                    html.Span("Angle", style=label_style),
                    angle_input,
                    html.Span(
                        id=f"g{i + 1}-angle-value",
                        children=_format_angle_display(angle),
                        style=value_style,
                    ),
                ],
                style=row_style,
            ),
            html.Div(
                [
                    html.Span("Length", style=label_style),
                    r_input,
                    html.Span(
                        id=f"g{i + 1}-length-value",
                        children=_format_length_display(r),
                        style=value_style,
                    ),
                ],
                style={**row_style, "margin-bottom": "12px"},
            ),
        ],
    )
    return div, angle_input, r_input


def open_browser() -> None:
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new("http://127.0.0.1:1222/")


if __name__ == "__main__":
    main()
