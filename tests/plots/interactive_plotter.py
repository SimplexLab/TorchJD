import logging
import os
import webbrowser
from threading import Timer

import numpy as np
import torch
from dash import Dash, Input, Output, callback, dcc, html
from plotly.graph_objs import Figure

from plots._utils import Plotter, angle_to_coord, coord_to_angle
from torchjd.aggregation import (
    IMTLG,
    MGDA,
    AlignedMTL,
    CAGrad,
    ConFIG,
    DualProj,
    GradDrop,
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

    aggregators = [
        AlignedMTL(),
        CAGrad(c=0.5),
        ConFIG(),
        DualProj(),
        GradDrop(),
        IMTLG(),
        Mean(),
        MGDA(),
        NashMTL(n_tasks=matrix.shape[0]),
        PCGrad(),
        Random(),
        Sum(),
        TrimmedMean(trim_number=1),
        UPGrad(),
    ]

    aggregators_dict = {str(aggregator): aggregator for aggregator in aggregators}

    plotter = Plotter([], matrix)

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

    aggregator_strings = [str(aggregator) for aggregator in aggregators]
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

    @callback(
        Output("aggregations-fig", "figure", allow_duplicate=True),
        *gradient_slider_inputs,
        prevent_initial_call=True,
    )
    def update_gradient_coordinate(*values: str) -> Figure:
        values_ = [float(value) for value in values]

        for j in range(len(values_) // 2):
            angle = values_[2 * j]
            r = values_[2 * j + 1]
            x, y = angle_to_coord(angle, r)
            plotter.matrix[j, 0] = x
            plotter.matrix[j, 1] = y

        return plotter.make_fig()

    @callback(
        Output("aggregations-fig", "figure", allow_duplicate=True),
        Input("aggregator-checklist", "value"),
        prevent_initial_call=True,
    )
    def update_aggregators(value: list[str]) -> Figure:
        aggregator_keys = value
        new_aggregators = [aggregators_dict[key] for key in aggregator_keys]
        plotter.aggregators = new_aggregators
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

    div = html.Div(
        [
            html.P(f"g{i + 1}", style={"display": "inline-block", "margin-right": 20}),
            angle_input,
            r_input,
        ],
    )
    return div, angle_input, r_input


def open_browser() -> None:
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new("http://127.0.0.1:1222/")


if __name__ == "__main__":
    main()
