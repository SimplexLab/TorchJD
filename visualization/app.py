import logging
from urllib.parse import parse_qs, urlencode

import numpy as np
import torch
from _utils import Plotter, angle_to_coord, coord_to_angle
from dash import Dash, Input, Output, State, dcc, html, no_update

from torchjd.aggregation import (
    IMTLG,
    MGDA,
    AlignedMTL,
    CAGrad,
    ConFIG,
    DualProj,
    FairGrad,
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
from torchjd.linalg import QuadprogProjector

logging.getLogger("werkzeug").setLevel(logging.CRITICAL)

MIN_LENGTH = 0.01
MAX_LENGTH = 25.0
N_TASKS = 3

DEFAULT_MATRIX = torch.tensor(
    [
        [0.0, 1.0],
        [1.0, -1.0],
        [1.0, 0.0],
    ]
)

AGGREGATOR_FACTORIES = {
    "AlignedMTL-min": lambda: AlignedMTL(scale_mode="min"),
    "AlignedMTL-median": lambda: AlignedMTL(scale_mode="median"),
    "AlignedMTL-RMSE": lambda: AlignedMTL(scale_mode="rmse"),
    "CAGrad": lambda: CAGrad(c=0.5),
    "ConFIG": lambda: ConFIG(),
    "DualProj": lambda: DualProj(projector=QuadprogProjector(reg_eps=1e-7)),
    "FairGrad": lambda: FairGrad(alpha=1.0),
    "GradDrop": lambda: GradDrop(),
    "GradVac": lambda: GradVac(),
    "IMTLG": lambda: IMTLG(),
    "Mean": lambda: Mean(),
    "MGDA": lambda: MGDA(),
    "NashMTL": lambda: NashMTL(n_tasks=N_TASKS),
    "PCGrad": lambda: PCGrad(),
    "Random": lambda: Random(),
    "Sum": lambda: Sum(),
    "TrimmedMean": lambda: TrimmedMean(trim_number=1),
    "UPGrad": lambda: UPGrad(projector=QuadprogProjector(reg_eps=1e-7)),
}

ALL_KEYS = list(AGGREGATOR_FACTORIES.keys())

# Default slider values derived from DEFAULT_MATRIX
_DEFAULT_ANGLES_RS: list[float] = []
for _i in range(N_TASKS):
    _x, _y = DEFAULT_MATRIX[_i, 0].item(), DEFAULT_MATRIX[_i, 1].item()
    _a, _r = coord_to_angle(_x, _y)
    _DEFAULT_ANGLES_RS.extend([_a, _r])


def _format_angle(angle: float) -> str:
    return f"{np.degrees(angle):.1f}°"


def _format_length(r: float) -> str:
    return f"{r:.2f}"


def _make_gradient_div(i: int, angle: float, r: float) -> html.Div:
    label_style = {
        "display": "inline-block",
        "width": "52px",
        "margin-right": "8px",
        "vertical-align": "middle",
    }
    value_style = {
        "display": "inline-block",
        "margin-left": "10px",
        "min-width": "140px",
        "font-family": "monospace",
        "font-size": "13px",
        "vertical-align": "middle",
    }
    row_style = {"display": "block", "margin-bottom": "6px"}

    return html.Div(
        [
            dcc.Markdown(
                f"$g_{{{i + 1}}}$",
                mathjax=True,
                style={"margin": "0 0 6px 0", "font-weight": "bold", "display": "block"},
            ),
            html.Div(
                [
                    html.Span("Angle", style=label_style),
                    dcc.Input(
                        id=f"g{i + 1}-angle",
                        type="range",
                        value=angle,
                        min=0,
                        max=2 * np.pi,
                        style={"width": "250px"},
                    ),
                    html.Span(
                        id=f"g{i + 1}-angle-display",
                        children=_format_angle(angle),
                        style=value_style,
                    ),
                ],
                style=row_style,
            ),
            html.Div(
                [
                    html.Span("Length", style=label_style),
                    dcc.Input(
                        id=f"g{i + 1}-r",
                        type="range",
                        value=r,
                        min=MIN_LENGTH,
                        max=MAX_LENGTH,
                        style={"width": "250px"},
                    ),
                    html.Span(
                        id=f"g{i + 1}-r-display",
                        children=_format_length(r),
                        style=value_style,
                    ),
                ],
                style={**row_style, "margin-bottom": "12px"},
            ),
        ]
    )


_default_fig = Plotter(AGGREGATOR_FACTORIES, [], DEFAULT_MATRIX.clone(), 0).make_fig()

_gradient_divs = [
    _make_gradient_div(i, _DEFAULT_ANGLES_RS[2 * i], _DEFAULT_ANGLES_RS[2 * i + 1])
    for i in range(N_TASKS)
]

app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        # Tracks whether URL params have been applied on first load
        dcc.Store(id="initialized", data=False),
        html.Div(
            [dcc.Graph(id="figure", figure=_default_fig)],
            style={"display": "inline-block"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P("Seed", style={"display": "inline-block", "margin-right": 20}),
                        dcc.Input(
                            id="seed",
                            type="number",
                            value=0,
                            style={
                                "display": "inline-block",
                                "border": "1px solid black",
                                "width": "25%",
                            },
                        ),
                    ],
                    style={"display": "inline-block", "width": "100%"},
                ),
                *_gradient_divs,
                dcc.Checklist(ALL_KEYS, [], id="agg-checklist"),
            ],
            style={"display": "inline-block", "vertical-align": "top"},
        ),
    ]
)

# Inputs/outputs shared by both callbacks
_gradient_angle_r_inputs = []
for _i in range(N_TASKS):
    _gradient_angle_r_inputs.extend(
        [
            Input(f"g{_i + 1}-angle", "value"),
            Input(f"g{_i + 1}-r", "value"),
        ]
    )

_display_outputs = []
for _i in range(N_TASKS):
    _display_outputs.extend(
        [
            Output(f"g{_i + 1}-angle-display", "children"),
            Output(f"g{_i + 1}-r-display", "children"),
        ]
    )

_gradient_angle_r_outputs = []
for _i in range(N_TASKS):
    _gradient_angle_r_outputs.extend(
        [
            Output(f"g{_i + 1}-angle", "value"),
            Output(f"g{_i + 1}-r", "value"),
        ]
    )


@app.callback(
    *_gradient_angle_r_outputs,
    Output("agg-checklist", "value"),
    Output("seed", "value"),
    Output("initialized", "data"),
    Input("url", "search"),
    State("initialized", "data"),
    prevent_initial_call=False,
)
def init_from_url(search: str | None, initialized: bool) -> tuple:
    """Reads URL query params once on page load and sets initial slider values."""
    n_outputs = N_TASKS * 2 + 3  # angles+rs + checklist + seed + initialized flag
    if initialized:
        return (*[no_update] * (n_outputs - 1), no_update)

    if not search:
        return (*_DEFAULT_ANGLES_RS, [], 0, True)

    params = parse_qs(search.lstrip("?"))

    agg_param = params.get("agg", [""])[0]
    selected = [a for a in agg_param.split(",") if a in AGGREGATOR_FACTORIES] if agg_param else []

    seed = max(0, int(params.get("seed", ["0"])[0]))

    angles_rs: list[float] = []
    for i in range(N_TASKS):
        default_angle = _DEFAULT_ANGLES_RS[2 * i]
        default_r = _DEFAULT_ANGLES_RS[2 * i + 1]
        g_param = params.get(f"g{i + 1}", [""])[0]
        if g_param:
            parts = g_param.split(",")
            if len(parts) == 2:
                try:
                    default_angle = float(parts[0])
                    default_r = max(MIN_LENGTH, min(MAX_LENGTH, float(parts[1])))
                except ValueError:
                    pass
        angles_rs.extend([default_angle, default_r])

    return (*angles_rs, selected, seed, True)


@app.callback(
    Output("figure", "figure"),
    Output("url", "search"),
    *_display_outputs,
    *_gradient_angle_r_inputs,
    Input("agg-checklist", "value"),
    Input("seed", "value"),
    prevent_initial_call=True,
)
def update_all(*args: object) -> tuple:
    """Updates the figure and URL whenever any control changes."""
    gradient_values = args[: N_TASKS * 2]
    selected = list(args[-2] or [])
    seed = max(0, int(args[-1] or 0))

    matrix = DEFAULT_MATRIX.clone()
    search_params: dict[str, str] = {}
    display_values: list[str] = []

    for i in range(N_TASKS):
        angle = float(gradient_values[2 * i])
        r = float(gradient_values[2 * i + 1])
        x, y = angle_to_coord(angle, r)
        matrix[i, 0] = x
        matrix[i, 1] = y
        search_params[f"g{i + 1}"] = f"{angle},{r}"
        display_values.extend([_format_angle(angle), _format_length(r)])

    if selected:
        search_params["agg"] = ",".join(selected)
    search_params["seed"] = str(seed)

    plotter = Plotter(AGGREGATOR_FACTORIES, selected, matrix, seed)
    fig = plotter.make_fig()
    search = "?" + urlencode(search_params)

    return (fig, search, *display_values)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
