import logging

import gradio as gr
import numpy as np
import torch
from _utils import Plotter, angle_to_coord, coord_to_angle

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

_DEFAULT_ANGLES_RS: list[float] = []
for _i in range(N_TASKS):
    _x, _y = DEFAULT_MATRIX[_i, 0].item(), DEFAULT_MATRIX[_i, 1].item()
    _a, _r = coord_to_angle(_x, _y)
    _DEFAULT_ANGLES_RS.extend([float(_a), float(_r)])


def _build_matrix(angles_rs: list[float]) -> torch.Tensor:
    matrix = DEFAULT_MATRIX.clone()
    for i in range(N_TASKS):
        x, y = angle_to_coord(angles_rs[2 * i], angles_rs[2 * i + 1])
        matrix[i, 0] = x
        matrix[i, 1] = y
    return matrix


def update_plot(seed: float, *args: float | list[str]) -> gr.Plot:
    gradient_values = args[: N_TASKS * 2]
    selected = list(args[-1] or [])
    angles_rs = [float(v) if v is not None else 0.0 for v in gradient_values]
    matrix = _build_matrix(angles_rs)
    plotter = Plotter(AGGREGATOR_FACTORIES, selected, matrix, int(seed or 0))
    return plotter.make_fig()


def load_from_url(request: gr.Request) -> list:
    params = dict(request.query_params)

    agg_param = params.get("agg", "")
    selected = [a for a in agg_param.split(",") if a in AGGREGATOR_FACTORIES] if agg_param else []

    seed = max(0, int(params.get("seed", 0) or 0))

    angles_rs = list(_DEFAULT_ANGLES_RS)
    for i in range(N_TASKS):
        g_param = params.get(f"g{i + 1}", "")
        if g_param:
            parts = g_param.split(",")
            if len(parts) == 2:
                try:
                    angles_rs[2 * i] = float(parts[0])
                    angles_rs[2 * i + 1] = max(MIN_LENGTH, min(MAX_LENGTH, float(parts[1])))
                except ValueError:
                    pass

    matrix = _build_matrix(angles_rs)
    plotter = Plotter(AGGREGATOR_FACTORIES, selected, matrix, seed)
    fig = plotter.make_fig()

    return [fig, float(seed), *[float(v) for v in angles_rs], selected]


with gr.Blocks(title="TorchJD Interactive Plotter") as demo:
    with gr.Row():
        with gr.Column(scale=3):
            plot = gr.Plot()
        with gr.Column(scale=1):
            seed_input = gr.Number(value=0, label="Seed", precision=0)

            gradient_sliders: list[gr.Slider] = []
            for i in range(N_TASKS):
                gr.Markdown(f"**$g_{{{i + 1}}}$**")
                angle_slider = gr.Slider(
                    minimum=0,
                    maximum=2 * np.pi,
                    value=_DEFAULT_ANGLES_RS[2 * i],
                    step=0.01,
                    label=f"g{i + 1} angle (rad)",
                )
                r_slider = gr.Slider(
                    minimum=MIN_LENGTH,
                    maximum=MAX_LENGTH,
                    value=_DEFAULT_ANGLES_RS[2 * i + 1],
                    step=0.01,
                    label=f"g{i + 1} length",
                )
                gradient_sliders.extend([angle_slider, r_slider])

            agg_check = gr.CheckboxGroup(ALL_KEYS, label="Aggregators", value=[])

    all_inputs = [seed_input, *gradient_sliders, agg_check]

    for component in all_inputs:
        component.change(update_plot, inputs=all_inputs, outputs=plot)

    demo.load(load_from_url, inputs=None, outputs=[plot, seed_input, *gradient_sliders, agg_check])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
