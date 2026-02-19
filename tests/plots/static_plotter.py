import os

import torch
from plotly import graph_objects as go

from plots._utils import (
    angle_to_coord,
    compute_2d_non_conflicting_cone,
    make_cone_scatter,
    make_polygon_scatter,
    make_right_angle,
    make_segment_scatter,
    make_vector_scatter,
    project,
)
from torchjd.aggregation import (
    MGDA,
    DualProj,
    Mean,
    UPGrad,
)

RIGHT_ANGLE_SIZE = 0.07


def main(
    *,
    gradients: bool = False,
    cone: bool = False,
    projections: bool = False,
    upgrad: bool = False,
    mean: bool = False,
    dual_proj: bool = False,
    mgda: bool = False,
) -> None:
    angle1 = 2.6
    angle2 = 0.3277
    norm1 = 0.9
    norm2 = 2.8
    g1 = torch.tensor(angle_to_coord(angle1, norm1))
    g2 = torch.tensor(angle_to_coord(angle2, norm2))
    matrix = torch.stack([g1, g2])
    g1_proj = g1 - project(g1, onto=g2)
    g2_proj = g2 - project(g2, onto=g1)
    filename = ""

    aggregators = {
        "UPGrad": UPGrad(),
        "Mean": Mean(),
        "DualProj": DualProj(),
        "MGDA": MGDA(),
    }
    results = {name: aggregator(matrix) for name, aggregator in aggregators.items()}

    fig = go.Figure()
    aggregation_labels = []  # Collect aggregator names to add labels as text elements at the end

    if gradients:
        filename += "gradients"
        for i in range(len(matrix)):
            label = r"$\huge{" + f"g_{i + 1}" + r"}$"

            gradient_scatter = make_vector_scatter(
                matrix[i],
                color="rgb(40, 40, 40)",
                label=label,
                showlegend=False,
                dash=False,
                textposition="bottom center",
                text_size=32,
                marker_size=22,
                line_width=4,
            )
            fig.add_trace(gradient_scatter)

    if cone:
        filename += "_cone"
        start_angle, opening = compute_2d_non_conflicting_cone(matrix.numpy())
        cone_scatter = make_cone_scatter(
            start_angle,
            opening,
            label="Non-conflicting cone",
            printable=False,
        )
        fig.add_trace(cone_scatter)

    if projections:
        filename += "_projections"
        g1_proj_segment = make_segment_scatter(g1, g1_proj)
        g2_proj_segment = make_segment_scatter(g2, g2_proj)
        origin_g1_proj_vector = make_vector_scatter(
            g1_proj,
            color="rgb(100, 100, 100)",
            label=r"$\huge{" + r"\pi_J(g_1)" + r"}$",
            line_width=3,
            marker_size=16,
            textposition="top left",
        )
        origin_g2_proj_vector = make_vector_scatter(
            g2_proj,
            color="rgb(100, 100, 100)",
            label=r"$\huge{" + r"\pi_J(g_2)" + r"}$",
            line_width=3,
            marker_size=16,
            textposition="top right",
        )

        g1_proj_right_angle = make_polygon_scatter(
            make_right_angle(g1_proj, size=RIGHT_ANGLE_SIZE, positive_para=False),
        )
        g2_proj_right_angle = make_polygon_scatter(
            make_right_angle(
                g2_proj,
                size=RIGHT_ANGLE_SIZE,
                positive_orth=False,
                positive_para=False,
            ),
        )

        fig.add_trace(g1_proj_segment)
        fig.add_trace(g2_proj_segment)

        fig.add_trace(g1_proj_right_angle)
        fig.add_trace(g2_proj_right_angle)

        fig.add_trace(origin_g1_proj_vector)
        fig.add_trace(origin_g2_proj_vector)

    if upgrad:
        filename += "_upgrad"
        g1_proj_upgrad_segment = make_segment_scatter(g1_proj, results["UPGrad"])
        g2_proj_upgrad_segment = make_segment_scatter(g2_proj, results["UPGrad"])

        fig.add_trace(g1_proj_upgrad_segment)
        fig.add_trace(g2_proj_upgrad_segment)

        name = "UPGrad"
        result = results[name]
        aggregation_scatter = make_vector_scatter(
            result,
            color="rgb(0, 0, 215)",
            label="",  # Label will be added as text element at the end
            textposition="top center",
            showlegend=False,
            dash=False,
            text_size=32,
            marker_size=22,
            line_width=4,
        )
        fig.add_trace(aggregation_scatter)
        aggregation_labels.append(name)

    if mean:
        filename += "_mean"
        g1_g2_segment = make_segment_scatter(g1, g2)

        fig.add_trace(g1_g2_segment)

        name = "Mean"
        result = results[name]
        aggregation_scatter = make_vector_scatter(
            result,
            color="rgb(0, 0, 215)",
            label="",  # Label will be added as text element at the end
            textposition="top center",
            showlegend=False,
            dash=False,
            text_size=32,
            marker_size=22,
            line_width=4,
        )
        fig.add_trace(aggregation_scatter)
        aggregation_labels.append(name)

    if dual_proj:
        filename += "_dual_proj"
        dual_proj_segment = make_segment_scatter(results["Mean"], results["DualProj"])

        dual_proj_right_angle = make_polygon_scatter(
            make_right_angle(
                results["DualProj"],
                size=RIGHT_ANGLE_SIZE,
                positive_orth=False,
                positive_para=False,
            ),
        )

        fig.add_trace(dual_proj_segment)
        fig.add_trace(dual_proj_right_angle)

        name = "DualProj"
        result = results[name]
        aggregation_scatter = make_vector_scatter(
            result,
            color="rgb(0, 0, 215)",
            label="",  # Label will be added as text element at the end
            textposition="top center",
            showlegend=False,
            dash=False,
            text_size=32,
            marker_size=22,
            line_width=4,
        )
        fig.add_trace(aggregation_scatter)
        aggregation_labels.append(name)

    if mgda:
        filename += "_mgda"
        if not mean:  # Otherwise the segment between g1 and g2 is already plotted
            g1_g2_segment = make_segment_scatter(g1, g2)
            fig.add_trace(g1_g2_segment)

        mgda_right_angle = make_polygon_scatter(
            make_right_angle(
                results["MGDA"],
                size=RIGHT_ANGLE_SIZE,
                positive_para=False,
                positive_orth=False,
            ),
        )
        fig.add_trace(mgda_right_angle)

        name = "MGDA"
        result = results[name]
        aggregation_scatter = make_vector_scatter(
            result,
            color="rgb(0, 0, 215)",
            label="",  # Label will be added as text element at the end
            textposition="top center",
            showlegend=False,
            dash=False,
            text_size=32,
            marker_size=22,
            line_width=4,
        )
        fig.add_trace(aggregation_scatter)
        aggregation_labels.append(name)

    # Add aggregation labels as text elements at the end so they appear on top
    for name in aggregation_labels:
        result = results[name]
        label_text = r"$\huge{\mathcal{A}_{\mathrm{" + name + r"}}(J)}$"
        fig.add_annotation(
            x=result[0].item(),
            y=result[1].item(),
            text=label_text,
            showarrow=False,
            font={"size": 32, "color": "rgb(0, 0, 215)"},
            yanchor="bottom",
            xanchor="center",
        )

    fig.update_layout(
        hovermode=False,
        width=912,
        height=528,
        plot_bgcolor="white",
        showlegend=False,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        range=[-0.95, 2.85],
        showgrid=False,
        zeroline=False,
        visible=False,
    )
    fig.update_yaxes(range=[-0.1, 2.1], showgrid=False, zeroline=False, visible=False)

    os.makedirs("images/", exist_ok=True)
    fig.write_image(f"images/{filename}.pdf")
    # Alternative: use .svg here and then convert to pdf using rsvg-convert. Install
    # [rsvg-convert](https://manpages.ubuntu.com/manpages/bionic/man1/rsvg-convert.1.html) and run:
    # `rsvg-convert -f pdf -o filename.pdf filename.svg`
    # To do that on all files at ones, run:
    # ```
    # for file in images/*.svg; do rsvg-convert -f pdf -o "${file%.svg}.pdf" "$file"; done
    # ```


if __name__ == "__main__":
    # Step-by-step construction of UPGrad for the presentation
    main(gradients=True)
    main(gradients=True, mean=True)
    main(gradients=True, mean=True, cone=True)
    main(gradients=True, mean=True, cone=True, projections=True)
    main(gradients=True, mean=True, cone=True, projections=True, upgrad=True)

    # Plot with UPGrad only
    main(gradients=True, cone=True, projections=True, upgrad=True)

    # Plot with Mean, DualProj and MGDA
    main(gradients=True, mean=True, cone=True, dual_proj=True, mgda=True)
