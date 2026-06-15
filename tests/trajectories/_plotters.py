from abc import ABC, abstractmethod
from typing import TypeAlias

import matplotlib.patheffects as pe
import numpy as np
from matplotlib import cm as cm, colors as mcolors, pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

Color: TypeAlias = str | tuple[float, float, float] | tuple[float, float, float, float]


class Plotter(ABC):
    """Abstract base class to modify a matplotlib Axes object."""

    @abstractmethod
    def __call__(self, ax: plt.Axes) -> None:
        pass

    def __add__(self, other: "Plotter") -> "Plotter":
        return MultiPlotter((self, other))


class EmptyPlotter(Plotter):
    """Plotter that does nothing"""

    def __call__(self, ax: plt.Axes) -> None:
        pass


class MultiPlotter(Plotter):
    """Plotter applying several plotters."""

    def __init__(self, plotters: tuple["Plotter", ...]) -> None:
        self.plotters = plotters

    def __call__(self, ax: plt.Axes) -> None:
        for plotter in self.plotters:
            plotter(ax)


class PointPlotter(Plotter, ABC):
    """Abstract plotter storing a single point."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class InitialPointPlotter(PointPlotter):
    """PointPlotter that can draw the initial point."""

    def __init__(self, x: float, y: float, color: Color = "black") -> None:
        super().__init__(x, y)
        self.color = color

    def __call__(self, ax: plt.Axes) -> None:
        ax.scatter(
            self.x, self.y, color=self.color, edgecolors="black", s=30, linewidth=0.7, zorder=3
        )


class OptimalPointPlotter(PointPlotter):
    """PointPlotter that can draw the optimal point."""

    def __init__(self, x: float, y: float, color: Color) -> None:
        super().__init__(x, y)
        self.color = color

    def __call__(self, ax: plt.Axes) -> None:
        ax.scatter(
            self.x,
            self.y,
            marker="*",
            color=self.color,
            zorder=3,
            s=60,
        )


class OptimalLinePlotter(Plotter):
    """
    Plotter that can draw a continuous path with uniform color linking the provided optimal points.
    """

    def __init__(self, points: np.ndarray, color: Color) -> None:
        self.points = points
        self.color = color

    def __call__(self, ax: plt.Axes) -> None:
        ax.plot(self.points[:, 0], self.points[:, 1], color=self.color, linewidth=2.0)


class AxesPlotter(Plotter):
    """Plotter that can draw the x=0 and y=0 axes."""

    def __call__(self, ax: plt.Axes) -> None:
        ax.axhline(y=0, color="black", linewidth=0.75, alpha=0.5)
        ax.axvline(x=0, color="black", linewidth=0.75, alpha=0.5)


class CirclePlotter(Plotter):
    """Plotter that can draw a circle."""

    def __init__(self, radius: float, color: Color) -> None:
        self.radius = radius
        self.color = color

    def __call__(self, ax: plt.Axes) -> None:
        circle = plt.Circle(
            (0, 0),
            self.radius,
            color=self.color,
            fill=False,
            linestyle="--",
            alpha=0.5,
            linewidth=1.5,
        )
        ax.add_patch(circle)


class ContourCirclesPlotter(MultiPlotter):
    """
    MultiPlotter that can draw several circles of different radii and colors, to make contour
    lines centered at zero.
    """

    def __init__(self) -> None:
        radiuses = [1.0, 2.5, 4, 5.5, 7, 8.5]
        colormap = cm.inferno_r
        norm = mcolors.Normalize(vmin=-1, vmax=max(radiuses))
        plotters = tuple(CirclePlotter(radius, colormap(norm(radius))) for radius in radiuses)
        super().__init__(plotters)


class SegmentPlotter(Plotter):
    """Plotter that can draw a single segment of a given color."""

    def __init__(self, xp: np.ndarray, yp: np.ndarray, color: Color) -> None:
        self.xp = xp
        self.yp = yp
        self.color = color

    def __call__(self, ax: plt.Axes) -> None:
        ax.plot(self.xp, self.yp, color=self.color, solid_capstyle="round", linewidth=1.5)


class PathPlotter(MultiPlotter):
    """Plotter that can draw a path of segments with colors varying along a gradient."""

    def __init__(self, points: np.ndarray) -> None:
        x_view = sliding_window_view(points[:, 0], window_shape=2)
        y_view = sliding_window_view(points[:, 1], window_shape=2)

        colors = PathPlotter._get_color_gradient("#FF0000", "#FFEE00", len(points) - 1)
        plotters = tuple(
            SegmentPlotter(xp, yp, color)
            for xp, yp, color in zip(x_view, y_view, colors, strict=False)
        )
        super().__init__(plotters)

    @staticmethod
    def _get_color_gradient(c1: str, c2: str, n: int) -> list[str]:
        """Given two hex colors, returns a color gradient with n colors."""

        assert n > 1
        c1_rgb = np.array(PathPlotter._hex_to_rgb(c1)) / 255
        c2_rgb = np.array(PathPlotter._hex_to_rgb(c2)) / 255
        mix_pcts = [x / (n - 1) for x in range(n)]
        rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
        return [
            "#" + "".join([format(round(val * 255), "02x") for val in item]) for item in rgb_colors
        ]

    @staticmethod
    def _hex_to_rgb(hex_str: str) -> list[int]:
        """Map a hex color string to an [R, G, B] list of ints."""
        return [int(hex_str[i : i + 2], 16) for i in range(1, 6, 2)]


class TrajPlotter(MultiPlotter):
    """Plotter that can draw a trajectory: initial point + path."""

    def __init__(self, points: np.ndarray, initial_point_color: Color) -> None:
        x = points[0, 0]
        y = points[0, 1]
        plotters = (InitialPointPlotter(x, y, initial_point_color), PathPlotter(points))
        super().__init__(plotters)


class MultiTrajPlotter(MultiPlotter):
    """Plotter that can draw several trajectories (one for each initial point)."""

    CMAP = plt.get_cmap("Set2")

    def __init__(self, points_matrix: np.ndarray) -> None:
        plotters = tuple(
            TrajPlotter(points, self.CMAP(i)) for i, points in enumerate(points_matrix)
        )
        super().__init__(plotters)


class EvolutionPlotter(Plotter):
    """Plotter that can draw an evolution over the discrete timesteps."""

    def __init__(self, values: np.ndarray, color: Color) -> None:
        self.x = np.arange(len(values)) + 1
        self.y = values
        self.color = color

    def __call__(self, ax: plt.Axes) -> None:
        (line,) = ax.plot(self.x, self.y, color=self.color, linewidth=1.5)

        # Add thin black outline around the lines
        line.set_path_effects(
            [
                pe.Stroke(linewidth=2.1, foreground="black"),  # outline
                pe.Normal(),  # original line on top
            ]
        )
        ax.grid(linewidth=0.5)


class MultiEvolutionPlotter(MultiPlotter):
    """
    Plotter that can draw the evolution of some value over timestamps for each initial point.
    """

    CMAP = plt.get_cmap("Set2")

    def __init__(self, values_vector: np.ndarray) -> None:
        plotters = tuple(
            EvolutionPlotter(values, self.CMAP(i)) for i, values in enumerate(values_vector)
        )
        super().__init__(plotters)


class SetPlotter(Plotter):
    """
    Plotter that can represent an optimal set.

    If the provided array of optimal points contains a single point, the set will be represented by
    a star. Otherwise, it will be represented as a connected line plot. This does not necessarily
    work for all optimal sets, but it should be fine for those that are convex.
    """

    def __init__(self, points: np.ndarray, color: Color) -> None:
        self.points = points
        self.color = color

        if len(points) == 1:
            self.plotter: Plotter = OptimalPointPlotter(points[0, 0], points[0, 1], color=color)
        else:
            self.plotter = OptimalLinePlotter(points, color=color)

    def __call__(self, ax: plt.Axes) -> None:
        self.plotter(ax)


class SPSPlotter(SetPlotter):
    """Plotter that can represent the Strong Pareto stationary set: black SetPlotter"""

    def __init__(self, sps_points: np.ndarray) -> None:
        super().__init__(points=sps_points, color="#282828")


class PFPlotter(SetPlotter):
    """Plotter that can represent the Pareto front: black SetPlotter"""

    def __init__(self, pf_points: np.ndarray) -> None:
        super().__init__(points=pf_points, color="#282828")


class HeatmapPlotter(Plotter):
    """
    Plotter that can draw a heatmap with the given values extending between the provided
    coordinates.
    """

    def __init__(
        self,
        values: np.ndarray,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        vmin: float,
        vmax: float,
        cmap: str,
    ) -> None:
        self.values = values
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, ax: plt.Axes) -> None:
        ax.imshow(
            self.values.T,
            origin="lower",
            cmap=self.cmap,
            aspect="auto",
            vmin=self.vmin,
            vmax=self.vmax,
            extent=(self.x_min, self.x_max, self.y_min, self.y_max),
            alpha=0.4,
            interpolation="bicubic",
        )


class LimAdjuster(Plotter):
    """Plotter that adjusts the xlim and ylim of the plot to the specified xlim and ylim."""

    def __init__(self, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
        self.xlim = xlim
        self.ylim = ylim

    def __call__(self, ax: plt.Axes) -> None:
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)


class ContentLimAdjuster(LimAdjuster):
    """Plotter that adjusts the xlim and ylim of the plot to the coordinates of the content."""

    def __init__(self, content: np.ndarray) -> None:
        x_min, y_min = content.min(axis=0)
        x_max, y_max = content.max(axis=0)
        x_range = x_max - x_min
        y_range = y_max - y_min
        margin = 0.05
        super().__init__(
            xlim=(x_min - margin * x_range, x_max + margin * x_range),
            ylim=(y_min - margin * y_range, y_max + margin * y_range),
        )


class XTicksClearer(Plotter):
    """Plotter that hides the xticks."""

    def __call__(self, ax: plt.Axes) -> None:
        ax.tick_params(bottom=False, labelbottom=False)


class YTicksClearer(Plotter):
    """Plotter that hides the yticks."""

    def __call__(self, ax: plt.Axes) -> None:
        ax.tick_params(left=False, labelleft=False)


class XAxisLabeller(Plotter):
    """Plotter that labels the x-axis."""

    def __init__(self, xlabel: str) -> None:
        self.xlabel = xlabel

    def __call__(self, ax: plt.Axes) -> None:
        ax.set_xlabel(self.xlabel)


class YAxisLabeller(Plotter):
    """Plotter that labels the y-axis."""

    def __init__(self, ylabel: str) -> None:
        self.ylabel = ylabel

    def __call__(self, ax: plt.Axes) -> None:
        ax.set_ylabel(self.ylabel)


class TitleSetter(Plotter):
    """Plotter that sets the title."""

    def __init__(self, title: str) -> None:
        self.title = title

    def __call__(self, ax: plt.Axes) -> None:
        ax.set_title(self.title)


class SquareBoxAspectSetter(Plotter):
    """Plotter that sets a square box aspect."""

    def __call__(self, ax: plt.Axes) -> None:
        ax.set_box_aspect(1)
