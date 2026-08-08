"""Configurable visualization of the Frank-Wolfe algorithm in two dimensions.

To use a different problem, only ``objective`` and ``DOMAIN`` in the problem
definition section need to be changed. All remaining values are determined
numerically from these inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import art3d


# ---------------------------------------------------------------------------
# Problem definition: These are the two inputs that need to be adjusted.
# DOMAIN must contain the vertices of a convex polygon in boundary order.
# NumPy operations in objective improve rendering speed; regular scalar
# functions are supported automatically as well.
# ---------------------------------------------------------------------------
DOMAIN = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    
])

# Desired starting point inside DOMAIN.
# Set START_POINT = None to use the centroid of the vertices automatically.
START_POINT = np.array([0.2, 0.32])


def objective(x1, x2):
    """Enter the desired objective function f(x1, x2) here."""
    return   6*(0*(x1-0.5)**2 + (x2-0.5)**2) +6


# Stop once the Frank-Wolfe gap is smaller than this tolerance.
TOLERANCE = 50e-2

# Safety limit for unsuitable or non-convex inputs.
MAX_ITERATIONS = 500
LINE_SAMPLES = np.linspace(0.0, 1.0, 220)
SAVE_ANIMATION = True
ANIMATION_FILE = "franke_wolfe_animation.gif"
GIF_FRAME_PIXEL_LIMIT = 100_000_000
GIF_LIMIT_SAFETY_FACTOR = 0.95
GIF_MAX_DPI = 120


@dataclass
class FrankWolfeProblem:
    """Objective function and polygonal feasible set of a 2D problem.

    ``start`` and ``gradient`` are optional. If omitted, the centroid of the
    vertices and central finite differences are used, respectively.
    """

    objective: Callable
    domain: np.ndarray
    start: np.ndarray | None = None
    gradient: Callable[[np.ndarray], np.ndarray] | None = None
    gradient_step: float = 1e-6

    def __post_init__(self):
        self.domain = np.asarray(self.domain, dtype=float)
        self._validate_domain()
        if self.start is None:
            self.start = self.domain.mean(axis=0)
        else:
            self.start = np.asarray(self.start, dtype=float)
        if self.start.shape != (2,) or not self._contains(self.start):
            raise ValueError("The starting point must lie inside DOMAIN.")

    def _validate_domain(self):
        if self.domain.ndim != 2 or self.domain.shape[1] != 2:
            raise ValueError("DOMAIN must have shape (number of vertices, 2).")
        if len(self.domain) < 3 or not np.all(np.isfinite(self.domain)):
            raise ValueError("DOMAIN requires at least three finite vertices.")

        edges = np.roll(self.domain, -1, axis=0) - self.domain
        next_edges = np.roll(edges, -1, axis=0)
        cross = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
        nonzero = cross[np.abs(cross) > 1e-12]
        if len(nonzero) == 0 or np.any(nonzero * nonzero[0] < 0):
            raise ValueError(
                "DOMAIN must describe a convex polygon in boundary order."
            )

    def _contains(self, point):
        edges = np.roll(self.domain, -1, axis=0) - self.domain
        relative = point - self.domain
        cross = edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0]
        return bool(np.all(cross >= -1e-10) or np.all(cross <= 1e-10))

    def evaluate(self, x1, x2):
        """Evaluate the function, with a fallback for non-vectorized functions."""
        x1_array, x2_array = np.broadcast_arrays(x1, x2)
        try:
            values = np.asarray(self.objective(x1_array, x2_array), dtype=float)
            values = np.broadcast_to(values, x1_array.shape)
        except (TypeError, ValueError):
            scalar_evaluation = np.vectorize(
                lambda a, b: float(self.objective(float(a), float(b))),
                otypes=[float],
            )
            values = scalar_evaluation(x1_array, x2_array)
        if not np.all(np.isfinite(values)):
            raise ValueError("The objective function returned non-finite values.")
        return float(values) if values.ndim == 0 else values

    def value(self, x):
        x = np.asarray(x, dtype=float)
        return self.evaluate(x[0], x[1])

    def grad(self, x):
        x = np.asarray(x, dtype=float)
        if self.gradient is not None:
            result = np.asarray(self.gradient(x), dtype=float)
        else:
            result = np.empty(2)
            for coordinate in range(2):
                h = self.gradient_step * max(1.0, abs(x[coordinate]))
                offset = np.zeros(2)
                offset[coordinate] = h
                result[coordinate] = (
                    self.value(x + offset) - self.value(x - offset)
                ) / (2.0 * h)
        if result.shape != (2,) or not np.all(np.isfinite(result)):
            raise ValueError("The gradient must have two finite components.")
        return result


PROBLEM = FrankWolfeProblem(
    objective=objective,
    domain=DOMAIN,
    start=START_POINT,
)


def line_search_gamma(problem, x, y, tolerance=1e-8):
    """Perform a numerical line search on the segment from ``x`` to ``y``."""
    direction = y - x
    if np.linalg.norm(direction) == 0.0:
        return 0.0

    # Golden-section search is robust and derivative-free in one dimension
    # for the convex objective function assumed by the Frank-Wolfe algorithm.
    left, right = 0.0, 1.0
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    value_c = problem.value(x + c * direction)
    value_d = problem.value(x + d * direction)
    while right - left > tolerance:
        if value_c <= value_d:
            right, d, value_d = d, c, value_c
            c = right - ratio * (right - left)
            value_c = problem.value(x + c * direction)
        else:
            left, c, value_c = c, d, value_d
            d = left + ratio * (right - left)
            value_d = problem.value(x + d * direction)

    candidates = np.array([0.0, 0.5 * (left + right), 1.0])
    values = [problem.value(x + gamma * direction) for gamma in candidates]
    return float(candidates[np.argmin(values)])


def frank_wolfe_gap(problem, x):
    """Compute the Frank-Wolfe point, gradient, and duality gap at ``x``."""
    gradient = problem.grad(x)
    fw_point = problem.domain[np.argmin(problem.domain @ gradient)]
    gap = float(gradient @ (x - fw_point))
    return fw_point, gradient, max(0.0, gap)


def frank_wolfe_steps(problem, tolerance, max_iterations=MAX_ITERATIONS):
    """Compute steps until the Frank-Wolfe tolerance is reached."""
    if tolerance <= 0:
        raise ValueError("TOLERANCE must be greater than 0.")
    if max_iterations < 1:
        raise ValueError("MAX_ITERATIONS must be at least 1.")
    x = problem.start.copy()
    steps = []

    for k in range(max_iterations):
        y, _, gap = frank_wolfe_gap(problem, x)
        if gap <= tolerance:
            break

        gamma = line_search_gamma(problem, x, y)
        direction = y - x
        x_new = x + gamma * direction
        segment = x + LINE_SAMPLES[:, None] * direction

        steps.append({
            "k": k,
            "x": x.copy(),
            "y": y.copy(),
            "x_new": x_new.copy(),
            "gamma": gamma,
            "fw_gap": gap,
            "segment_points": segment,
            "phi_values": problem.evaluate(segment[:, 0], segment[:, 1]),
        })
        x = x_new

    return steps


def estimate_minimum(problem, iterations=250):
    """Estimate a reference point using additional Frank-Wolfe steps."""
    x = problem.start.copy()
    for _ in range(iterations):
        gradient = problem.grad(x)
        y = problem.domain[np.argmin(problem.domain @ gradient)]
        gamma = line_search_gamma(problem, x, y)
        x_new = x + gamma * (y - x)
        if np.linalg.norm(x_new - x) <= 1e-9:
            break
        x = x_new
    return x


def print_steps(problem, steps, tolerance):
    """Print a compact summary of the computed iterations."""
    print("Frank-Wolfe iterations")
    for step in steps:
        print(
            f"k={step['k']:02d}: gap={step['fw_gap']:.3e}, "
            f"gamma={step['gamma']:.6f}, "
            f"x={step['x']}, x_new={step['x_new']}, "
            f"f(x_new)={problem.value(step['x_new']):.10f}"
        )

    final_x = steps[-1]["x_new"] if steps else problem.start
    _, _, final_gap = frank_wolfe_gap(problem, final_x)
    status = "Tolerance reached" if final_gap <= tolerance else "Safety limit reached"
    print(
        f"{status}: Frank-Wolfe gap={final_gap:.3e}, "
        f"tolerance={tolerance:.3e}, iterations={len(steps)}"
    )


def set_view_from_vector(ax, vx, vy, vz):
    """Set the 3D viewing angle from a direction vector."""
    azimuth = np.degrees(np.arctan2(vy, vx))
    elevation = np.degrees(np.arctan2(vz, np.hypot(vx, vy)))
    ax.view_init(elev=elevation, azim=azimuth)


def build_frames(step_count):
    """Build the sequence of stages for the animation frames."""
    frames = [("surface", -1, 1.0), ("domain", -1, 1.0), ("minimum", -1, 1.0)]
    for iteration in range(step_count):
        frames.extend((stage, iteration, 1.0) for stage in (
            "x", "plane", "fw_point", "direction"
        ))
        frames.append(("line_search", iteration, 1.0))
        frames.append(("new_point", iteration, 1.0))
    frames.extend([("finish", -1, 1.0)] * 3)
    return frames


class FrankWolfePlot:
    """Encapsulate the figure, drawing operations, and animation state."""

    _STAGE_LEVEL = {
        "x": 0,
        "plane": 1,
        "fw_point": 2,
        "direction": 3,
        "line_search": 4,
        "new_point": 5,
    }

    def __init__(self, problem, steps, tolerance):
        self.problem = problem
        self.steps = steps
        self.tolerance = tolerance
        self.reference_minimum = estimate_minimum(problem)
        self.frames = build_frames(len(steps))
        self.dynamic_artists = []
        self.label_effects = [
            patheffects.withStroke(linewidth=3.5, foreground="white")
        ]
        self.fig = plt.figure(figsize=(10, 9))
        self.fig.suptitle(
            rf"Frank-Wolfe for Convex Optimization"
            rf"  |  Stopping tolerance $\varepsilon={self.tolerance:g}$",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )
        self.ax = self.fig.add_subplot(
            111, projection="3d", computed_zorder=False
        )
        self._create_grids()
        self._configure_axes()
        self._draw_static_scene()

    def _create_grids(self):
        lower = self.problem.domain.min(axis=0)
        upper = self.problem.domain.max(axis=0)
        span = np.maximum(upper - lower, 1e-3)
        self.x_span, self.y_span = span
        surface_lower = lower - 0.12 * span
        surface_upper = upper + 0.12 * span
        plane_lower = lower - 0.20 * span
        plane_upper = upper + 0.20 * span
        surface_axes = (
            np.linspace(surface_lower[0], surface_upper[0], 140),
            np.linspace(surface_lower[1], surface_upper[1], 140),
        )
        plane_axes = (
            np.linspace(plane_lower[0], plane_upper[0], 190),
            np.linspace(plane_lower[1], plane_upper[1], 190),
        )
        self.surface_u, self.surface_v = np.meshgrid(*surface_axes)
        self.plane_u, self.plane_v = np.meshgrid(*plane_axes)
        self.surface_values = self.problem.evaluate(
            self.surface_u, self.surface_v
        )
        value_min = float(np.min(self.surface_values))
        value_max = float(np.max(self.surface_values))
        self.z_span = max(value_max - value_min, abs(value_max) * 0.05, 1e-3)
        self.floor_z = 0.0
        self.z_max = max(value_max + 0.28 * self.z_span, 0.25 * self.z_span)
        self.x_limits = (surface_lower[0], surface_upper[0])
        self.y_limits = (surface_lower[1], surface_upper[1])

    def _draw_static_scene(self):
        ax = self.ax
        ax.plot_surface(
            self.surface_u,
            self.surface_v,
            self.surface_values,
            cmap="Blues_r",
            alpha=0.98,
            linewidth=0,
            antialiased=True,
        )

        self.domain = Polygon(
            self.problem.domain, closed=True, facecolor="turquoise", edgecolor="teal",
            alpha=0.30, linewidth=2,
        )
        ax.add_patch(self.domain)
        art3d.pathpatch_2d_to_3d(self.domain, z=self.floor_z, zdir="z")

        minimum = self.reference_minimum
        minimum_value = self.problem.value(minimum)
        self.minimum_stem, = ax.plot(
            [minimum[0]] * 2,
            [minimum[1]] * 2,
            [self.floor_z, minimum_value + 0.06 * self.z_span],
            linestyle="--",
            color="deeppink",
            linewidth=2.4,
            zorder=85,
        )
        self.minimum_projection = ax.scatter(
            *minimum, self.floor_z + 0.01 * self.z_span,
            marker="*", s=125, color="deeppink",
            edgecolor="white", linewidth=1.0, depthshade=False, zorder=86,
        )
        self.minimum_marker = ax.scatter(
            *minimum, minimum_value + 0.03 * self.z_span, marker="o", s=135,
            color="#ff1493", edgecolor="none",
            depthshade=False, zorder=88,
        )
        self.minimum_label = ax.text(
            minimum[0] + 0.04 * self.x_span,
            minimum[1] + 0.02 * self.y_span,
            minimum_value + 0.12 * self.z_span,
            r"Global minimum $x^\ast$", fontsize=15, fontweight="bold",
            color="deeppink", zorder=90,
            path_effects=self.label_effects,
        )
        label_point = self.problem.domain[np.argmax(self.problem.domain[:, 1])]
        self.function_label = ax.text(
            label_point[0], label_point[1],
            self.problem.value(label_point) + 0.08 * self.z_span,
            r"$f$", fontsize=19,
            color="royalblue", zorder=90, path_effects=self.label_effects,
        )

        self.minimum_artists = (
            self.minimum_stem,
            self.minimum_projection,
            self.minimum_marker,
            self.minimum_label,
        )
        for artist in (self.domain, *self.minimum_artists, self.function_label):
            artist.set_visible(False)

    def _configure_axes(self):
        ax = self.ax
        ax.set(
            xlabel=r"$x_1$",
            ylabel=r"$x_2$",
            zlabel=r"$f(x_1,x_2)$",
            xlim=self.x_limits,
            ylim=self.y_limits,
            zlim=(self.floor_z, self.z_max),
        )
        ax.xaxis.label.set(fontsize=16)
        ax.yaxis.label.set(fontsize=16)
        ax.zaxis.label.set(fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=11)
        set_view_from_vector(ax, 1,0.5, 0.6)
        ax.set_proj_type("ortho")
        ax.set_box_aspect((1, 1, 0.85))
        ax.grid(True)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_alpha(0.04)

    def _track(self, *artist_groups):
        for group in artist_groups:
            if isinstance(group, (list, tuple)):
                self.dynamic_artists.extend(group)
            else:
                self.dynamic_artists.append(group)

    def _clear_dynamic(self):
        while self.dynamic_artists:
            artist = self.dynamic_artists.pop()
            try:
                artist.remove()
            except (AttributeError, ValueError, NotImplementedError):
                pass

    def _text(self, x1, x2, z, text, color, fontsize=15):
        artist = self.ax.text(
            x1, x2, z, text, fontsize=fontsize, color=color, zorder=90,
            path_effects=self.label_effects,
        )
        self._track(artist)

    def _draw_current_point(self, step):
        x, value = step["x"], self.problem.value(step["x"])
        self._track(
            self.ax.scatter(
                *x, self.floor_z, s=78, color="gold",
                edgecolor="black", zorder=50,
            ),
            self.ax.scatter(
                *x, value + 0.006 * self.z_span,
                s=58, color="black", zorder=65,
            ),
            self.ax.plot(
                [x[0]] * 2, [x[1]] * 2, [self.floor_z, value], ":",
                color="black", linewidth=1.5,
            ),
        )
        self._text(
            x[0] + 0.02 * self.x_span, x[1],
            value + 0.06 * self.z_span,
            rf"$x_{step['k']}$", "black", 16,
        )

    def _draw_fw_point(self, step):
        y = step["y"]
        self._track(self.ax.scatter(
            *y, self.floor_z, s=82, color="yellowgreen",
            edgecolor="black", zorder=55,
        ))
        self._text(
            y[0] - 0.06 * self.x_span,
            y[1] - 0.02 * self.y_span,
            self.floor_z + 0.03 * self.z_span,
            rf"$s_{step['k']}$", "darkgreen", 16,
        )

    def _draw_direction(self, step):
        x, y = step["x"], step["y"]
        direction = y - x
        midpoint = 0.5 * (x + y)
        self._track(self.ax.quiver(
            *x, self.floor_z + 0.015 * self.z_span,
            *direction, 0.0, color="green", linewidth=3,
            arrow_length_ratio=0.08,
        ))
        self._text(
            midpoint[0], midpoint[1] - 0.04 * self.y_span,
            self.floor_z + 0.04 * self.z_span,
            rf"$d_{step['k']}=s_{step['k']}-x_{step['k']}$", "green", 13,
        )

    def _draw_tangent_plane(self, step):
        x = step["x"]
        gradient = self.problem.grad(x)
        plane = (
            self.problem.value(x)
            + gradient[0] * (self.plane_u - x[0])
            + gradient[1] * (self.plane_v - x[1])
        )
        plane_surface = self.ax.plot_surface(
            self.plane_u, self.plane_v, plane, color="crimson", alpha=0.30,
            linewidth=0, antialiased=True,
        )
        plane_grid = self.ax.plot_wireframe(
            self.plane_u,
            self.plane_v,
            plane,
            rstride=15,
            cstride=15,
            color="darkred",
            linewidth=0.65,
            alpha=0.72,
        )
        self._track(plane_surface, plane_grid)

    def _draw_line_search(self, step):
        points = step["segment_points"]
        self._track(self.ax.plot(
            points[:, 0], points[:, 1],
            step["phi_values"] + 0.007 * self.z_span,
            color="darkorange", linewidth=4.2, zorder=75,
        ))
        line_search_label = self.ax.text2D(
            0.97,
            0.05,
            (
                r"$\min_{\alpha \in [0,1]} f(x+\alpha(s-x))$"
                "\n"
                rf"$\alpha_{step['k']}={step['gamma']:.3f}$"
            ),
            transform=self.ax.transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
            color="darkorange",
            fontsize=14,
            zorder=100,
            path_effects=self.label_effects,
        )
        self._track(line_search_label)

    def _draw_new_point(self, step):
        x, value = step["x_new"], self.problem.value(step["x_new"])
        self._track(
            self.ax.scatter(
                *x, value + 0.008 * self.z_span, s=95, color="darkorange",
                edgecolor="black", zorder=85,
            ),
            self.ax.scatter(
                *x, self.floor_z, s=46, color="darkorange", zorder=60,
            ),
            self.ax.plot(
                [x[0]] * 2, [x[1]] * 2, [self.floor_z, value], ":",
                color="darkorange", linewidth=1.3,
            ),
        )
        self._text(
            x[0] + 0.02 * self.x_span,
            x[1] + 0.01 * self.y_span,
            value + 0.075 * self.z_span,
            rf"$x_{step['k'] + 1}$", "darkorange", 16,
        )

    def _draw_history(self, last_iteration):
        if last_iteration < 0:
            return
        points = np.vstack([
            self.steps[0]["x"],
            *(step["x_new"] for step in self.steps[:last_iteration + 1]),
        ])
        surface_history = self.ax.plot(
            points[:, 0], points[:, 1],
            self.problem.evaluate(points[:, 0], points[:, 1])
            + 0.012 * self.z_span,
            color="hotpink", linewidth=2.5, marker="o", markersize=4, zorder=78,
        )
        domain_history = self.ax.plot(
            points[:, 0],
            points[:, 1],
            np.full(len(points), self.floor_z + 0.01 * self.z_span),
            color="hotpink",
            linewidth=2.2,
            marker="o",
            markersize=6,
            markerfacecolor="gold",
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=79,
        )
        self._track(surface_history, domain_history)

    def _set_static_visibility(self, stage):
        self.domain.set_visible(stage != "surface")
        show_minimum = stage not in {"surface", "domain"}
        for artist in self.minimum_artists:
            artist.set_visible(show_minimum)
        self.function_label.set_visible(stage != "surface")

    def _draw_finish(self):
        self._draw_history(len(self.steps) - 1)
        final_x = self.steps[-1]["x_new"] if self.steps else self.problem.start
        _, _, fw_gap = frank_wolfe_gap(self.problem, final_x)
        result = (
            "Tolerance reached"
            if fw_gap <= self.tolerance
            else "Safety limit reached"
        )
        self.ax.set_title(
            rf"{result}: $g_{{FW}}(x_k)={fw_gap:.1e}$ "
            rf"$(\varepsilon={self.tolerance:.1e})$",
            fontsize=15,
            pad=18,
        )

    def update(self, frame_index):
        """Draw exactly one animation frame."""
        self._clear_dynamic()
        stage, iteration, progress = self.frames[frame_index]
        self._set_static_visibility(stage)

        intro_titles = {
            "surface": "1. Objective function f",
            "domain": "2. Feasible set D",
            "minimum": "3. Numerically estimated reference minimum",
        }
        if stage in intro_titles:
            self.ax.set_title(intro_titles[stage], fontsize=16, pad=18)
            return self.dynamic_artists
        if stage == "finish":
            self._draw_finish()
            return self.dynamic_artists

        step = self.steps[iteration]
        level = self._STAGE_LEVEL[stage]
        self._draw_history(iteration - 1)
        self._draw_current_point(step)
        if stage == "plane":
            self._draw_tangent_plane(step)
        if level >= 2:
            self._draw_fw_point(step)
        if level >= 3:
            self._draw_direction(step)
        if level >= 4:
            self._draw_line_search(step)
        if level >= 5:
            self._draw_new_point(step)

        titles = {
            "x": (
                rf"Iteration {iteration}: current point $x_{iteration}$ "
                rf"$(g_{{FW}}={step['fw_gap']:.1e})$"
            ),
            "plane": rf"Iteration {iteration}: linearization at $x_{iteration}$",
            "fw_point": rf"Iteration {iteration}: Frank-Wolfe point $s_{iteration}$",
            "direction": rf"Iteration {iteration}: direction $d_{iteration}=s_{iteration}-x_{iteration}$",
            "line_search": rf"Iteration {iteration}: numerical line search",
            "new_point": rf"Iteration {iteration}: new point $x_{iteration + 1}$",
        }
        self.ax.set_title(titles[stage], fontsize=15, pad=18)
        return self.dynamic_artists


def create_animation(problem, steps, tolerance):
    """Create the figure and animation from precomputed steps."""
    plot = FrankWolfePlot(problem, steps, tolerance)
    animation = FuncAnimation(
        plot.fig,
        plot.update,
        frames=len(plot.frames),
        interval=850,
        repeat=False,
        blit=False,
        cache_frame_data=False,
    )
    plot.fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return plot.fig, animation


def wiki_safe_gif_dpi(figure, frame_count):
    """Choose the highest GIF DPI that stays safely below the wiki limit."""
    figure_area = figure.get_figwidth() * figure.get_figheight()
    safe_limit = GIF_FRAME_PIXEL_LIMIT * GIF_LIMIT_SAFETY_FACTOR
    maximum_dpi = np.sqrt(safe_limit / (figure_area * frame_count))
    return max(1, min(GIF_MAX_DPI, int(np.floor(maximum_dpi))))


def run_animation(
    problem=PROBLEM,
    tolerance=TOLERANCE,
    max_iterations=MAX_ITERATIONS,
    save=SAVE_ANIMATION,
    filename=ANIMATION_FILE,
    show=True,
):
    """Compute and display (or save) the animation for a problem."""
    steps = frank_wolfe_steps(problem, tolerance, max_iterations)
    print_steps(problem, steps, tolerance)
    figure, animation = create_animation(problem, steps, tolerance)
    if save:
        frame_count = len(build_frames(len(steps)))
        gif_dpi = wiki_safe_gif_dpi(figure, frame_count)
        width = round(figure.get_figwidth() * gif_dpi)
        height = round(figure.get_figheight() * gif_dpi)
        frame_pixels = width * height * frame_count
        print(
            f"GIF export: {width} x {height} x {frame_count} frames "
            f"= {frame_pixels:,} frame pixels (DPI={gif_dpi})"
        )
        animation.save(filename, writer="pillow", fps=1, dpi=gif_dpi)
    if show:
        plt.show()
    return animation


def main():
    return run_animation()


if __name__ == "__main__":
    _animation = main()
