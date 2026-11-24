"""Measure how long the main per-frame operations take.

This is a development tool for checking that a change makes things faster,
or at least no slower. It runs headless and prints the median time of each
case in milliseconds::

    python scripts/benchmark.py               # every case
    python scripts/benchmark.py --only grid   # cases whose name contains "grid"
    python scripts/benchmark.py --repeat 50   # more repeats, steadier numbers

Times vary from run to run (CPU frequency, caches, other programs), so
compare medians of a few runs, and compare on the same machine.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import replace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

import pygridplot as pgp

SIZE = (1100, 720)
"""Surface size used for every case."""

Case = Callable[[pygame.Surface], None]
"""Draws one benchmark case onto a surface."""


def grid_for(**kwargs: object) -> pgp.Grid:
    """Return a plane covering the whole benchmark surface.

    The cache is off unless asked for, so that the cases measure the work of
    drawing, not of reusing the previous frame.
    """
    kwargs.setdefault("cache", False)
    return pgp.Grid(pygame.Rect(0, 0, *SIZE), **kwargs)  # type: ignore[arg-type]


def grid_case(**kwargs: object) -> Case:
    """Draw just the grid of a plane created with ``kwargs``."""
    grid = grid_for(**kwargs)
    return grid.draw


def plot_case(
    draw: Callable[[pgp.Grid, pygame.Surface], None],
    *,
    draggables: int = 0,
    **kwargs: object,
) -> Case:
    """Draw plots (without the grid) on a plane created with ``kwargs``.

    ``draggables`` free draggable points are added, spread over the plane,
    the first of them hovered.
    """
    grid = grid_for(**kwargs)
    for k in range(draggables):
        point = grid.draggable((k % 5 * 2.0 - 4.0, k // 5 * 1.5 - 2.0))
        point.set_interaction(hovered=k == 0, dragging=False)

    def run(surface: pygame.Surface) -> None:
        grid.begin_frame()
        draw(grid, surface)

    return run


def cached_case(
    draw: Callable[[pgp.Grid, pygame.Surface], None], **kwargs: object
) -> Case:
    """Draw the grid and plots with the cache on, nothing changing between frames.

    This is an idle frame: what a program costs while the user only looks.
    """
    grid = grid_for(cache=True, **kwargs)

    def run(surface: pygame.Surface) -> None:
        grid.draw(surface)
        draw(grid, surface)

    return run


def damped(x: np.ndarray) -> np.ndarray:
    """A damped wave, differentiated exactly in the derivative cases."""
    return np.sin(x) * np.exp(-x / 5)


def damped_math(x: float) -> float:
    """The same wave with :mod:`math`, differentiated numerically."""
    return math.sin(x) * math.exp(-x / 5)


def drag_case() -> Case:
    """Drag a point along a spirograph: one mouse motion per call."""
    grid = grid_for(scale=60.0)
    point = grid.draggable(
        curve=(
            lambda t: (
                (BIG - SMALL) * np.cos(t) + PEN * np.cos((BIG - SMALL) / SMALL * t)
            ),
            lambda t: (
                (BIG - SMALL) * np.sin(t) - PEN * np.sin((BIG - SMALL) / SMALL * t)
            ),
            (0, 18 * math.pi),
        )
    )
    step = [0]

    def run(surface: pygame.Surface) -> None:
        if step[0] % 30 == 0:
            point.begin_frame()  # a new frame: the curve is sampled again
        step[0] += 1
        sx, sy = point.screen_position
        point.drag_to(sx + 3.0, sy + 1.0)

    return run


def slider_case() -> Case:
    """Draw five sliders in two corners, one of them hovered."""
    grid = grid_for(scale=60.0)
    for k in range(4):
        grid.slider(f"s{k}", (0.0, 1.0), value=k / 4)
    grid.slider("n", (1, 50), value=7, step=1, corner="top-right")
    grid.sliders[0].set_interaction(hovered=True, dragging=False)
    return grid.draw_controls


def titled_grid_case(**kwargs: object) -> Case:
    """Draw the grid with arrowheads on the axes and both axis titles."""
    grid = grid_for(theme=replace(pgp.LIGHT, show_arrows=True), **kwargs)

    def run(surface: pygame.Surface) -> None:
        grid.draw(surface, x_title="time t (s)", y_title="speed v (m/s)")

    return run


def text_case() -> Case:
    """Write 20 labels, some of two lines, anchored round points."""
    grid = grid_for(scale=60.0)
    anchors = ("left", "bottom", "top-right", "center")

    def run(surface: pygame.Surface) -> None:
        for k in range(20):
            text = f"point {k}" if k % 3 else f"point {k}\n({k}, {k % 5})"
            grid.text(
                surface,
                text,
                (k % 5 * 3.0 - 6.0, k // 5 * 2.0 - 3.0),
                anchor=anchors[k % 4],
                gap=6,
            )

    return run


def legend_case(change: str) -> Case:
    """Draw a legend of eight entries.

    The legend's panel and swatches are reused while unchanged, so their
    coverage is not rasterised again. ``change`` is ``"nothing"`` (a legend
    at rest), ``"label"`` (a label changing every frame, as when it shows a
    value: only its text is drawn anew) or ``"all"`` (built and rasterised
    afresh every frame).
    """
    grid = grid_for(scale=60.0)
    styles = [
        pgp.LineStyle(),
        pgp.FillStyle(),
        pgp.PolygonStyle(),
        pgp.PointStyle(),
        pgp.SequenceStyle(),
        pgp.ArrowStyle(),
        pgp.LineStyle(width=1.5, opacity=0.5),
        pgp.PointStyle(marker="diamond"),
    ]
    frame = [0]

    def run(surface: pygame.Surface) -> None:
        grid.begin_frame()
        frame[0] += 1
        if change == "all":
            grid._legend_drawings.clear()
        for k, style in enumerate(styles):
            label = f"entry {k}"
            if change == "label" and k == 0:
                label += f" = {frame[0] % 10}"
            grid._add_entry(label, style, grid.theme.palette[k % 6])
        grid.legend(surface)

    return run


def color_bar_case(*, afresh: bool) -> Case:
    """Draw a colour bar with a title, the same every frame or built afresh."""
    grid = grid_for(scale=60.0)

    def run(surface: pygame.Surface) -> None:
        grid.begin_frame()
        if afresh:
            grid._color_bar_drawing = None
        grid.color_bar(surface, "viridis", (0, 10), title="value")

    return run


def category_grid_case(names: list[str]) -> Case:
    """Draw the grid with names along x, zoomed out so they are thinned."""
    grid = grid_for(x_range=(-2.0, 102.0))

    def run(surface: pygame.Surface) -> None:
        grid.draw(surface, x_labels=names)

    return run


def graph_drag_case() -> Case:
    """Drag a point along a wavy graph, sideways: one mouse motion per call."""
    grid = grid_for(scale=60.0)
    point = grid.draggable(
        curve=(lambda t: t, lambda t: np.sin(t) + 0.3 * np.sin(2.7 * t), (-12, 12))
    )
    step = [0]

    def run(surface: pygame.Surface) -> None:
        if step[0] % 30 == 0:
            point.begin_frame()  # a new frame: the curve is sampled again
        step[0] += 1
        sx, _ = point.screen_position
        point.drag_to(100 + (sx + 3.0) % 900, 200.0)

    return run


BIG, SMALL, PEN = 2.5, 0.9, 1.6
"""Wheel sizes of the spirograph (a hypotrochoid) case."""


def cases() -> dict[str, Case]:
    """Return every benchmark case by name."""
    test_set = (lambda x: 1 / x, np.tan, np.sqrt, np.log, np.floor, lambda x: x**2)
    xs = np.linspace(-8, 8, 2000)
    starts200 = np.random.default_rng(0).uniform(-6, 6, (200, 2))
    ellipse = np.column_stack(
        (
            8 * np.cos(np.linspace(0, 2 * math.pi, 200)),
            5 * np.sin(np.linspace(0, 2 * math.pi, 200)),
        )
    )
    square50 = pgp.fourier(np.sign, 50)
    spectrum = pgp.fourier(np.sign, 100).amplitudes
    wave20000 = np.sin(np.arange(20000) / 300)
    rng = np.random.default_rng(0)
    scatter = rng.normal(size=(2000, 2)) * 3
    scatter_colors = rng.integers(0, 256, (2000, 3))
    scatter_sizes = rng.uniform(2, 8, 2000)
    normal100k = rng.normal(size=100_000) * 2
    heights = rng.uniform(1, 5, 20)
    speeds = np.hypot(scatter[:, 0], scatter[:, 1])
    groups1k = [rng.normal(k, 1 + k / 4, 1000) for k in range(4)]
    groups10k = [rng.normal(k, 1 + k / 4, 10_000) for k in range(4)]
    names = [f"category {k}" for k in range(100)]
    taylor10 = pgp.taylor(damped, 0.0, 10)
    return {
        "grid linear": grid_case(),
        "grid linear unequal": grid_case(x_range=(-3, 7), y_range=(-0.01, 0.02)),
        "grid log-log": grid_case(log_x=True, log_y=True),
        "grid log fine": grid_case(log_x=True, log_y=True, scale=1500.0, center=(3, 3)),
        "plot test set": plot_case(
            lambda p, s: [p.plot(s, f) for f in test_set], scale=60.0
        ),
        "plot sin(1/x) zoomed": plot_case(
            lambda p, s: p.plot(s, lambda x: np.sin(1 / x)), scale=300.0
        ),
        "plot thick parabola": plot_case(
            lambda p, s: p.plot(s, lambda x: x**2 / 4, width=8)
        ),
        "curve circle": plot_case(
            lambda p, s: p.parametric(s, np.cos, np.sin, (0, 2 * math.pi)), scale=150.0
        ),
        "curve spirograph": plot_case(
            lambda p, s: p.parametric(
                s,
                lambda t: (
                    (BIG - SMALL) * np.cos(t) + PEN * np.cos((BIG - SMALL) / SMALL * t)
                ),
                lambda t: (
                    (BIG - SMALL) * np.sin(t) - PEN * np.sin((BIG - SMALL) / SMALL * t)
                ),
                (0, 18 * math.pi),
                width=1.8,
            ),
            scale=60.0,
        ),
        "curve spiral 12 turns": plot_case(
            lambda p, s: p.polar(s, lambda a: 0.15 * a, (0, 24 * math.pi)), scale=60.0
        ),
        "implicit circle": plot_case(
            lambda p, s: p.implicit(s, lambda x, y: x**2 + y**2, 16), scale=60.0
        ),
        "implicit heart": plot_case(
            lambda p, s: p.implicit(
                s, lambda x, y: (x**2 + y**2 - 4) ** 3 - 4 * x**2 * y**3
            ),
            scale=60.0,
        ),
        "implicit y = tan x": plot_case(
            lambda p, s: p.implicit(s, lambda x, y: y - np.tan(x)), scale=60.0
        ),
        "implicit busy sin*sin": plot_case(
            lambda p, s: p.implicit(s, lambda x, y: np.sin(3 * x) * np.sin(3 * y), 0.2),
            scale=60.0,
        ),
        "implicit scalar math": plot_case(
            lambda p, s: p.implicit(s, lambda x, y: math.sin(x) - y), scale=60.0
        ),
        "inequality disc": plot_case(
            lambda p, s: p.inequality(s, lambda x, y: x**2 + y**2 < 16), scale=60.0
        ),
        "inequality y < tan x": plot_case(
            lambda p, s: p.inequality(s, lambda x, y: y < np.tan(x)), scale=60.0
        ),
        "inequality busy sin*sin": plot_case(
            lambda p, s: p.inequality(
                s, lambda x, y: np.sin(3 * x) * np.sin(3 * y) > 0.2
            ),
            scale=60.0,
        ),
        "inequality scalar math": plot_case(
            lambda p, s: p.inequality(s, lambda x, y: y < math.sin(x)), scale=60.0
        ),
        "fill between sin and x/3": plot_case(
            lambda p, s: p.fill_between(s, np.sin, lambda x: x / 3), scale=60.0
        ),
        "fill between tan and 0": plot_case(
            lambda p, s: p.fill_between(s, np.tan), scale=60.0
        ),
        "vector field at rest": plot_case(
            lambda p, s: p.vector_field(s, lambda x, y: (-y, x)), scale=50.0
        ),
        "vector field fading in": plot_case(
            lambda p, s: p.vector_field(s, lambda x, y: (-y, x)), scale=74.0
        ),
        "vector field scalar math": plot_case(
            lambda p, s: p.vector_field(s, lambda x, y: (math.sin(y), math.cos(x))),
            scale=50.0,
        ),
        "slope field x - y": plot_case(
            lambda p, s: p.slope_field(s, lambda x, y: x - y), scale=60.0
        ),
        "arrows 200 short": plot_case(
            lambda p, s: p.arrow(s, starts200, starts200 + 0.4), scale=60.0
        ),
        "plot 1st derivative (NumPy)": plot_case(
            lambda p, s: p.plot(s, pgp.derivative(damped)), scale=60.0
        ),
        "plot 4th derivative (NumPy)": plot_case(
            lambda p, s: p.plot(s, pgp.derivative(damped, 4)), scale=60.0
        ),
        "plot 1st derivative (math)": plot_case(
            lambda p, s: p.plot(s, pgp.derivative(damped_math)), scale=60.0
        ),
        "integral smooth (NumPy)": lambda s: pgp.integral(damped, -10, 10),
        "integral smooth (math)": lambda s: pgp.integral(damped_math, -10, 10),
        "integral 1/sqrt(x) from 0": lambda s: pgp.integral(
            lambda x: 1 / np.sqrt(x), 0, 1
        ),
        "integral exp(-x²) to ±inf": lambda s: pgp.integral(
            lambda x: np.exp(-(x**2)), -math.inf, math.inf
        ),
        "plot antiderivative (NumPy)": plot_case(
            lambda p, s: p.plot(s, pgp.antiderivative(damped)), scale=60.0
        ),
        "plot antiderivative of 1/x": plot_case(
            lambda p, s: p.plot(s, pgp.antiderivative(lambda x: 1 / x, 1.0)),
            scale=60.0,
        ),
        "fourier 100 terms (create)": lambda s: pgp.fourier(np.sign, 100),
        "plot fourier 50 terms": plot_case(
            lambda p, s: p.plot(s, square50), scale=60.0
        ),
        "plot taylor degree 10": plot_case(
            lambda p, s: p.plot(s, taylor10), scale=60.0
        ),
        "sequence spectrum, 101 terms": plot_case(
            lambda p, s: p.sequence(s, spectrum), scale=10.0
        ),
        "sequence (-1)^n/n": plot_case(
            lambda p, s: p.sequence(s, lambda n: (-1) ** n / n, start=1), scale=60.0
        ),
        "sequence zoomed out": plot_case(
            lambda p, s: p.sequence(s, np.sin), scale=(0.05, 60.0), equal_aspect=False
        ),
        "sequence 20000-term array": plot_case(
            lambda p, s: p.sequence(s, wave20000),
            scale=(0.05, 60.0),
            equal_aspect=False,
            center=(10000.0, 0.0),
        ),
        "draw 5 sliders": slider_case(),
        "grid with titles and arrows": titled_grid_case(),
        "grid log-log with titles and arrows": titled_grid_case(log_x=True, log_y=True),
        "text 20 labels": text_case(),
        "legend 8 entries": legend_case("nothing"),
        "legend 8 entries, a label changing": legend_case("label"),
        "legend 8 entries, built afresh": legend_case("all"),
        "drag along a spirograph": drag_case(),
        "drag along a graph": graph_drag_case(),
        "draw 20 draggable points": plot_case(
            lambda p, s: p.draw_controls(s), scale=60.0, draggables=20
        ),
        "cached: idle test set": cached_case(
            lambda p, s: [p.plot(s, f) for f in test_set], scale=60.0
        ),
        "cached: idle implicit heart": cached_case(
            lambda p, s: p.implicit(
                s, lambda x, y: (x**2 + y**2 - 4) ** 3 - 4 * x**2 * y**3
            ),
            scale=60.0,
        ),
        "cached: idle vector field": cached_case(
            lambda p, s: p.vector_field(s, lambda x, y: (-y, x)), scale=50.0
        ),
        "cached: idle antiderivative": cached_case(
            lambda p, s: p.plot(s, pgp.antiderivative(damped)), scale=60.0
        ),
        "cached: idle sequence": cached_case(
            lambda p, s: p.sequence(s, lambda n: (-1) ** n / n, start=1), scale=60.0
        ),
        "cached: idle inequality disc": cached_case(
            lambda p, s: p.inequality(s, lambda x, y: x**2 + y**2 < 16), scale=60.0
        ),
        "points 2000": plot_case(
            lambda p, s: p.points(s, np.column_stack((xs, np.sin(xs))))
        ),
        "polygon big fill": plot_case(lambda p, s: p.polygon(s, ellipse), scale=55.0),
        "points 2000, a colour each": plot_case(
            lambda p, s: p.points(s, scatter, color=scatter_colors), scale=60.0
        ),
        "points 2000, a colour and size each, outlined": plot_case(
            lambda p, s: p.points(
                s,
                scatter,
                color=scatter_colors,
                radius=scatter_sizes,
                edge_color="white",
            ),
            scale=60.0,
        ),
        "points 500 with x and y error bars": plot_case(
            lambda p, s: p.points(s, scatter[:500], x_error=0.2, y_error=0.3),
            scale=60.0,
        ),
        "polyline 2000 steps": plot_case(
            lambda p, s: p.polyline(s, np.column_stack((xs, np.sin(xs))), step="post")
        ),
        "bars 20": plot_case(lambda p, s: p.bars(s, heights, start=-10), scale=60.0),
        "histogram 10^5 samples": plot_case(
            lambda p, s: p.histogram(s, normal100k), scale=60.0
        ),
        "cached: idle histogram 10^5 samples": cached_case(
            lambda p, s: p.histogram(s, normal100k), scale=60.0
        ),
        "histogram 10^5 samples, cumulative": plot_case(
            lambda p, s: p.histogram(s, normal100k, cumulative=True), scale=60.0
        ),
        "points 2000 coloured by value (color_map)": plot_case(
            lambda p, s: p.points(
                s, scatter, color=pgp.color_map(speeds, "plasma", (0, 9))
            ),
            scale=60.0,
        ),
        "colour bar": color_bar_case(afresh=False),
        "colour bar, built afresh": color_bar_case(afresh=True),
        "stacked bars 3 x 20": plot_case(
            lambda p, s: [
                p.bars(s, heights * (k + 1), start=-10, baseline=heights * k)
                for k in range(3)
            ],
            scale=30.0,
        ),
        "box plot 4 groups of 1000": plot_case(
            lambda p, s: p.box_plot(s, groups1k), scale=60.0
        ),
        "violin plot 4 groups of 10^4": plot_case(
            lambda p, s: p.violin_plot(s, groups10k), scale=60.0
        ),
        "grid with 100 category names": category_grid_case(names),
        "dashed: plot test set": plot_case(
            lambda p, s: [p.plot(s, f, dash="dashed") for f in test_set], scale=60.0
        ),
        "dashed: plot thick parabola": plot_case(
            lambda p, s: p.plot(s, lambda x: x**2 / 4, width=8, dash="dashed")
        ),
        "dashed: curve spiral 12 turns, dotted": plot_case(
            lambda p, s: p.polar(
                s, lambda a: 0.15 * a, (0, 24 * math.pi), dash="dotted"
            ),
            scale=60.0,
        ),
        "dashed: implicit circle": plot_case(
            lambda p, s: p.implicit(s, lambda x, y: x**2 + y**2, 16, dash="dashed"),
            scale=60.0,
        ),
        "dashed: implicit busy sin*sin": plot_case(
            lambda p, s: p.implicit(
                s, lambda x, y: np.sin(3 * x) * np.sin(3 * y), 0.2, dash="dashed"
            ),
            scale=60.0,
        ),
        "dashed: polyline 2000 steps": plot_case(
            lambda p, s: p.polyline(
                s, np.column_stack((xs, np.sin(xs))), step="post", dash="dashed"
            )
        ),
        "dashed: polygon big fill": plot_case(
            lambda p, s: p.polygon(s, ellipse, dash="dashed"), scale=55.0
        ),
    }


def measure(case: Case, surface: pygame.Surface, repeat: int) -> float:
    """Return the median time of ``case`` in milliseconds, after a warm-up run."""
    case(surface)
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        case(surface)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark as configured on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default="", help="run cases whose name contains this")
    parser.add_argument("--repeat", type=int, default=20, help="timed runs per case")
    args = parser.parse_args(argv)
    pygame.init()
    pygame.display.set_mode((1, 1))
    surface = pygame.Surface(SIZE)
    for name, case in cases().items():
        if args.only in name:
            print(f"{name:24s} {measure(case, surface, args.repeat):7.2f} ms")
    pygame.quit()


if __name__ == "__main__":
    main()
