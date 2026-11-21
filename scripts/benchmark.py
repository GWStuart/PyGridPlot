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

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

import pygridplot as pgp

SIZE = (1100, 720)
"""Surface size used for every case."""

Case = Callable[[pygame.Surface], None]
"""Draws one benchmark case onto a surface."""


def plane_for(**kwargs: object) -> pgp.Plane:
    """Return a plane covering the whole benchmark surface.

    The cache is off unless asked for, so that the cases measure the work of
    drawing, not of reusing the previous frame.
    """
    kwargs.setdefault("cache", False)
    return pgp.Plane(pygame.Rect(0, 0, *SIZE), **kwargs)  # type: ignore[arg-type]


def grid_case(**kwargs: object) -> Case:
    """Draw just the grid of a plane created with ``kwargs``."""
    plane = plane_for(**kwargs)
    return plane.draw_grid


def plot_case(
    draw: Callable[[pgp.Plane, pygame.Surface], None],
    *,
    draggables: int = 0,
    **kwargs: object,
) -> Case:
    """Draw plots (without the grid) on a plane created with ``kwargs``.

    ``draggables`` free draggable points are added, spread over the plane,
    the first of them hovered.
    """
    plane = plane_for(**kwargs)
    for k in range(draggables):
        point = plane.draggable((k % 5 * 2.0 - 4.0, k // 5 * 1.5 - 2.0))
        point.set_interaction(hovered=k == 0, dragging=False)

    def run(surface: pygame.Surface) -> None:
        plane.begin_frame()
        draw(plane, surface)

    return run


def cached_case(
    draw: Callable[[pgp.Plane, pygame.Surface], None], **kwargs: object
) -> Case:
    """Draw the grid and plots with the cache on, nothing changing between frames.

    This is an idle frame: what a program costs while the user only looks.
    """
    plane = plane_for(cache=True, **kwargs)

    def run(surface: pygame.Surface) -> None:
        plane.draw_grid(surface)
        draw(plane, surface)

    return run


def damped(x: np.ndarray) -> np.ndarray:
    """A damped wave, differentiated exactly in the derivative cases."""
    return np.sin(x) * np.exp(-x / 5)


def damped_math(x: float) -> float:
    """The same wave with :mod:`math`, differentiated numerically."""
    return math.sin(x) * math.exp(-x / 5)


def drag_case() -> Case:
    """Drag a point along a spirograph: one mouse motion per call."""
    plane = plane_for(scale=60.0)
    point = plane.draggable(
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


BIG, SMALL, PEN = 2.5, 0.9, 1.6
"""Wheel sizes of the spirograph (a hypotrochoid) case."""


GRAPH_LINES = (
    "a = 1.5",
    "f(x) = a sin(x) + x / 4",
    "y = f'(x)",
    "y = f'(P.x)(x - P.x) + P.y",
    "(3cos(t) + cos(3t), 2sin(t))",
    "r = 1 + cos(3θ) / 2",
    "x^2 / 9 + y^2 / 4 = 1",
    "Q = (P.x + 1, f(P.x + 1))",
)
"""A typical graph: parameters, plots, a derivative, curves and points."""


def graph_for(**options: object) -> pgp.Graph:
    """Return the graph of :data:`GRAPH_LINES`, with a draggable point P."""
    graph = pgp.Graph()
    for line in GRAPH_LINES:
        graph.add(line)
    graph.draggable(curve="f", t=1.2, name="P", **options)  # type: ignore[arg-type]
    return graph


def by_hand(plane: pgp.Plane, surface: pygame.Surface) -> None:
    """Draw what :func:`graph_for` holds with plane calls written in NumPy."""
    point = by_hand.point  # type: ignore[attr-defined]
    a = 1.5

    def f(x: np.ndarray) -> np.ndarray:
        return a * np.sin(x) + x / 4

    df = pgp.derivative(f)
    plane.plot(surface, f)
    plane.plot(surface, df)
    px, py = point.position
    slope = float(df(np.array([px]))[0])
    plane.plot(surface, lambda x: slope * (x - px) + py)
    plane.parametric(
        surface,
        lambda t: 3 * np.cos(t) + np.cos(3 * t),
        lambda t: 2 * np.sin(t),
        (0, 2 * math.pi),
    )
    plane.polar(surface, lambda th: 1 + np.cos(3 * th) / 2)
    plane.implicit(surface, lambda x, y: x**2 / 9 + y**2 / 4, 1.0)
    plane.points(surface, [(px + 1, float(f(np.array([px + 1]))[0]))])
    plane.draw_draggables(surface, point)


def graph_case(*, cache: bool = False) -> Case:
    """Draw the grid and the graph of :data:`GRAPH_LINES`."""
    plane = plane_for(scale=60.0, cache=cache)
    graph = graph_for()

    def run(surface: pygame.Surface) -> None:
        plane.draw_grid(surface)
        graph.draw(plane, surface)

    return run


def by_hand_case() -> Case:
    """Draw the grid and the same as :func:`graph_case`, by hand."""
    plane = plane_for(scale=60.0)
    f = pgp.expression("1.5 sin(x) + x / 4")
    by_hand.point = plane.draggable(  # type: ignore[attr-defined]
        curve=(lambda t: t, f, (-30.0, 30.0)), t=1.2
    )

    def run(surface: pygame.Surface) -> None:
        plane.draw_grid(surface)
        by_hand(plane, surface)

    return run


def compile_case() -> Case:
    """Compile the graph of :data:`GRAPH_LINES` again, as after any edit."""
    graph = graph_for()
    first = graph.items[1]

    def run(surface: pygame.Surface) -> None:
        first.expression = first.expression  # type: ignore[attr-defined]
        assert not graph.errors  # compiles

    return run


def cases() -> dict[str, Case]:
    """Return every benchmark case by name."""
    test_set = (lambda x: 1 / x, np.tan, np.sqrt, np.log, np.floor, lambda x: x**2)
    xs = np.linspace(-8, 8, 2000)
    ellipse = np.column_stack(
        (
            8 * np.cos(np.linspace(0, 2 * math.pi, 200)),
            5 * np.sin(np.linspace(0, 2 * math.pi, 200)),
        )
    )
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
        "plot 1st derivative (NumPy)": plot_case(
            lambda p, s: p.plot(s, pgp.derivative(damped)), scale=60.0
        ),
        "plot 4th derivative (NumPy)": plot_case(
            lambda p, s: p.plot(s, pgp.derivative(damped, 4)), scale=60.0
        ),
        "plot 1st derivative (math)": plot_case(
            lambda p, s: p.plot(s, pgp.derivative(damped_math)), scale=60.0
        ),
        "drag along a spirograph": drag_case(),
        "draw 20 draggable points": plot_case(
            lambda p, s: p.draw_draggables(s), scale=60.0, draggables=20
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
        "graph: 8 lines + point": graph_case(),
        "graph: same, by hand": by_hand_case(),
        "graph: compile 8 lines": compile_case(),
        "cached: idle graph": graph_case(cache=True),
        "points 2000": plot_case(
            lambda p, s: p.points(s, np.column_stack((xs, np.sin(xs))))
        ),
        "polygon big fill": plot_case(lambda p, s: p.polygon(s, ellipse), scale=55.0),
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
