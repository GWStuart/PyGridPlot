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
        "cached: idle vector field": cached_case(
            lambda p, s: p.vector_field(s, lambda x, y: (-y, x)), scale=50.0
        ),
        "cached: idle antiderivative": cached_case(
            lambda p, s: p.plot(s, pgp.antiderivative(damped)), scale=60.0
        ),
        "cached: idle inequality disc": cached_case(
            lambda p, s: p.inequality(s, lambda x, y: x**2 + y**2 < 16), scale=60.0
        ),
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
