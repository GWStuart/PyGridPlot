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
    """Return a plane covering the whole benchmark surface."""
    return pgp.Plane(pygame.Rect(0, 0, *SIZE), **kwargs)  # type: ignore[arg-type]


def grid_case(**kwargs: object) -> Case:
    """Draw just the grid of a plane created with ``kwargs``."""
    plane = plane_for(**kwargs)
    return plane.draw_grid


def plot_case(
    draw: Callable[[pgp.Plane, pygame.Surface], None], **kwargs: object
) -> Case:
    """Draw plots (without the grid) on a plane created with ``kwargs``."""
    plane = plane_for(**kwargs)

    def run(surface: pygame.Surface) -> None:
        plane.begin_frame()
        draw(plane, surface)

    return run


BIG, SMALL, PEN = 2.5, 0.9, 1.6
"""Wheel sizes of the spirograph (a hypotrochoid) case."""


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
