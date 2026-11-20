"""Plotting explicit functions, including the awkward ones.

Press ``Space`` to switch between sets of functions:

1. The shared test set from the project plan: ``1/x``, ``tan(x)``,
   ``sqrt(x)``, ``log(x)``, ``floor(x)`` and ``x**2``.
2. Functions written with :mod:`math`, which only work on single numbers.
   PyGridPlot falls back to calling them point by point.
3. Hard cases: a removable singularity, a fast oscillation and a function
   that raises an error for half of its inputs.

Pan and zoom as usual. Asymptotes and jumps are never joined by vertical
lines, and gaps appear wherever a function is undefined.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.render.fonts import bundled_font_path

FunctionSet = list[tuple[str, Callable[[object], object]]]


def safe_sqrt_of_sin(x: float) -> float:
    """Return ``sqrt(sin(x))``, raising ``ValueError`` where it is undefined.

    Args:
        x: A single number.

    Returns:
        The square root of ``sin(x)``.
    """
    return math.sqrt(math.sin(x))  # raises for negative values: drawn as gaps


SETS: list[tuple[str, FunctionSet]] = [
    (
        "shared test set",
        [
            ("1/x", lambda x: 1 / x),
            ("tan(x)", np.tan),
            ("sqrt(x)", np.sqrt),
            ("log(x)", np.log),
            ("floor(x)", np.floor),
            ("x^2", lambda x: x**2),
        ],
    ),
    (
        "scalar-only math functions",
        [
            ("math.sin(x)", math.sin),
            ("math.exp(-x^2)", lambda x: math.exp(-x * x)),
            ("math.gamma(x)", math.gamma),
        ],
    ),
    (
        "hard cases",
        [
            ("sin(x)/x", lambda x: np.sin(x) / x),
            ("sin(1/x)", lambda x: np.sin(1 / x)),
            ("sqrt(sin(x))", safe_sqrt_of_sin),
            ("abs(x) - 2", lambda x: np.abs(x) - 2),
        ],
    ),
]


def draw_caption(
    screen: pygame.Surface,
    font: pygame.font.Font,
    title: str,
    functions: FunctionSet,
    palette: tuple[object, ...],
) -> None:
    """Draw the set's title and each function's name in its plot colour.

    Args:
        screen: The display surface.
        font: Caption font.
        title: Name of the function set.
        functions: The functions being shown.
        palette: The theme palette (functions use it in order).
    """
    panel = pygame.Surface((240, 26 + 20 * len(functions)), pygame.SRCALPHA)
    panel.fill((255, 255, 255, 225))
    panel.blit(font.render(f"{title}  (Space)", True, (40, 40, 40)), (10, 6))
    for i, (name, _) in enumerate(functions):
        color = pygame.Color(palette[i % len(palette)])  # type: ignore[arg-type]
        panel.blit(font.render(name, True, color), (10, 26 + 20 * i))
    screen.blit(panel, (12, 12))


def main() -> None:
    """Run the example."""
    pygame.init()
    screen = pygame.display.set_mode((1100, 720), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: functions")
    font = pygame.font.Font(bundled_font_path(), 14)
    plane = pgp.Plane(scale=60)
    clock = pygame.time.Clock()
    current = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                current = (current + 1) % len(SETS)
            plane.handle_event(event)

        title, functions = SETS[current]
        plane.draw_grid(screen)
        for _, func in functions:
            plane.plot(screen, func)
        draw_caption(screen, font, title, functions, plane.theme.palette)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
