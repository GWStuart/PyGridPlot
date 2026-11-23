"""Plotting explicit functions, including the awkward ones.

Press ``Space`` to switch between sets of functions:

1. The shared test set from the project plan: ``1/x``, ``tan(x)``,
   ``sqrt(x)``, ``log(x)``, ``floor(x)`` and ``x**2``.
2. Functions written with :mod:`math`, which only work on single numbers.
   PyGridPlot falls back to calling them point by point.
3. Hard cases: a removable singularity, a fast oscillation and a function
   that raises an error for half of its inputs.

Pan and zoom as usual. Asymptotes and jumps are never joined by vertical
lines, and gaps appear wherever a function is undefined. Each function is
plotted with a ``label=``, and the legend (top right) names them in their
colours.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.render.fonts import bundled_font_path

FunctionSet = list[tuple[str, Callable[[Any], Any]]]


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


def draw_caption(screen: pygame.Surface, font: pygame.font.Font, title: str) -> None:
    """Draw the set's title and the key that switches sets.

    Args:
        screen: The display surface.
        font: Caption font.
        title: Name of the function set.
    """
    text = font.render(f"{title}  (Space: next set)", True, (40, 40, 40))
    panel = pygame.Surface(
        (text.get_width() + 20, text.get_height() + 12), pygame.SRCALPHA
    )
    panel.fill((255, 255, 255, 225))
    panel.blit(text, (10, 6))
    screen.blit(panel, (12, 12))


def main() -> None:
    """Run the example."""
    pygame.init()
    screen = pygame.display.set_mode((1100, 720), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: functions")
    font = pygame.font.Font(bundled_font_path(), 14)
    grid = pgp.Grid(scale=60)
    clock = pygame.time.Clock()
    current = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                current = (current + 1) % len(SETS)
            grid.handle_event(event)

        title, functions = SETS[current]
        grid.draw(screen)
        for name, func in functions:
            grid.plot(screen, func, label=name)
        grid.legend(screen)
        draw_caption(screen, font, title)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
