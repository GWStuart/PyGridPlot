"""Implicit curves ``F(x, y) = c``: any shape, found without a parameter.

Press ``Space`` to switch between sets of curves:

1. **Classic curves**: a circle, a heart, a lemniscate and the folium of
   Descartes (both cross themselves), and an elliptic curve
   ``y² = x³ - 3x + c`` whose oval drifts into its main branch and
   splits off again as ``c`` changes: the curve's shape changes, which an
   implicit plot handles without any special effort.
2. **Level sets**: the contours ``F = c`` of one function for several
   values of ``c``, like the height lines of a map, drawn with one call,
   with the landscape drifting slowly.
3. **Poles, jumps and edges**: ``y = tan x`` and ``y = floor x`` written
   implicitly, with no lines drawn down the poles or up the jumps;
   ``x = sin y`` (not a function of ``x``); and a curve that stops where
   ``sqrt`` becomes undefined.

Pan and zoom as usual: curves are found again every frame, so they stay
sharp however far you zoom in. ``P`` pauses the animation.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.render.fonts import bundled_font_path

Drawing = Callable[[pgp.Plane, pygame.Surface, float], None]
"""Draws one set of curves at time ``t`` (in seconds)."""


def draw_classics(plane: pgp.Plane, screen: pygame.Surface, t: float) -> None:
    """Draw the classic curves."""
    plane.implicit(screen, lambda x, y: x**2 + y**2, 4)  # the circle x² + y² = 4
    plane.implicit(  # a heart
        screen,
        lambda x, y: (
            ((x - 5) ** 2 + (y - 2) ** 2 - 2) ** 3 - 3 * (x - 5) ** 2 * (y - 2) ** 3
        ),
    )
    plane.implicit(  # a lemniscate of Bernoulli
        screen, lambda x, y: ((x + 5) ** 2 + y**2) ** 2 - 8 * ((x + 5) ** 2 - y**2)
    )
    plane.implicit(  # the folium of Descartes
        screen, lambda x, y: x**3 + (y + 3) ** 3 - 4 * x * (y + 3), width=2
    )
    c = 2.6 * math.sin(0.5 * t)  # the oval merges with the branch at c = -2
    plane.implicit(screen, lambda x, y: y**2 - x**3 + 3 * x - c, width=2)


def draw_level_sets(plane: pgp.Plane, screen: pygame.Surface, t: float) -> None:
    """Draw contours of a drifting landscape."""
    shift = 0.3 * t

    def height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sin(x + shift) + np.cos(0.8 * y) + 0.3 * np.sin(0.5 * x * y)

    # One call for all the levels: the function is evaluated only once.
    plane.implicit(screen, height, np.linspace(-1.6, 1.6, 9), width=1.5)
    plane.implicit(screen, height, 0, width=3)  # the sea level, emphasised


def draw_undefined(plane: pgp.Plane, screen: pygame.Surface, t: float) -> None:
    """Draw curves with poles, jumps and domain edges."""
    plane.implicit(screen, lambda x, y: y - np.tan(x))
    plane.implicit(screen, lambda x, y: y + 3 - np.floor(x))
    plane.implicit(screen, lambda x, y: x - 2 * np.sin(y + 0.5 * t) - 5)
    plane.implicit(screen, lambda x, y: y - 2 - np.sqrt(x + 6))
    plane.implicit(screen, lambda x, y: x * y, 1, width=2, opacity=0.6)  # y = 1/x


SETS: list[tuple[str, Drawing]] = [
    ("classic curves F(x, y) = 0", draw_classics),
    ("level sets F(x, y) = c", draw_level_sets),
    ("poles, jumps and domain edges", draw_undefined),
]


def draw_caption(screen: pygame.Surface, font: pygame.font.Font, title: str) -> None:
    """Draw the set's title and the keys."""
    text = font.render(f"{title}   (Space: next   P: pause)", True, (40, 40, 40))
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
    pygame.display.set_caption("PyGridPlot: implicit curves")
    font = pygame.font.Font(bundled_font_path(), 14)
    plane = pgp.Plane(x_range=(-9, 9))
    clock = pygame.time.Clock()
    current = 0
    paused = False
    elapsed = 0.0
    last = time.perf_counter()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                current = (current + 1) % len(SETS)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                paused = not paused
            plane.handle_event(event)

        now = time.perf_counter()
        if not paused:
            elapsed += now - last
        last = now
        title, draw = SETS[current]
        plane.draw_grid(screen)
        draw(plane, screen, elapsed)
        draw_caption(screen, font, title)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
