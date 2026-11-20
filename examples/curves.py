"""Parametric curves ``(x(t), y(t))`` and polar curves ``r(θ)``.

Press ``Space`` to switch between sets of curves:

1. **Parametric**: a Lissajous figure whose phase drifts over time, a
   spirograph (hypotrochoid), a cycloid with its sharp cusps, and
   ``x = tan t``, whose poles break the curve instead of joining it up.
2. **Polar**: a rose whose number of petals changes smoothly, a cardioid,
   a lemniscate (undefined for half of the angles, so it ends exactly at the
   origin) and an Archimedean spiral over many turns.

Pan and zoom as usual: the curves are resampled every frame, so they stay
smooth however far you zoom in. ``P`` pauses the animation.
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


def draw_parametric(plane: pgp.Plane, screen: pygame.Surface, t: float) -> None:
    """Draw the parametric set."""
    phase = 0.4 * t
    plane.parametric(
        screen,
        lambda s: -5 + 2 * np.sin(3 * s + phase),
        lambda s: 2.5 + 2 * np.sin(2 * s),
        (0, 2 * math.pi),
    )
    big, small, pen = 2.5, 0.9, 1.6  # a hypotrochoid ("spirograph")
    plane.parametric(
        screen,
        lambda s: (
            2 + (big - small) * np.cos(s) + pen * np.cos((big - small) / small * s)
        ),
        lambda s: (
            2 + (big - small) * np.sin(s) - pen * np.sin((big - small) / small * s)
        ),
        (0, 18 * math.pi),
        width=1.8,
    )
    plane.parametric(  # a cycloid: a point on a rolling wheel
        screen, lambda s: s - np.sin(s), lambda s: -4 + (1 - np.cos(s)), (-12, 12)
    )
    plane.parametric(screen, np.tan, lambda s: -1.5 + 0.4 * s, (-4, 4), width=2)


def draw_polar(plane: pgp.Plane, screen: pygame.Surface, t: float) -> None:
    """Draw the polar set."""
    petals = 3 + 2 * math.sin(0.3 * t)  # k petals (2k when k is even)
    plane.polar(screen, lambda a: 2.5 * np.cos(petals * a), (0, 4 * math.pi))
    plane.polar(screen, lambda a: 1.5 * (1 + np.cos(a - 0.2 * t)))  # cardioid
    plane.polar(screen, lambda a: 3 * np.sqrt(np.cos(2 * a)), width=2)  # lemniscate
    plane.polar(screen, lambda a: 0.15 * a, (0, 12 * math.pi), width=1.2, opacity=0.7)


SETS: list[tuple[str, Drawing]] = [
    ("parametric curves (x(t), y(t))", draw_parametric),
    ("polar curves r(θ)", draw_polar),
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
    pygame.display.set_caption("PyGridPlot: parametric and polar curves")
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
