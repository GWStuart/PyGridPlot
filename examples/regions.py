"""Inequalities and areas between graphs: shading regions of the plane.

Press ``Space`` to switch between sets:

1. **Inequalities**: a disc and an annulus, the region above a drifting
   sine wave, and a feasible region (several conditions combined with
   ``&``, as in linear programming) with its corners marked. A condition
   is plain Python: ``lambda x, y: x**2 + y**2 < 4``.
2. **The area between two graphs**, between two draggable bounds on the x
   axis, with its value: drag the bounds, and the shading and the number
   follow. Where the graphs cross, both parts are shaded (and the value
   counts the part where ``g`` is above ``f`` as negative).
3. **Hard cases**: regions that end at the asymptotes of ``tan``, steps of
   ``floor``, a region that starts where ``sqrt`` becomes defined, and
   areas that stop cleanly at poles and run off to the bottom of the plane.

Regions and areas have no outline of their own: the boundaries are drawn
with ``implicit`` and ``plot``. Pan and zoom as usual; ``P`` pauses the
animation.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.render.fonts import bundled_font_path

Drawing = Callable[[pgp.Plane, pygame.Surface, float], str]
"""Draws one set at time ``t`` (in seconds) and returns its caption."""

BLUE = (40, 100, 200)
RED = (200, 50, 60)
GREEN = (40, 150, 90)
ORANGE = (230, 140, 30)


def draw_inequalities(plane: pgp.Plane, screen: pygame.Surface, t: float) -> str:
    """Draw a few inequalities, alone and combined."""
    # A disc, with its boundary circle drawn on top.
    plane.inequality(screen, lambda x, y: (x + 5) ** 2 + (y - 2) ** 2 < 4, color=BLUE)
    plane.implicit(screen, lambda x, y: (x + 5) ** 2 + (y - 2) ** 2, 4, color=BLUE)
    # An annulus: two conditions at once. Brackets matter, as & binds
    # more tightly than <.
    plane.inequality(
        screen,
        lambda x, y: (
            ((x + 5) ** 2 + (y + 3) ** 2 > 1) & ((x + 5) ** 2 + (y + 3) ** 2 < 4)
        ),
        color=GREEN,
    )
    # Above a drifting wave (and below y = 5).
    phase = 0.6 * t
    plane.inequality(
        screen, lambda x, y: (y > np.sin(x - phase) + 3) & (y < 5), color=ORANGE
    )
    plane.plot(screen, lambda x: np.sin(x - phase) + 3, color=ORANGE, width=2)
    # A feasible region: x >= 0, y >= 0, x + 2y <= 8, 3x + y <= 9.
    plane.inequality(
        screen,
        lambda x, y: (x >= 0) & (y >= 0) & (x + 2 * y <= 8) & (3 * x + y <= 9),
        color=RED,
        opacity=0.3,
    )
    plane.points(screen, [(0, 0), (3, 0), (2, 3), (0, 4)], color=RED)
    return "inequalities: x² + y² < 4, annulus, y > sin x, a feasible region"


class Area:
    """The area between two graphs, between two draggable bounds."""

    def __init__(self, plane: pgp.Plane) -> None:
        self.a = plane.draggable((-1.5, 0.0), lock="y", color=BLUE)
        self.b = plane.draggable((2.5, 0.0), lock="y", color=BLUE)
        self.points = (self.a, self.b)

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
        """The upper graph (mostly)."""
        return np.sin(x) + 1.5

    @staticmethod
    def g(x: np.ndarray) -> np.ndarray:
        """The lower graph (mostly)."""
        return 0.25 * x**2

    def draw(self, plane: pgp.Plane, screen: pygame.Surface, t: float) -> str:
        """Draw the area and the graphs; return the caption."""
        low, high = sorted((self.a.x, self.b.x))
        if high > low:
            plane.fill_between(screen, self.f, self.g, domain=(low, high), color=BLUE)
        plane.plot(screen, self.f, color=BLUE)
        plane.plot(screen, self.g, color=GREEN)
        plane.draw_draggables(screen, *self.points)
        value = pgp.integral(lambda x: self.f(x) - self.g(x), low, high)
        bounds = f"from {low:.2f} to {high:.2f}"
        return f"area between sin x + 1.5 and x²/4 {bounds}: {value:.4f}"


def draw_hard_cases(plane: pgp.Plane, screen: pygame.Surface, t: float) -> str:
    """Draw regions and areas with poles, jumps and domain edges."""
    # Below tan x, above y = 1: each region ends at an asymptote.
    plane.inequality(screen, lambda x, y: (y < np.tan(x)) & (y > 1), color=RED)
    plane.plot(screen, np.tan, color=RED, width=2)
    # Below the steps of floor x (and above y = -5).
    plane.inequality(
        screen, lambda x, y: (y < np.floor(x) - 1) & (y > -5) & (x > 2), color=GREEN
    )
    # sqrt is undefined for x < -6: the region starts there.
    plane.inequality(
        screen, lambda x, y: (y < np.sqrt(x + 6) - 1) & (y > -4) & (x < -1), color=BLUE
    )
    # The area under 1/x, down to the bottom of the plane, stopping at the
    # pole without joining its branches.
    plane.fill_between(screen, lambda x: 1 / x, -math.inf, domain=(-1, 1), color=ORANGE)
    plane.plot(screen, lambda x: 1 / x, color=ORANGE, width=2, domain=(-1, 1))
    return "hard cases: asymptotes, jumps, domain edges, poles"


def draw_caption(screen: pygame.Surface, font: pygame.font.Font, title: str) -> None:
    """Draw the set's caption and the keys."""
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
    pygame.display.set_caption("PyGridPlot: inequalities and areas")
    font = pygame.font.Font(bundled_font_path(), 14)
    plane = pgp.Plane(x_range=(-9, 9))
    area = Area(plane)
    sets: list[Drawing] = [draw_inequalities, area.draw, draw_hard_cases]
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
                current = (current + 1) % len(sets)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                paused = not paused
            plane.handle_event(event)
        # Only the area's bounds can be dragged, and only while they show.
        for point in area.points:
            point.enabled = sets[current] == area.draw

        now = time.perf_counter()
        if not paused:
            elapsed += now - last
        last = now
        plane.draw_grid(screen)
        caption = sets[current](plane, screen, elapsed)
        draw_caption(screen, font, caption)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
