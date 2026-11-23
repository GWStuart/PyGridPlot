"""Draggable points: free, locked to an axis, and on a curve.

Press ``Space`` to switch between two sets:

1. **A wave you can shape**: drag the free point to move the crest of a sine
   wave, and the two sliders (points locked to a vertical and a horizontal
   line) to change its amplitude and its wavelength.
2. **Points on curves**: a point on a circle, with its angle and its tangent
   line; and a point on a spirograph that follows the curve through its
   crossings.

Hover over a point to see its coordinates. Pan and zoom as usual, even in
the middle of a drag: a press on a point drags the point, anywhere else it
pans the plane.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.render.fonts import bundled_font_path

TRACK = pgp.LineStyle(color=(170, 170, 170), width=4.0, opacity=0.6)
"""How the sliders' tracks are drawn."""

BIG, SMALL, PEN = 2.5, 0.9, 1.6
"""Wheel sizes of the spirograph (a hypotrochoid)."""


def spirograph_x(t: np.ndarray) -> np.ndarray:
    """The spirograph's x coordinate, centred at x = 4."""
    return 4 + (BIG - SMALL) * np.cos(t) + PEN * np.cos((BIG - SMALL) / SMALL * t)


def spirograph_y(t: np.ndarray) -> np.ndarray:
    """The spirograph's y coordinate."""
    return (BIG - SMALL) * np.sin(t) - PEN * np.sin((BIG - SMALL) / SMALL * t)


class Wave:
    """A sine wave shaped by a free point and two sliders."""

    def __init__(self, grid: pgp.Grid) -> None:
        self.crest = grid.draggable((math.pi / 2, 2.0), color=(200, 60, 60))
        self.amplitude = grid.draggable(
            (-7.0, 2.0), lock="x", y_range=(0.2, 4.0), color=(60, 90, 200)
        )
        self.wavelength = grid.draggable(
            (2 * math.pi - 6, -4.5), lock="y", x_range=(-5.0, 2.0), color=(40, 150, 90)
        )
        self.points = (self.crest, self.amplitude, self.wavelength)

    def draw(self, grid: pgp.Grid, screen: pygame.Surface) -> list[str]:
        """Draw the wave and its controls; return the caption lines."""
        a = self.amplitude.y
        k = 2 * math.pi / (self.wavelength.x + 6)  # the slider runs from x = -6
        cx, cy = self.crest.position
        grid.polyline(screen, [(-7, 0.2), (-7, 4)], style=TRACK)
        grid.polyline(screen, [(-5, -4.5), (2, -4.5)], style=TRACK)
        grid.plot(screen, lambda x: a * np.cos(k * (x - cx)) + cy - a)
        return [
            f"y = {a:.2f} cos({k:.2f} (x − {cx:.2f})) + {cy - a:.2f}",
            "drag the red crest, the blue (amplitude) and green (wavelength) sliders",
        ]


class Curves:
    """A point on a circle and a point on a spirograph."""

    def __init__(self, grid: pgp.Grid) -> None:
        self.on_circle = grid.draggable(
            curve=(
                lambda t: -4 + 2 * np.cos(t),
                lambda t: 2 * np.sin(t),
                (0, 2 * math.pi),
            ),
            t=math.pi / 4,
        )
        self.on_spirograph = grid.draggable(
            curve=(spirograph_x, spirograph_y, (0, 18 * math.pi))
        )
        self.points = (self.on_circle, self.on_spirograph)

    def draw(self, grid: pgp.Grid, screen: pygame.Surface) -> list[str]:
        """Draw the curves, the radius and the tangent; return the caption lines."""
        theta = self.on_circle.t or 0.0
        px, py = self.on_circle.position
        grid.parametric(
            screen,
            lambda t: -4 + 2 * np.cos(t),
            lambda t: 2 * np.sin(t),
            (0, 2 * math.pi),
        )
        grid.polyline(screen, [(-4, 0), (px, py)], width=1.5)
        tx, ty = -math.sin(theta), math.cos(theta)  # the tangent direction
        grid.polyline(screen, [(px - 3 * tx, py - 3 * ty), (px + 3 * tx, py + 3 * ty)])
        grid.parametric(
            screen, spirograph_x, spirograph_y, (0, 18 * math.pi), width=1.5
        )
        return [
            f"angle on the circle: {math.degrees(theta):.1f}°",
            f"spirograph parameter: t = {self.on_spirograph.t or 0.0:.2f}",
        ]


def draw_caption(
    screen: pygame.Surface, font: pygame.font.Font, lines: list[str]
) -> None:
    """Draw caption lines in a panel at the top left."""
    texts = [font.render(line, True, (40, 40, 40)) for line in lines]
    width = max(text.get_width() for text in texts) + 20
    height = sum(text.get_height() for text in texts) + 12
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    panel.fill((255, 255, 255, 225))
    y = 6
    for text in texts:
        panel.blit(text, (10, y))
        y += text.get_height()
    screen.blit(panel, (12, 12))


def main() -> None:
    """Run the example."""
    pygame.init()
    screen = pygame.display.set_mode((1100, 720), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: draggable points")
    font = pygame.font.Font(bundled_font_path(), 14)
    grid = pgp.Grid(x_range=(-8, 8))
    sets: list[Wave | Curves] = [Wave(grid), Curves(grid)]
    current = 0
    clock = pygame.time.Clock()

    def show(index: int) -> None:
        """Make only the points of one set draggable."""
        for number, scene in enumerate(sets):
            for point in scene.points:
                point.enabled = number == index

    show(current)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                current = (current + 1) % len(sets)
                show(current)
            grid.handle_event(event)

        scene = sets[current]
        grid.draw(screen)
        lines = scene.draw(grid, screen)
        grid.draw_controls(screen, *scene.points)
        draw_caption(screen, font, [*lines, "Space: next set"])
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
