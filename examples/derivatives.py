"""Derivatives of any order, driven by draggable points.

Press ``Space`` to switch between three sets:

1. **The tangent line**: drag the point along the curve ``f``; its tangent
   line follows, and the graphs of ``f'`` and ``f''`` show the slope and the
   bending at that point.
2. **Taylor polynomials** (:func:`pygridplot.taylor`): drag the expansion
   point along the curve, and the slider (:meth:`pygridplot.Grid.slider`,
   in the bottom-left corner) to change the degree; the
   polynomial hugs the curve near the point and further away as the degree
   grows.
3. **The osculating circle**: the circle that best fits the curve at the
   point, with radius ``(1 + f'²)^(3/2) / |f''|``.

The derivatives are exact: :func:`pygridplot.derivative` differentiates the
NumPy function itself, to any order.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.render.fonts import bundled_font_path


def f(x: np.ndarray) -> np.ndarray:
    """The function every set works with."""
    return np.sin(x) + 0.3 * np.sin(2.7 * x) + 0.05 * x**2


df = pgp.derivative(f)
d2f = pgp.derivative(f, 2)
FADED = pgp.LineStyle(width=1.8, opacity=0.55)


def on_f(grid: pgp.Grid, t: float) -> pgp.DraggablePoint:
    """Create a draggable point on the graph of ``f``, at ``x = t``."""
    return grid.draggable(curve=(lambda s: s, f, (-12.0, 12.0)), t=t)


class Tangent:
    """The tangent line at a point on the curve, with ``f'`` and ``f''``."""

    def __init__(self, grid: pgp.Grid) -> None:
        self.point = on_f(grid, 1.0)
        self.points = (self.point,)

    def draw(self, grid: pgp.Grid, screen: pygame.Surface) -> list[str]:
        """Draw the set; return the caption lines."""
        a = self.point.x
        value, slope, bend = pgp.derivatives(f, a, 2)
        grid.plot(screen, d2f, style=FADED, color=(150, 110, 200))
        grid.plot(screen, df, style=FADED, color=(60, 150, 90))
        grid.plot(screen, f, color=(40, 90, 200))
        grid.plot(screen, lambda x: value + slope * (x - a), color=(210, 70, 60))
        grid.points(screen, [(a, slope), (a, bend)], radius=4.0, color=(80, 80, 80))
        return [
            f"x = {a:.3f}   f(x) = {value:.3f}   "
            f"f'(x) = {slope:.3f}   f''(x) = {bend:.3f}",
            "blue: f   green: f'   purple: f''   red: the tangent line",
        ]


class Taylor:
    """Taylor polynomials of a chosen degree about a draggable point."""

    def __init__(self, grid: pgp.Grid) -> None:
        self.point = on_f(grid, 0.5)
        self.degree = grid.slider("degree", (0, 20), value=3, step=1)
        self.points = (self.point, self.degree)

    def draw(self, grid: pgp.Grid, screen: pygame.Surface) -> list[str]:
        """Draw the set; return the caption lines."""
        a = self.point.x
        degree = self.degree.value  # an int: the slider's step is 1
        polynomial = pgp.taylor(f, a, degree)  # from the exact derivatives at a
        grid.plot(screen, f, color=(40, 90, 200))
        grid.plot(screen, polynomial, color=(210, 70, 60))
        return [
            f"Taylor polynomial of degree {degree} about x = {a:.3f}",
            "drag the point on the curve, and the slider for the degree",
        ]


class Osculating:
    """The circle of curvature at a point on the curve."""

    def __init__(self, grid: pgp.Grid) -> None:
        self.point = on_f(grid, -1.0)
        self.points = (self.point,)

    def draw(self, grid: pgp.Grid, screen: pygame.Surface) -> list[str]:
        """Draw the set; return the caption lines."""
        a = self.point.x
        value, slope, bend = pgp.derivatives(f, a, 2)
        grid.plot(screen, f, color=(40, 90, 200))
        lines = [f"x = {a:.3f}"]
        if abs(bend) > 1e-9:
            radius = (1 + slope**2) ** 1.5 / abs(bend)
            norm = math.hypot(1.0, slope)
            side = math.copysign(1.0, bend)  # towards the inside of the bend
            cx = a - side * radius * slope / norm
            cy = value + side * radius / norm
            grid.parametric(
                screen,
                lambda t: cx + radius * np.cos(t),
                lambda t: cy + radius * np.sin(t),
                (0, 2 * math.pi),
                color=(210, 70, 60),
            )
            grid.polyline(screen, [(cx, cy), (a, value)], width=1.2)
            lines.append(f"curvature {1 / radius:.3f}, radius {radius:.3f}")
        else:
            lines.append("an inflection point: the curve is straight here")
        return lines


def draw_caption(
    screen: pygame.Surface, font: pygame.font.Font, lines: list[str]
) -> None:
    """Draw caption lines in a panel at the top left."""
    texts = [font.render(line, True, (40, 40, 40)) for line in lines]
    panel = pygame.Surface(
        (
            max(text.get_width() for text in texts) + 20,
            sum(text.get_height() for text in texts) + 12,
        ),
        pygame.SRCALPHA,
    )
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
    pygame.display.set_caption("PyGridPlot: derivatives")
    font = pygame.font.Font(bundled_font_path(), 14)
    grid = pgp.Grid(x_range=(-8, 8))
    sets: list[Tangent | Taylor | Osculating] = [
        Tangent(grid),
        Taylor(grid),
        Osculating(grid),
    ]
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
