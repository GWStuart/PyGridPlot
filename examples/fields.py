"""Vector fields, slope fields and arrows.

Press ``Space`` to switch between sets:

1. **Vector fields**: a rotation, a saddle, a pair of opposite charges and
   a shear flow (``F`` cycles through them). A draggable probe shows the
   vector at its point as a bold arrow. ``M`` switches between arrows of
   one length (the field's direction) and arrows as long as the field is
   strong (its magnitude too).
2. **Slope fields** of ``dy/dx = x - y`` and of the logistic equation
   ``dy/dx = y (1 - y)`` (``F`` switches), with the exact solution through
   a draggable point: drag it, and the solution follows the segments.
3. **Arrows**: vector addition, with two draggable vectors ``a`` and ``b``
   and their sum drawn tip to tail; and a point running round a curve with
   its velocity and acceleration, from exact derivatives.

Zoom in and out to see the field's arrows grow and fade in between the
others as the spacing allows, and shrink and fade out again: their density
on screen stays the same, and nothing jumps. ``P`` pauses the animation.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.render.fonts import bundled_font_path

BLUE = (40, 100, 200)
RED = (200, 50, 60)
GREEN = (40, 150, 90)
GRAY = (120, 120, 130)


def dipole(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The electric field of charges +1 at (-2, 0) and -1 at (2, 0)."""
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    for cx, q in ((-2.0, 1.0), (2.0, -1.0)):
        dx, dy = x - cx, y
        r3 = np.hypot(dx, dy) ** 3
        u = u + q * dx / r3
        v = v + q * dy / r3
    return u, v


@dataclass(frozen=True)
class Field:
    """A vector field to show, and the scale that suits its magnitudes."""

    name: str
    func: Callable[[np.ndarray, np.ndarray], object]
    scale: float


FIELDS = (
    Field("rotation (-y, x)", lambda x, y: (-y, x), 6.0),
    Field("saddle (x, -y)", lambda x, y: (x, -y), 6.0),
    Field("two opposite charges", dipole, 200.0),
    Field("shear flow (sin y, 0)", lambda x, y: (np.sin(y), 0 * x), 25.0),
)


class Vectors:
    """Vector fields, with a probe showing the vector at a point."""

    def __init__(self, grid: pgp.Grid) -> None:
        self.probe = grid.draggable((3.0, 2.0), color=RED)
        self.points = (self.probe,)
        self.current = 0
        self.magnitude = False

    def key(self, key: int) -> None:
        """Cycle the fields (F) and the length of the arrows (M)."""
        if key == pygame.K_f:
            self.current = (self.current + 1) % len(FIELDS)
        elif key == pygame.K_m:
            self.magnitude = not self.magnitude

    def draw(self, grid: pgp.Grid, screen: pygame.Surface, t: float) -> str:
        """Draw the field and the probe; return the caption."""
        field = FIELDS[self.current]
        scale = field.scale if self.magnitude else None
        grid.vector_field(screen, field.func, scale=scale, color=BLUE)
        # The vector at the probe, drawn as a bold arrow from it.
        x, y = self.probe.x, self.probe.y
        u, v = (float(c) for c in np.ravel(field.func(np.array([x]), np.array([y]))))
        size = math.hypot(u, v)
        if size > 0 and math.isfinite(size):
            reach = 1.5 / size  # always 1.5 units long
            grid.arrow(screen, (x, y), (x + u * reach, y + v * reach), color=RED)
        grid.draw_controls(screen, *self.points)
        lengths = "magnitude" if self.magnitude else "direction only"
        return f"{field.name}, {lengths}   (F: next field, M: lengths)"


@dataclass(frozen=True)
class Equation:
    """A differential equation and its solution through a point."""

    name: str
    slope: Callable[[np.ndarray, np.ndarray], np.ndarray]
    solution: Callable[[float, float], Callable[[np.ndarray], np.ndarray]]


def linear_solution(x0: float, y0: float) -> Callable[[np.ndarray], np.ndarray]:
    """The solution of dy/dx = x - y through (x0, y0): x - 1 + c e^(-x)."""
    c = (y0 - x0 + 1) * math.exp(x0)
    return lambda x: x - 1 + c * np.exp(-x)


def logistic_solution(x0: float, y0: float) -> Callable[[np.ndarray], np.ndarray]:
    """The solution of dy/dx = y (1 - y) through (x0, y0)."""
    if y0 == 0:
        return lambda x: 0 * x
    c = (1 / y0 - 1) * math.exp(x0)
    return lambda x: 1 / (1 + c * np.exp(-x))


EQUATIONS = (
    Equation("dy/dx = x - y", lambda x, y: x - y, linear_solution),
    Equation("dy/dx = y (1 - y)", lambda x, y: y * (1 - y), logistic_solution),
)


class Slopes:
    """Slope fields, with the solution through a draggable point."""

    def __init__(self, grid: pgp.Grid) -> None:
        self.start = grid.draggable((-2.0, 1.5), color=RED)
        self.points = (self.start,)
        self.current = 0

    def key(self, key: int) -> None:
        """Switch the equation (F)."""
        if key == pygame.K_f:
            self.current = (self.current + 1) % len(EQUATIONS)

    def draw(self, grid: pgp.Grid, screen: pygame.Surface, t: float) -> str:
        """Draw the slope field and the solution; return the caption."""
        equation = EQUATIONS[self.current]
        grid.slope_field(screen, equation.slope, color=GRAY)
        solution = equation.solution(self.start.x, self.start.y)
        grid.plot(screen, solution, color=RED)
        grid.draw_controls(screen, *self.points)
        return f"slope field of {equation.name}, solution through the point   (F: next)"


class Arrows:
    """Vector addition, and velocity and acceleration on a curve."""

    def __init__(self, grid: pgp.Grid) -> None:
        self.a = grid.draggable((3.0, 1.0), color=BLUE)
        self.b = grid.draggable((1.0, 3.0), color=GREEN)
        self.points = (self.a, self.b)

    def key(self, key: int) -> None:
        """No keys of its own."""

    @staticmethod
    def x(t: np.ndarray) -> np.ndarray:
        """The curve's x coordinate."""
        return -4.5 + 2.5 * np.cos(t)

    @staticmethod
    def y(t: np.ndarray) -> np.ndarray:
        """The curve's y coordinate."""
        return -2.0 + 1.5 * np.sin(2 * t)

    def draw(self, grid: pgp.Grid, screen: pygame.Surface, t: float) -> str:
        """Draw both demonstrations; return the caption."""
        # a and b from the origin, the same vectors along the far sides of
        # the parallelogram they span, and their sum across it. The handles
        # are drawn first, so that the arrowheads stay visible on top.
        grid.draw_controls(screen, *self.points)
        a, b = (self.a.x, self.a.y), (self.b.x, self.b.y)
        total = (a[0] + b[0], a[1] + b[1])
        grid.polygon(screen, [(0, 0), a, total, b], color=GRAY, width=0)
        grid.arrow(screen, b, total, color=BLUE, opacity=0.4)
        grid.arrow(screen, a, total, color=GREEN, opacity=0.4)
        grid.arrow(screen, (0, 0), a, color=BLUE)
        grid.arrow(screen, (0, 0), b, color=GREEN)
        grid.arrow(screen, (0, 0), total, color=RED, width=3.5)
        # A point running round a curve, with its velocity and acceleration.
        s = 0.8 * t
        px, vx, ax = pgp.derivatives(self.x, s, 2)
        py, vy, ay = pgp.derivatives(self.y, s, 2)
        loop = (0, 2 * math.pi)
        grid.parametric(screen, self.x, self.y, loop, color=GRAY, width=1.5)
        grid.arrow(screen, (px, py), (px + vx / 2, py + vy / 2), color=BLUE)
        grid.arrow(screen, (px, py), (px + ax / 4, py + ay / 4), color=GREEN)
        grid.points(screen, (px, py), color=RED)
        return "a + b (drag a and b), and velocity and acceleration on a curve"


class Demo(Protocol):
    """One set of the example: its draggable points, keys and drawing."""

    @property
    def points(self) -> tuple[pgp.DraggablePoint, ...]:
        """The set's draggable points."""
        ...

    def key(self, key: int) -> None:
        """Handle a key press of the set's own."""

    def draw(self, grid: pgp.Grid, screen: pygame.Surface, t: float) -> str:
        """Draw the set at time ``t``; return its caption."""
        ...


def draw_caption(screen: pygame.Surface, font: pygame.font.Font, title: str) -> None:
    """Draw the set's caption and the keys."""
    text = font.render(f"{title}   (Space: next set   P: pause)", True, (40, 40, 40))
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
    pygame.display.set_caption("PyGridPlot: vector fields, slope fields and arrows")
    font = pygame.font.Font(bundled_font_path(), 14)
    grid = pgp.Grid(x_range=(-9, 9))
    sets: list[Demo] = [Vectors(grid), Slopes(grid), Arrows(grid)]
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
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    current = (current + 1) % len(sets)
                elif event.key == pygame.K_p:
                    paused = not paused
                else:
                    sets[current].key(event.key)
            grid.handle_event(event)
        # Only the points of the set on show can be dragged.
        for number, demo in enumerate(sets):
            for point in demo.points:
                point.enabled = number == current

        now = time.perf_counter()
        if not paused:
            elapsed += now - last
        last = now
        grid.draw(screen)
        caption = sets[current].draw(grid, screen, elapsed)
        draw_caption(screen, font, caption)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
