"""Integrals and antiderivatives, driven by draggable points.

Press ``Space`` to switch between three sets:

1. **The area under a curve**: drag the two points along the curve. The
   area between them is shaded and its value shown, counting the parts
   below the x axis as negative. The faded curve is the antiderivative
   ``F`` from the first point: its height at the second point is the area
   (the fundamental theorem of calculus).
2. **Riemann sums**: drag the slider to change the number of rectangles,
   and press ``R`` to switch between left, right and midpoint sums and the
   trapezoidal rule. The sum closes in on the integral as the rectangles
   narrow, and the midpoint and trapezoidal rules close in much faster.
3. **Improper integrals**: drag the point. The area under ``1/x²`` from the
   point to infinity is finite (``1/a``); under ``1/x`` it is not, and
   :func:`pygridplot.integral` says so with NaN.

Everything is computed with :func:`pygridplot.integral` and
:func:`pygridplot.antiderivative`, to about 10 significant digits.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.render.fonts import bundled_font_path

BLUE = (40, 100, 200)
RED = (200, 50, 60)
GREEN = (40, 150, 90)
FADED = pgp.LineStyle(width=1.8, opacity=0.55)


def f(x: np.ndarray) -> np.ndarray:
    """The function of the first two sets."""
    return np.sin(x) + 0.3 * np.sin(2.7 * x) + 0.05 * x**2


def on_f(plane: pgp.Plane, t: float) -> pgp.DraggablePoint:
    """Create a draggable point on the graph of ``f``, at ``x = t``."""
    return plane.draggable(curve=(lambda s: s, f, (-12.0, 12.0)), t=t)


class Area:
    """The area under ``f`` between two draggable points, and its antiderivative."""

    def __init__(self, plane: pgp.Plane) -> None:
        self.a = on_f(plane, -1.0)
        self.b = on_f(plane, 2.5)
        self.points = (self.a, self.b)

    def draw(self, plane: pgp.Plane, screen: pygame.Surface) -> list[str]:
        """Draw the set; return the caption lines."""
        a, b = self.a.x, self.b.x
        area = pgp.integral(f, a, b)
        from_a = pgp.antiderivative(f, a)  # F(x) = ∫ f from a to x
        if a != b:
            plane.fill_between(screen, f, domain=(min(a, b), max(a, b)), color=BLUE)
        plane.plot(screen, from_a, style=FADED, color=GREEN)
        plane.plot(screen, f, color=BLUE)
        plane.polyline(screen, [(b, 0), (b, area)], width=1.2, color=GREEN)
        plane.points(screen, [(b, area)], radius=4.0, color=GREEN)
        plane.draw_draggables(screen, *self.points)
        return [
            f"∫ f(x) dx from {a:.3f} to {b:.3f} = {area:.6f}",
            "blue: f and the area   green: F(x), the integral from the first point",
        ]


class Riemann:
    """Riemann sums and the trapezoidal rule, with a slider for the rectangles."""

    RULES = ("left", "right", "midpoint", "trapezoidal")
    LOW, HIGH = -3.0, 4.0

    def __init__(self, plane: pgp.Plane) -> None:
        # Whole numbers on a geometric track: as much room for 1 to 8 as for
        # 8 to 64.
        self.slider = plane.slider("n", (1, 64), value=8, step=1, log=True)
        self.points = (self.slider,)
        self.rule = 0

    def next_rule(self) -> None:
        """Switch to the next rule."""
        self.rule = (self.rule + 1) % len(self.RULES)

    def draw(self, plane: pgp.Plane, screen: pygame.Surface) -> list[str]:
        """Draw the set; return the caption lines."""
        n = self.slider.value
        rule = self.RULES[self.rule]
        x = np.linspace(self.LOW, self.HIGH, n + 1)
        width = (self.HIGH - self.LOW) / n
        if rule == "trapezoidal":
            ys = f(x)
            total = float(np.sum((ys[:-1] + ys[1:]) / 2) * width)
            plane.fill_between(
                screen,
                lambda t: np.interp(t, x, ys),
                domain=(self.LOW, self.HIGH),
                color=RED,
            )
            gap = np.full(n, np.nan)
            zero = np.zeros(n)
            shapes = np.column_stack(
                [x[:-1], zero, x[:-1], ys[:-1], x[1:], ys[1:], x[1:], zero, gap, gap]
            )
        else:
            sample = {"left": x[:-1], "right": x[1:], "midpoint": (x[:-1] + x[1:]) / 2}
            h = f(sample[rule])
            total = float(h.sum() * width)

            def step(t: np.ndarray) -> np.ndarray:
                return h[np.clip(np.searchsorted(x, t, side="right") - 1, 0, n - 1)]

            plane.fill_between(screen, step, domain=(self.LOW, self.HIGH), color=RED)
            gap = np.full(n, np.nan)
            zero = np.zeros(n)
            shapes = np.column_stack(
                [x[:-1], zero, x[:-1], h, x[1:], h, x[1:], zero, gap, gap]
            )
        plane.polyline(screen, shapes.reshape(-1, 2), width=1.2, color=RED)
        plane.plot(screen, f, color=BLUE)
        plane.draw_draggables(screen, *self.points)
        exact = pgp.integral(f, self.LOW, self.HIGH)
        return [
            f"{rule} rule with n = {n}: {total:.6f}   integral: {exact:.6f}   "
            f"error: {total - exact:+.2e}",
            "drag the slider for the number of rectangles; R: next rule",
        ]


class Improper:
    """Integrals to infinity: one converges, one diverges."""

    def __init__(self, plane: pgp.Plane) -> None:
        self.a = plane.draggable((1.0, 0.0), lock="y", x_range=(0.25, 8.0))
        self.points = (self.a,)

    def draw(self, plane: pgp.Plane, screen: pygame.Surface) -> list[str]:
        """Draw the set; return the caption lines."""
        a = self.a.x
        inverse_square = lambda x: 1 / x**2  # noqa: E731
        inverse = lambda x: 1 / x  # noqa: E731
        finite = pgp.integral(inverse_square, a, math.inf)
        infinite = pgp.integral(inverse, a, math.inf)
        plane.fill_between(
            screen, inverse, domain=(a, math.inf), color=RED, opacity=0.15
        )
        plane.fill_between(screen, inverse_square, domain=(a, math.inf), color=BLUE)
        plane.plot(screen, inverse, color=RED, domain=(0.05, math.inf))
        plane.plot(screen, inverse_square, color=BLUE, domain=(0.05, math.inf))
        plane.draw_draggables(screen, *self.points)
        diverges = "diverges" if math.isnan(infinite) else f"{infinite:.6f}"
        return [
            f"blue: ∫ 1/x² dx from {a:.3f} to ∞ = {finite:.6f}   (1/a = {1 / a:.6f})",
            f"red: ∫ 1/x dx from {a:.3f} to ∞: {diverges}",
        ]


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
    pygame.display.set_caption("PyGridPlot: integrals")
    font = pygame.font.Font(bundled_font_path(), 14)
    plane = pgp.Plane(x_range=(-8, 8))
    riemann = Riemann(plane)
    sets: list[Area | Riemann | Improper] = [Area(plane), riemann, Improper(plane)]
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
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                riemann.next_rule()
            plane.handle_event(event)

        scene = sets[current]
        plane.draw_grid(screen)
        lines = scene.draw(plane, screen)
        draw_caption(screen, font, [*lines, "Space: next set"])
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
