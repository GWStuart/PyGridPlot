"""Axis titles, arrowheads, legends and text: annotating a plot.

Press ``Space`` to switch between sets:

1. **A thrown ball**: its height ``h(t)`` and velocity ``v = dh/dt``, with
   axis titles, arrowheads on the axes, and a legend of the labelled plots
   (top right). Text marks the highest point and the landing, and a
   draggable point on the height curve carries its own label, which moves
   with it.
2. **A labelled triangle**: drag the vertices ``A``, ``B`` and ``C``. Their
   names sit beside them, on the side away from the triangle, and each
   side's length is written at its middle; both are anchored to world
   points, so they follow the triangle as you pan and zoom.
3. **Filters on log axes**: the gain of low-pass filters of orders 1 to 4,
   on log-log axes with titles. Each order falls by another 20 dB per
   decade beyond the corner frequency, as the text along the slopes says.

Pan and zoom as usual: the titles stay on screen, pinned to the edges when
their axes leave it, and the legend stays in its corner.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.core.plot.text import Anchor
from pygridplot.render.fonts import bundled_font_path
from pygridplot.render.pygame_renderer import PygameRenderer

BLUE = (40, 100, 200)
RED = (200, 50, 60)
GREEN = (40, 150, 90)
GRAY = (110, 110, 120)
ARROWS = replace(pgp.LIGHT, axis_arrows=True)

COMPASS: tuple[Anchor, ...] = (
    "left",
    "bottom-left",
    "bottom",
    "bottom-right",
    "right",
    "top-right",
    "top",
    "top-left",
)
"""The anchor that puts text beside a point, by direction (east, then anticlockwise)."""


def away(point: tuple[float, float], center: tuple[float, float]) -> Anchor:
    """The anchor that puts text beside ``point``, on the side away from ``center``."""
    angle = math.atan2(point[1] - center[1], point[0] - center[0])
    return COMPASS[round(angle / (math.pi / 4)) % 8]


class Demo(Protocol):
    """One set of the example."""

    plane: pgp.Plane

    def draw(self, screen: pygame.Surface) -> str:
        """Draw the set; return its caption."""
        ...


class Ball:
    """The height and velocity of a ball thrown straight up."""

    def __init__(self, renderer: PygameRenderer) -> None:
        self.plane = pgp.Plane(
            x_range=(-0.6, 5.0), y_range=(-26, 30), theme=ARROWS, renderer=renderer
        )
        self.top = 20 / 9.8  # when the velocity is zero
        self.land = 2 * self.top
        self.marker = self.plane.draggable(
            curve=(lambda t: t, self.height, (0, self.land)), t=1.0, color=BLUE
        )

    @staticmethod
    def height(t: np.ndarray) -> np.ndarray:
        """Height in metres, t seconds after the throw."""
        return 20 * t - 4.9 * t**2

    def draw(self, screen: pygame.Surface) -> str:
        """Draw the set; return its caption."""
        plane = self.plane
        plane.draw_grid(screen, x_title="t (s)", y_title="h (m), v (m/s)")
        domain = (0.0, self.land)
        velocity = pgp.derivative(self.height)
        plane.fill_between(
            screen,
            velocity,
            0,
            domain=domain,
            color=RED,
            opacity=0.15,
            label="rising, then falling",
        )
        plane.plot(
            screen, velocity, color=RED, domain=domain, label="velocity v = dh/dt"
        )
        plane.plot(screen, self.height, color=BLUE, domain=domain, label="height h(t)")
        top = (self.top, float(self.height(np.array(self.top))))
        plane.points(
            screen, [top, (self.land, 0.0)], color=GRAY, radius=4, label="events"
        )
        plane.text(
            screen, f"highest point, {top[1]:.1f} m", top, anchor="bottom", gap=10
        )
        plane.text(screen, "lands", (self.land, 0.0), anchor="bottom-left", gap=6)
        plane.legend(screen)
        plane.draw_draggables(screen, self.marker)
        x, y = self.marker.position
        plane.text(
            screen,
            f"t = {x:.2f} s\nh = {y:.1f} m",
            (x, y),
            anchor="top-left",
            gap=10,
            color=BLUE,
        )
        return "A thrown ball: drag the blue point along the height curve"


class Triangle:
    """A triangle with draggable, labelled vertices and its side lengths."""

    def __init__(self, renderer: PygameRenderer) -> None:
        self.plane = pgp.Plane(x_range=(-8, 8), theme=ARROWS, renderer=renderer)
        self.vertices = [
            self.plane.draggable(position, color=GRAY, coordinates="never")
            for position in ((-3.0, -2.0), (4.0, -1.0), (0.5, 3.5))
        ]

    def draw(self, screen: pygame.Surface) -> str:
        """Draw the set; return its caption."""
        plane = self.plane
        plane.draw_grid(screen, x_title="x", y_title="y")
        corners = [point.position for point in self.vertices]
        center = (
            sum(x for x, _ in corners) / 3,
            sum(y for _, y in corners) / 3,
        )
        plane.polygon(screen, corners, color=BLUE, label="triangle ABC")
        circle = circumcircle(corners)
        if circle is not None:
            (cx, cy), r = circle
            plane.parametric(
                screen,
                lambda t: cx + r * np.cos(t),
                lambda t: cy + r * np.sin(t),
                (0, 2 * math.pi),
                color=GREEN,
                width=1.5,
                label="circumcircle",
            )
        plane.points(screen, center, color=RED, radius=3.5, label="centroid")
        for name, corner in zip("ABC", corners, strict=True):
            plane.text(
                screen, name, corner, anchor=away(corner, center), gap=9, size=16
            )
        for i, start in enumerate(corners):
            end = corners[(i + 1) % 3]
            middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            length = math.dist(start, end)
            plane.text(
                screen,
                f"{length:.2f}",
                middle,
                anchor=away(middle, center),
                gap=4,
                color=BLUE,
            )
        plane.legend(screen)
        plane.draw_draggables(screen, *self.vertices)
        return "A labelled triangle: drag the vertices A, B and C"


def circumcircle(
    corners: list[tuple[float, float]],
) -> tuple[tuple[float, float], float] | None:
    """The circle through three points, or ``None`` if they are in a line."""
    (ax, ay), (bx, by), (cx, cy) = corners
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (ux, uy), math.dist((ux, uy), (ax, ay))


class Filters:
    """The gains of Butterworth low-pass filters on log-log axes."""

    def __init__(self, renderer: PygameRenderer) -> None:
        self.plane = pgp.Plane(
            x_range=(0.01, 300),
            y_range=(1e-5, 4),
            log_x=True,
            log_y=True,
            theme=ARROWS,
            renderer=renderer,
        )

    def draw(self, screen: pygame.Surface) -> str:
        """Draw the set; return its caption."""
        plane = self.plane
        plane.draw_grid(screen, x_title="frequency ω (rad/s)", y_title="gain |H(iω)|")
        colors = (BLUE, RED, GREEN, GRAY)
        plane.polyline(screen, [(1, 1e-6), (1, 10)], color=GRAY, width=1, opacity=0.5)
        for order, color in enumerate(colors, start=1):
            plane.plot(screen, butterworth(order), color=color, label=f"order {order}")
            # Beyond the corner, the gain is about ω^-n: 20n dB per decade.
            at = (10.0, 10.0**-order)
            plane.text(
                screen,
                f"−{20 * order} dB/decade",
                at,
                anchor="bottom-left",
                gap=6,
                color=color,
            )
        plane.points(screen, (1, 1 / math.sqrt(2)), color=GRAY, radius=3.5)
        plane.text(
            screen,
            "−3 dB at the corner, ω = 1",
            (1, 1 / math.sqrt(2)),
            anchor="top-right",
            gap=6,
        )
        plane.legend(screen)
        return "Low-pass filters on log axes: every order falls 20 dB per decade faster"


def butterworth(order: int) -> Callable[[np.ndarray], np.ndarray]:
    """The gain of a Butterworth low-pass filter of an order, as a function of ω."""

    def gain(w: np.ndarray) -> np.ndarray:
        return 1 / np.sqrt(1 + w ** (2 * order))

    return gain


def draw_caption(screen: pygame.Surface, font: pygame.font.Font, caption: str) -> None:
    """Draw the set's caption and the keys at the bottom left."""
    text = font.render(f"{caption}   (Space: next set)", True, (40, 40, 40))
    panel = pygame.Surface(
        (text.get_width() + 20, text.get_height() + 12), pygame.SRCALPHA
    )
    panel.fill((255, 255, 255, 225))
    panel.blit(text, (10, 6))
    screen.blit(panel, (12, screen.get_height() - panel.get_height() - 12))


def main() -> None:
    """Run the example."""
    pygame.init()
    screen = pygame.display.set_mode((1100, 720), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: titles, arrowheads, legends and text")
    font = pygame.font.Font(bundled_font_path(), 14)
    renderer = PygameRenderer()  # the planes share it, and its caches
    sets: list[Demo] = [Ball(renderer), Triangle(renderer), Filters(renderer)]
    clock = pygame.time.Clock()
    current = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                current = (current + 1) % len(sets)
            sets[current].plane.handle_event(event)
        caption = sets[current].draw(screen)
        draw_caption(screen, font, caption)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
