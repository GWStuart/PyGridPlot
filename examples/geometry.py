"""Geometry: a triangle you can reshape, and its circumcircle.

Drag the corners. Every frame works out, from where the corners are, the
circumcentre (the point equally far from all three) and the circle
through them, and draws them with ``polygon``, ``parametric``,
``polyline``, ``points`` and ``text``.
"""

import math

import numpy as np
import pygame

import pygridplot as pgp

Point = tuple[float, float]


def circumcentre(a: Point, b: Point, c: Point) -> Point | None:
    """Return the point equally far from a, b and c (None if in a line)."""
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    a2, b2, c2 = ax**2 + ay**2, bx**2 + by**2, cx**2 + cy**2
    x = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    y = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return x, y


def main() -> None:
    """Open a window and draw until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: geometry")
    clock = pygame.time.Clock()
    grid = pgp.Grid(x_range=(-7.5, 7.5))
    corners = [grid.draggable(p) for p in [(-3.5, -2.0), (3.5, -1.5), (0.5, 3.0)]]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)

        a, b, c = (corner.position for corner in corners)
        grid.draw(screen)
        grid.polygon(screen, [a, b, c], label="triangle ABC")
        centre = circumcentre(a, b, c)
        if centre is not None:
            (x, y), r = centre, math.dist(centre, a)
            grid.parametric(
                screen,
                lambda t: x + r * np.cos(t),
                lambda t: y + r * np.sin(t),
                (0, 2 * np.pi),
                label=f"circumcircle, radius {r:.2f}",
            )
            for corner in (a, b, c):
                grid.polyline(screen, [centre, corner], color="gray", width=1)
            grid.points(screen, [centre], color="black", label="circumcentre")
        for name, corner in zip("ABC", (a, b, c), strict=True):
            grid.text(screen, name, corner, anchor="bottom-left", gap=10, size=16)
        grid.draw_controls(screen)
        grid.legend(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
