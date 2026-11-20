"""Animated plots: a moving sine wave, a point on a circle and a rotating polygon.

This is the "typical user program" from the project plan. Everything is
drawn again every frame (immediate mode), so animating only means computing
new values each frame. Pan and zoom work as usual while it runs.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

import pygridplot as pgp


def triangle(t: float) -> list[tuple[float, float]]:
    """Return the vertices of a triangle rotating about (3, 2).

    Args:
        t: Time in seconds.

    Returns:
        Three world-space vertices.
    """
    return [
        (
            3 + 1.5 * math.cos(t + k * 2 * math.pi / 3),
            2 + 1.5 * math.sin(t + k * 2 * math.pi / 3),
        )
        for k in range(3)
    ]


def main() -> None:
    """Open a window and animate until it is closed."""
    pygame.init()
    screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: animation")
    plane = pgp.Plane()
    clock = pygame.time.Clock()
    t = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            plane.handle_event(event)
        t += clock.tick(60) / 1000

        plane.draw_grid(screen)
        plane.plot(screen, lambda x, t=t: np.sin(x - t))  # bind t (late binding)
        plane.plot(screen, np.tan, width=1.5, opacity=0.6)
        plane.polyline(
            screen, [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 90)]
        )
        plane.polygon(screen, triangle(t))
        plane.points(screen, [(math.cos(t), math.sin(t))], radius=6, edge_color="white")
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
