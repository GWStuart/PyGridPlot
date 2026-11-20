"""Points, polylines and polygons.

* Points keep the same size in pixels at every zoom level; every marker
  shape is shown, with and without an outline.
* Polygons are in world units, so they grow and shrink with the zoom. A
  five-pointed star shows the difference between the two fill rules.
* A spiral polyline, and a polygon and points that move every frame.

Press ``Space`` to pause the animation.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.core.primitives import MarkerShape

MARKERS: tuple[MarkerShape, ...] = (
    "circle",
    "square",
    "diamond",
    "cross",
    "plus",
)


def star(center: tuple[float, float], radius: float) -> np.ndarray:
    """Return the vertices of a self-intersecting five-pointed star.

    Args:
        center: Centre of the star.
        radius: Distance from the centre to each point.

    Returns:
        Five vertices, shape ``(5, 2)``.
    """
    angles = np.pi / 2 + np.arange(5) * 4 * np.pi / 5
    return np.column_stack(
        (center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles))
    )


def regular_polygon(
    sides: int, center: tuple[float, float], angle: float
) -> np.ndarray:
    """Return the vertices of a regular polygon of radius 1.2.

    Args:
        sides: Number of sides.
        center: Centre of the polygon.
        angle: Rotation in radians.

    Returns:
        The vertices, shape ``(sides, 2)``.
    """
    a = angle + np.arange(sides) * 2 * np.pi / sides
    return np.column_stack((center[0] + 1.2 * np.cos(a), center[1] + 1.2 * np.sin(a)))


def draw_scene(plane: pgp.Plane, screen: pygame.Surface, t: float) -> None:
    """Draw every shape of the example.

    Args:
        plane: The plane to draw on.
        screen: The display surface.
        t: Animation time in seconds.
    """
    for i, marker in enumerate(MARKERS):
        x = -7 + 1.5 * i
        plane.points(screen, (x, 4.5), marker=marker, radius=7)
        plane.points(
            screen, (x, 3.3), marker=marker, radius=7, edge_color="white", edge_width=2
        )
    plane.polygon(screen, star((3.5, 3.0), 1.8), fill_rule="nonzero")
    plane.polygon(screen, star((7.5, 3.0), 1.8), fill_rule="evenodd")
    plane.polygon(screen, regular_polygon(6, (-4.0, -2.0), t * 0.7), fill_opacity=0.35)
    theta = np.linspace(0, 6 * np.pi, 400)
    spiral = np.column_stack(
        (3.5 + 0.12 * theta * np.cos(theta), -2.5 + 0.12 * theta * np.sin(theta))
    )
    plane.polyline(screen, spiral, width=2)
    xs = np.linspace(-7.5, 7.5, 61)
    wave = np.column_stack((xs, -4.6 + 0.4 * np.sin(1.5 * xs - 2 * t)))
    plane.polyline(screen, wave, width=1.5, opacity=0.5)
    plane.points(screen, wave, radius=3)
    plane.points(screen, (math.cos(t), math.sin(t)), radius=6, edge_color="white")


def main() -> None:
    """Run the example."""
    pygame.init()
    screen = pygame.display.set_mode((1100, 720), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: points, polylines and polygons")
    plane = pgp.Plane(scale=55)
    clock = pygame.time.Clock()
    t = 0.0
    paused = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = not paused
            plane.handle_event(event)
        dt = clock.tick(60) / 1000
        if not paused:
            t += dt
        plane.draw_grid(screen)
        draw_scene(plane, screen, t)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
