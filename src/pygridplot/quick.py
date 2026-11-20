"""A ready-made main loop for quickly looking at a plane.

:func:`show` opens a resizable window containing an interactive
:class:`~pygridplot.Plane` and runs until the window is closed (or Escape is
pressed). It is meant for quick experiments. For anything else, write your
own loop as shown in :mod:`pygridplot.plane`.
"""

from __future__ import annotations

import pygame

from pygridplot.core.theme import GridTheme
from pygridplot.plane import Plane


def show(
    *,
    size: tuple[int, int] = (960, 640),
    title: str = "PyGridPlot",
    theme: GridTheme | None = None,
    center: tuple[float, float] = (0.0, 0.0),
    scale: float = 50.0,
    fps: int = 60,
) -> None:
    """Open a window with an interactive plane and run until it is closed.

    Plotting functions through ``show`` will arrive with function plotting.
    For now it shows the grid itself.

    Args:
        size: Initial window size in pixels.
        title: Window title.
        theme: Grid appearance (defaults to :data:`pygridplot.LIGHT`).
        center: World point initially at the middle of the window.
        scale: Initial pixels per world unit.
        fps: Frame-rate cap.
    """
    owns_pygame = not pygame.get_init()
    pygame.init()
    try:
        screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption(title)
        plane = Plane(center=center, scale=scale, theme=theme)
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                plane.handle_event(event)
            plane.draw_grid(screen)
            pygame.display.flip()
            clock.tick(fps)
    finally:
        if owns_pygame:
            pygame.quit()
