"""A ready-made main loop for quickly looking at functions.

:func:`show` opens a resizable window with an interactive
:class:`~pygridplot.Plane`, plots the given functions, and runs until the
window is closed (or Escape is pressed)::

    import numpy as np
    import pygridplot as pgp

    pgp.show(np.sin, lambda x: x**2 / 4)

It is meant for quick experiments. For animation or interaction, write your
own loop as shown in :mod:`pygridplot.plane`.
"""

from __future__ import annotations

import pygame

from pygridplot.core.evaluation import Function
from pygridplot.core.theme import GridTheme
from pygridplot.plane import Plane


def show(
    *functions: Function,
    size: tuple[int, int] = (960, 640),
    title: str = "PyGridPlot",
    theme: GridTheme | None = None,
    center: tuple[float, float] = (0.0, 0.0),
    scale: float = 50.0,
    width: float | None = None,
    fps: int = 60,
) -> None:
    """Open a window, plot ``functions`` and run until the window is closed.

    Each function gets the next colour of the theme's palette.

    Args:
        *functions: Functions ``y = f(x)`` to plot.
        size: Initial window size in pixels.
        title: Window title.
        theme: Plane appearance (defaults to :data:`pygridplot.LIGHT`).
        center: World point initially at the middle of the window.
        scale: Initial pixels per world unit.
        width: Line width in pixels for the functions (default 2.5).
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
            for func in functions:
                plane.plot(screen, func, width=width)
            pygame.display.flip()
            clock.tick(fps)
    finally:
        if owns_pygame:
            pygame.quit()
