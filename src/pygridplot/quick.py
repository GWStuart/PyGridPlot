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

from pygridplot.core.grid.theme import GridTheme
from pygridplot.core.plot.evaluation import Function
from pygridplot.plane import Plane


def show(
    *functions: Function,
    size: tuple[int, int] = (960, 640),
    title: str = "PyGridPlot",
    theme: GridTheme | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    center: tuple[float, float] | None = None,
    scale: float | tuple[float, float] | None = None,
    width: float | None = None,
    fps: int = 60,
) -> None:
    """Open a window, plot ``functions`` and run until the window is closed.

    Each function gets the next colour of the theme's palette. The view
    options are the same as :class:`~pygridplot.Plane`'s, for example
    ``show(np.exp, log_y=True)`` or ``show(np.sin, x_range=(0, 10))``.

    Args:
        *functions: Functions ``y = f(x)`` to plot.
        size: Initial window size in pixels.
        title: Window title.
        theme: Plane appearance (defaults to :data:`pygridplot.LIGHT`).
        x_range: ``(x_min, x_max)`` to show at first.
        y_range: ``(y_min, y_max)`` to show at first.
        log_x: Make the x axis logarithmic.
        log_y: Make the y axis logarithmic.
        center: World point initially at the middle of the window.
        scale: Initial pixels per world unit (per decade on a log axis).
        width: Line width in pixels for the functions (default 2.5).
        fps: Frame-rate cap.
    """
    owns_pygame = not pygame.get_init()
    pygame.init()
    try:
        screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption(title)
        plane = Plane(
            x_range=x_range,
            y_range=y_range,
            log_x=log_x,
            log_y=log_y,
            center=center,
            scale=scale,
            equal_aspect=False if isinstance(scale, tuple) else None,
            theme=theme,
        )
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
