"""A ready-made main loop for quickly looking at functions and objects.

:func:`show` opens a resizable window with an interactive
:class:`~pygridplot.Plane`, draws what it is given, and runs until the
window is closed (or Escape is pressed)::

    import numpy as np
    import pygridplot as pgp

    pgp.show(np.sin, lambda x: x**2 / 4)
    pgp.show(pgp.ImplicitCurve(lambda x, y: x**2 + y**2, 9), np.cos)

It is meant for quick experiments. For animation or interaction, write your
own loop as shown in :mod:`pygridplot.plane`.
"""

from __future__ import annotations

import pygame

from pygridplot.core.grid.theme import GridTheme
from pygridplot.core.plot.evaluation import FunctionLike
from pygridplot.objects import Function, StyledObject
from pygridplot.plane import Drawable, Plane


def show(
    *functions: FunctionLike | Drawable,
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
    """Open a window, draw what is given and run until the window is closed.

    Each function or object without a colour gets the next colour of the
    theme's palette. The
    view options are the same as :class:`~pygridplot.Plane`'s, for example
    ``show(np.exp, log_y=True)`` or ``show(np.sin, x_range=(0, 10))``.

    Args:
        *functions: What to draw: functions ``y = f(x)``, or objects
            (``pgp.ImplicitCurve(...)``, ``pgp.Point((1, 2),
            draggable=True)``, ...).
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
    shown: list[Drawable] = []
    for thing in functions:
        if isinstance(thing, Drawable):  # an object (a Function is also callable)
            shown.append(thing)
            continue
        shown.append(Function(thing, width=width))
    slot = 0  # one palette sequence for everything given, in order
    for obj in shown:
        if isinstance(obj, StyledObject) and obj.palette is None:
            obj.palette, slot = slot, slot + 1
    owns_pygame = not pygame.get_init()
    pygame.init()
    plane: Plane | None = None
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
        if shown:
            plane.add(*shown)
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                plane.handle_event(event)
            plane.draw(screen)
            pygame.display.flip()
            clock.tick(fps)
    finally:
        if plane is not None:  # the objects outlive the window's plane
            plane.clear_objects()
        if owns_pygame:
            pygame.quit()
