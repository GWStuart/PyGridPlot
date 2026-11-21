"""A ready-made main loop for quickly looking at functions and graphs.

:func:`show` opens a resizable window with an interactive
:class:`~pygridplot.Plane`, draws what it is given, and runs until the
window is closed (or Escape is pressed)::

    import numpy as np
    import pygridplot as pgp

    pgp.show(np.sin, lambda x: x**2 / 4)
    pgp.show("y = x^2", "x^2 + y^2 = 9", "r = 1 + cos(θ)")
    pgp.show(graph)                    # a Graph, with its draggable points

It is meant for quick experiments. For animation or interaction, write your
own loop as shown in :mod:`pygridplot.plane`.
"""

from __future__ import annotations

import pygame

from pygridplot.core.grid.theme import GridTheme
from pygridplot.core.plot.evaluation import FunctionLike
from pygridplot.graph import Graph
from pygridplot.objects import Function, LineObject, StyledObject
from pygridplot.plane import Drawable, Plane


def show(
    *functions: FunctionLike | str | Drawable,
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

    Each function or line gets the next colour of the theme's palette. The
    view options are the same as :class:`~pygridplot.Plane`'s, for example
    ``show(np.exp, log_y=True)`` or ``show(np.sin, x_range=(0, 10))``.

    Args:
        *functions: What to draw: functions ``y = f(x)``, lines of maths as
            :meth:`Graph.add <pygridplot.Graph.add>` reads them
            (``"x^2 + y^2 = 9"``), objects (``pgp.Point((1, 2),
            draggable=True)``, ...), or whole :class:`~pygridplot.Graph`
            objects (a graph with a saved view opens at it, unless view
            options are given).
        size: Initial window size in pixels.
        title: Window title.
        theme: Plane appearance (defaults to :data:`pygridplot.LIGHT`).
        x_range: ``(x_min, x_max)`` to show at first.
        y_range: ``(y_min, y_max)`` to show at first.
        log_x: Make the x axis logarithmic.
        log_y: Make the y axis logarithmic.
        center: World point initially at the middle of the window.
        scale: Initial pixels per world unit (per decade on a log axis).
        width: Line width in pixels for the functions and lines (default 2.5).
        fps: Frame-rate cap.

    Raises:
        ExpressionError: If a line is not valid.
    """
    lines = Graph()  # the lines of maths, which may use each other's names
    shown: list[Drawable] = [lines]
    slot = 0  # one palette sequence for everything given, in order
    for thing in functions:
        if isinstance(thing, Drawable):  # an object (a Function is also callable)
            shown.append(thing)
            if isinstance(thing, StyledObject) and thing.palette is None:
                thing.palette, slot = slot, slot + 1
            continue
        item = lines.add(thing) if isinstance(thing, str) else Function(thing)
        if isinstance(item, StyledObject):
            item.palette, slot = slot, slot + 1
        if width is not None and isinstance(item, LineObject):
            item.width = width
        if item.graph is None:
            shown.append(item)
    view_given = any(v is not None for v in (x_range, y_range, center, scale))
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
        saved = next(
            (g for g in shown if isinstance(g, Graph) and g.view is not None), None
        )
        if saved is not None and not view_given and not (log_x or log_y):
            saved.apply_view(plane)
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
