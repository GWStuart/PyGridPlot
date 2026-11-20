"""PyGridPlot: an interactive, pannable and zoomable Cartesian plane for pygame.

The library draws an adaptive grid with axes and clean tick labels, handles
pan and zoom, and stays out of your way: your program owns the main loop.

Typical use::

    import pygame
    import pygridplot as pgp

    pygame.init()
    screen = pygame.display.set_mode((900, 600), pygame.RESIZABLE)
    plane = pgp.Plane()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            plane.handle_event(event)
        plane.draw_grid(screen)
        pygame.display.flip()

Or, for a quick look: ``pgp.show()``.
"""

from pygridplot.core.labels import NumberFormat
from pygridplot.core.theme import DARK, LIGHT, GridTheme
from pygridplot.core.view import View, ViewLimits, ViewState
from pygridplot.interaction import InteractionConfig, KeyBindings
from pygridplot.plane import Plane
from pygridplot.quick import show

__version__ = "0.1.0.dev0"

__all__ = [
    "DARK",
    "LIGHT",
    "GridTheme",
    "InteractionConfig",
    "KeyBindings",
    "NumberFormat",
    "Plane",
    "View",
    "ViewLimits",
    "ViewState",
    "__version__",
    "show",
]
