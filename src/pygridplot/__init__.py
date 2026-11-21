"""PyGridPlot: an interactive, pannable and zoomable Cartesian plane for pygame.

The library draws an adaptive grid with axes and clean tick labels on
linear or logarithmic axes, handles pan and zoom, and plots functions,
parametric, polar and implicit curves, points, polylines and polygons. It
stays out of your way: your program owns the main loop.

Everything a program needs is importable from here: :class:`Plane` (the
plane itself), the style classes (:class:`LineStyle`, :class:`PointStyle`,
:class:`PolygonStyle`), themes (:class:`GridTheme`, :data:`LIGHT`,
:data:`DARK`, :class:`NumberFormat`), configuration
(:class:`InteractionConfig`, :class:`KeyBindings`, :class:`SamplingConfig`,
:class:`CurveSampling`, :class:`ImplicitSampling`, :class:`ViewLimits`),
the :class:`View` with its :class:`ViewState`, and :func:`show`. The
subpackages (``pygridplot.core``, ``pygridplot.render``) are the
implementation; see ``docs/dev/index.md``.

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
        plane.plot(screen, lambda x: x**2)
        pygame.display.flip()

Or, for a quick look: ``pgp.show(np.sin, np.cos)``.
"""

from pygridplot.core.grid.formatting import NumberFormat
from pygridplot.core.grid.theme import DARK, LIGHT, GridTheme
from pygridplot.core.plot.curves import CurveSampling
from pygridplot.core.plot.implicit import ImplicitSampling
from pygridplot.core.plot.sampling import SamplingConfig
from pygridplot.core.plot.style import LineStyle, PointStyle, PolygonStyle
from pygridplot.core.view import View, ViewLimits, ViewState
from pygridplot.interaction import InteractionConfig, KeyBindings
from pygridplot.plane import Plane
from pygridplot.quick import show

__version__ = "0.1.0.dev0"

__all__ = [
    "DARK",
    "LIGHT",
    "CurveSampling",
    "GridTheme",
    "ImplicitSampling",
    "InteractionConfig",
    "KeyBindings",
    "LineStyle",
    "NumberFormat",
    "Plane",
    "PointStyle",
    "PolygonStyle",
    "SamplingConfig",
    "View",
    "ViewLimits",
    "ViewState",
    "__version__",
    "show",
]
