"""PyGridPlot: an interactive, pannable and zoomable Cartesian plane for pygame.

The library draws an adaptive grid with axes and clean tick labels on
linear or logarithmic axes, handles pan and zoom, and plots functions,
parametric, polar and implicit curves, points, polylines and polygons,
shades areas between graphs and regions where inequalities hold, draws
arrows, vector and slope fields and the terms of sequences, plots data
(scatter plots coloured by value, error bars, step plots, bar charts,
histograms, box and violin plots, named categories), writes text and
legends, and lets the user drag points and sliders around.
For calculus it differentiates and integrates functions and builds Taylor
polynomials and Fourier series, all as functions to plot. It stays out of
your way: your program owns the main loop.

Everything a program needs is importable from here: :class:`Grid` (the
plane itself), the style classes (:class:`LineStyle`, :class:`PointStyle`,
:class:`PolygonStyle`, :class:`FillStyle`, :class:`ArrowStyle`,
:class:`SequenceStyle`, :class:`BarStyle`, :class:`TextStyle`), themes
(:class:`GridTheme`, :data:`LIGHT`,
:data:`DARK`, :class:`NumberFormat`), configuration
(:class:`InteractionConfig`, :class:`KeyBindings`, :class:`FunctionSampling`,
:class:`CurveSampling`, :class:`ImplicitSampling`, :class:`ViewLimits`),
the :class:`View` with its :class:`ViewState`, :class:`DraggablePoint`
(made by :meth:`Grid.draggable`), :class:`Slider` (made by
:meth:`Grid.slider`) and :class:`Checkbox` (made by :meth:`Grid.checkbox`),
the calculus helpers :func:`derivative`
and :func:`derivatives` (exact derivatives of any order), :func:`integral`
and :func:`antiderivative`, :func:`taylor` and :func:`fourier`, and
:func:`color_map` (colours for values) and :func:`data_range` (ranges that
show data), :class:`Recorder` (videos and numbered pictures of a plane,
with :func:`find_ffmpeg`), and :func:`show`. The
subpackages (``pygridplot.core``, ``pygridplot.render``) are the
implementation; see ``docs/dev/index.md``.

Typical use::

    import pygame
    import pygridplot as pgp

    pygame.init()
    screen = pygame.display.set_mode((900, 600), pygame.RESIZABLE)
    grid = pgp.Grid()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            grid.handle_event(event)
        grid.draw(screen)
        grid.plot(screen, lambda x: x**2)
        pygame.display.flip()

Or, for a quick look: ``pgp.show(np.sin, np.cos)``.

PyGridPlot is free software under the GNU Lesser General Public License,
version 2.1 or later; see ``LICENSE``.
"""

# First: checks that pygame is pygame-ce, before anything uses it.
from pygridplot import _pygame_ce  # noqa: F401  # isort: skip
from pygridplot.color_maps import color_map
from pygridplot.core.grid.formatting import NumberFormat
from pygridplot.core.grid.theme import DARK, LIGHT, GridTheme
from pygridplot.core.handles.checkbox import Checkbox
from pygridplot.core.handles.point import DraggablePoint
from pygridplot.core.handles.slider import Slider
from pygridplot.core.plot.curves import CurveSampling
from pygridplot.core.plot.data import data_range
from pygridplot.core.plot.derivatives import derivative, derivatives
from pygridplot.core.plot.implicit import ImplicitSampling
from pygridplot.core.plot.integrals import antiderivative, integral
from pygridplot.core.plot.memo import CacheStats
from pygridplot.core.plot.sampling import FunctionSampling
from pygridplot.core.plot.series import fourier, taylor
from pygridplot.core.plot.style import (
    ArrowStyle,
    BarStyle,
    FillStyle,
    LineStyle,
    PointStyle,
    PolygonStyle,
    SequenceStyle,
    TextStyle,
)
from pygridplot.core.view import View, ViewLimits, ViewState
from pygridplot.grid import Grid
from pygridplot.interaction import InteractionConfig, KeyBindings
from pygridplot.quick import show
from pygridplot.recording import Recorder, find_ffmpeg

__version__ = "0.1.0.dev0"

__all__ = [
    "DARK",
    "LIGHT",
    "ArrowStyle",
    "BarStyle",
    "CacheStats",
    "Checkbox",
    "CurveSampling",
    "DraggablePoint",
    "FillStyle",
    "FunctionSampling",
    "Grid",
    "GridTheme",
    "ImplicitSampling",
    "InteractionConfig",
    "KeyBindings",
    "LineStyle",
    "NumberFormat",
    "PointStyle",
    "PolygonStyle",
    "Recorder",
    "SequenceStyle",
    "Slider",
    "TextStyle",
    "View",
    "ViewLimits",
    "ViewState",
    "__version__",
    "antiderivative",
    "color_map",
    "data_range",
    "derivative",
    "derivatives",
    "find_ffmpeg",
    "fourier",
    "integral",
    "show",
    "taylor",
]
