"""The public :class:`Grid` class: an interactive Cartesian plane for pygame.

A plane ties together a :class:`~pygridplot.core.view.View` (what is
visible), a :class:`~pygridplot.core.grid.theme.GridTheme` (how it looks), an
:class:`~pygridplot.interaction.InteractionController` (pan and zoom) and a
renderer (drawing). It is a *library* object: your program owns the main
loop and calls the plane each frame::

    import numpy as np
    import pygame
    import pygridplot as pgp

    pygame.init()
    screen = pygame.display.set_mode((900, 600), pygame.RESIZABLE)
    grid = pgp.Grid()
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid.handle_event(event)
        grid.draw(screen)
        grid.plot(screen, np.sin)
        grid.polar(screen, lambda t: 1 + np.cos(t))
        grid.implicit(screen, lambda x, y: x**2 + y**2, 9)
        grid.inequality(screen, lambda x, y: y < np.cos(x))
        grid.vector_field(screen, lambda x, y: (-y, x))
        grid.arrow(screen, (0, 0), (2, 1))
        grid.points(screen, [(0, 0), (np.pi, 0)])
        grid.polygon(screen, [(1, 1), (3, 1), (2, 2.5)])
        pygame.display.flip()
        clock.tick(60)

Drawing is *immediate mode*: everything is recomputed from the current view
every frame, so plots follow panning and zooming automatically and animating
them needs no extra work. The one exception is draggable points
(:meth:`Grid.draggable`), which are created once and keep the position the
user drags them to.
"""

from __future__ import annotations

import math
import os
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Final, Literal, overload

import numpy as np
import pygame
from numpy.typing import ArrayLike, NDArray

from pygridplot.color_maps import ColorMapLike, resolve_stops
from pygridplot.core.coords import ColorLike, ScreenRect
from pygridplot.core.framing import ViewFraming, aspect_choice, equal_aspect_for
from pygridplot.core.grid.categories import Categories, check_categories
from pygridplot.core.grid.scene import GridScene, GridState, build_grid
from pygridplot.core.grid.theme import LIGHT, GridTheme
from pygridplot.core.handles.checkbox import Checkbox
from pygridplot.core.handles.color_bar import (
    ColorBarPanel,
    color_bar_primitives,
    layout_color_bar,
)
from pygridplot.core.handles.constraints import Lock, Range
from pygridplot.core.handles.drawing import coordinate_item, point_primitives
from pygridplot.core.handles.legend import (
    LegendEntry,
    LegendPanel,
    SwatchStyle,
    check_label,
    layout_legend,
    legend_texts,
    swatch_primitives,
)
from pygridplot.core.handles.panel import (
    Control,
    Panel,
    background_primitives,
    layout_panels,
    panel_color,
    row_key,
    row_primitives,
)
from pygridplot.core.handles.point import Coordinates, CurveSpec, DraggablePoint
from pygridplot.core.handles.slider import (
    CORNERS,
    DEFAULT_DURATION,
    Corner,
    PlayMode,
    Slider,
)
from pygridplot.core.plot.builders import (
    Boundary,
    CurveComponent,
    area_fill,
    arrow_fill,
    bar_shapes,
    box_plot_primitives,
    error_bar_stroke,
    function_stroke,
    histogram_shapes,
    implicit_stroke,
    parametric_stroke,
    point_markers,
    polar_stroke,
    polygon_shapes,
    polyline_stroke,
    region_fill,
    sequence_primitives,
    slope_field_strokes,
    text_items,
    vector_field_fills,
    violin_primitives,
)
from pygridplot.core.plot.color_maps import check_range
from pygridplot.core.plot.curves import DEFAULT_CURVE_SAMPLING, CurveSampling
from pygridplot.core.plot.dashes import Dash
from pygridplot.core.plot.data import (
    Step,
    as_bars,
    as_baselines,
    as_errors,
    check_step,
    check_width,
    per_point_colors,
    per_point_radii,
)
from pygridplot.core.plot.distributions import (
    as_groups,
    check_bandwidth,
    check_positions,
    check_whisker,
)
from pygridplot.core.plot.evaluation import Function, VectorFunction, XYFunction
from pygridplot.core.plot.fields import DEFAULT_SPACING
from pygridplot.core.plot.geometry import as_points
from pygridplot.core.plot.histograms import as_samples, check_edges
from pygridplot.core.plot.implicit import DEFAULT_IMPLICIT_SAMPLING, ImplicitSampling
from pygridplot.core.plot.memo import CacheStats, PlotCache, same_inputs
from pygridplot.core.plot.sampling import DEFAULT_SAMPLING, FunctionSampling
from pygridplot.core.plot.sequences import (
    as_terms,
    check_baseline,
    check_start,
    integer_arguments,
)
from pygridplot.core.plot.style import (
    ArrowStyle,
    BarStyle,
    ColorCycle,
    FillStyle,
    LineStyle,
    PointStyle,
    PolygonStyle,
    SequenceStyle,
    TextStyle,
    with_overrides,
)
from pygridplot.core.plot.text import Anchor, check_anchor, check_gap
from pygridplot.core.primitives import (
    Fill,
    FillRule,
    FontSpec,
    Image,
    Markers,
    MarkerShape,
    Stroke,
    TextItem,
)
from pygridplot.core.scaling import check_scale
from pygridplot.core.view import View, ViewLimits, ViewState
from pygridplot.dragging import DragController, PanelController
from pygridplot.interaction import InteractionConfig, InteractionController
from pygridplot.render.pygame_renderer import PygameRenderer
from pygridplot.render.replay import Drawn, Replayer

_PICTURE_TYPES: Final = (".png", ".jpg", ".jpeg", ".bmp", ".tga")
"""The file types :meth:`Grid.save` writes (by extension)."""

_MAX_PLAY_MS: Final = 250
"""Longest time a playing slider moves on by in one frame, in milliseconds.

So a stalled program (a dragged window, a breakpoint) does not make sliders
jump, while one drawing as few as four frames a second plays in real time.
"""

_DEFAULT_LINE = LineStyle()
_DEFAULT_POINT = PointStyle()
_DEFAULT_POLYGON = PolygonStyle()
_DEFAULT_FILL = FillStyle()
_DEFAULT_ARROW = ArrowStyle()
_DEFAULT_FIELD_ARROW = ArrowStyle(width=1.5)
_DEFAULT_SLOPE = LineStyle(width=1.5)
_DEFAULT_SEQUENCE = SequenceStyle()
_DEFAULT_TEXT = TextStyle()
_DEFAULT_BAR = BarStyle()


class Grid:
    """An interactive Cartesian plane drawn into a rectangle of a pygame surface.

    The initial view can be given like a graphing calculator's window
    settings, ``x_range=(x_min, x_max)`` and ``y_range=(y_min, y_max)``, or as a
    ``center`` and ``scale``. Either axis can be logarithmic.

    Args:
        rect: The screen rectangle to draw into. ``None`` (the default) fills
            whatever surface :meth:`draw` is given and follows it when
            the window is resized.
        x_range: ``(x_min, x_max)`` to show across the plane's width at
            first. Overrides ``center`` and ``scale`` for the x axis.
        y_range: ``(y_min, y_max)`` to show across the plane's height at
            first. Overrides ``center`` and ``scale`` for the y axis.
        log_x: Make the x axis logarithmic (base 10). Only positive x values
            can be shown.
        log_y: Make the y axis logarithmic (base 10). Only positive y values
            can be shown.
        center: World point initially shown at the middle of the plane.
            Defaults to 0 on a linear axis and 1 on a log axis.
        scale: Initial pixels per world unit (per decade on a log axis): one
            number, or an ``(x, y)`` pair when the aspect is not equal.
            Defaults to 50 per unit and 150 per decade.
        equal_aspect: Keep both axes at the same scale, so circles stay
            round. ``None`` (the default) decides automatically: on, unless
            both ranges are given (each then fills its axis exactly) or only
            one axis is logarithmic.
        theme: Grid appearance. Defaults to :data:`pygridplot.LIGHT`.
        interactive: Whether the built-in pan/zoom controls are active.
        interaction: Settings for the built-in controls.
        limits: Zoom and pan limits (see :class:`~pygridplot.ViewLimits`).
        renderer: Renderer to draw with. Planes can share one renderer, and
            with it their font and label caches.
        sampling: How functions are sampled by :meth:`plot` (the defaults suit
            almost every function).
        cache: Reuse plots that have not changed since the previous frame
            (see :attr:`cache`). It only saves time; the drawing is the same.
    """

    def __init__(
        self,
        rect: object | None = None,
        *,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        log_x: bool = False,
        log_y: bool = False,
        center: tuple[float, float] | None = None,
        scale: float | tuple[float, float] | None = None,
        equal_aspect: bool | None = None,
        theme: GridTheme | None = None,
        interactive: bool = True,
        interaction: InteractionConfig | None = None,
        limits: ViewLimits | None = None,
        renderer: PygameRenderer | None = None,
        sampling: FunctionSampling | None = None,
        cache: bool = True,
    ) -> None:
        self._auto_rect = rect is None
        initial = _display_rect() if rect is None else ScreenRect.from_any(rect)
        choice = aspect_choice(equal_aspect, x_range, y_range)
        self._view = View(
            initial,
            center,
            scale,
            equal_aspect=equal_aspect_for(choice, log_x, log_y),
            limits=limits,
            log_x=log_x,
            log_y=log_y,
        )
        # An automatic plane only learns its real size when it is first drawn
        # (the window may not exist, or may still change size, until then).
        self._framing = ViewFraming(self._view, choice, settled=not self._auto_rect)
        if x_range is not None or y_range is not None:
            self._framing.fit((x_range, y_range), home=True)
        self._theme = theme if theme is not None else LIGHT
        self.interactive = interactive
        """Whether the built-in pan and zoom controls react to input.

        Draggable points can be dragged either way (see
        :attr:`DraggablePoint.enabled <pygridplot.DraggablePoint.enabled>`)."""
        self._renderer = renderer if renderer is not None else PygameRenderer()
        self._grid_state = GridState()
        self._controller = InteractionController(
            self._view, interaction, on_reset=self.reset_view
        )
        self._draggables: list[DraggablePoint] = []
        self._draggable_colors: dict[DraggablePoint, int] = {}
        self._draggables_created = 0
        self._drag = DragController(
            self._view,
            self._controller.config,
            lambda: self._draggables,
            blocked=lambda: self._controller.dragging or self._panel_input.owns_cursor,
        )
        # The sliders and checkboxes, in creation order, and the palette colour
        # each was given (they take palette colours in creation order).
        self._controls: list[Control] = []
        self._control_colors: dict[Control, int] = {}
        self._controls_created = 0
        self._last_tick: float | None = None  # the play clock at the last frame
        # The clock playing sliders follow, in milliseconds: pygame's, or a
        # recorder's while the plane is recorded (see Recorder).
        self._play_clock: Callable[[], float] | None = None
        # What was drawn in this frame, for saving it (see save); and what
        # draws it again, made when first needed.
        self._drawing: list[Drawn] = []
        self._replayer: Replayer | None = None
        self._panels: tuple[Panel, ...] = ()  # as last drawn: what presses hit
        # The primitives last drawn for each panel and slider, with their keys.
        self._panel_drawings: dict[object, tuple[object, Any]] = {}
        self._panel_input = PanelController(
            self._controller.config,
            self._overlays,
            blocked=lambda: self._controller.dragging or self._drag.dragged is not None,
        )
        self._frame = 0  # counts begin_frame calls
        # The sliders' panels as last drawn, with the frame they were drawn in:
        # a legend in the same corner goes beyond them.
        self._panel_layout: tuple[int, tuple[Panel, ...]] = (0, ())
        self._legend_entries: list[LegendEntry] = []  # labelled calls this frame
        self._legend_panel: LegendPanel | None = None  # as drawn this frame
        # Where the legend was last drawn: a colour bar in its corner goes
        # beyond it.
        self._legend_layout: tuple[int, Corner, ScreenRect] | None = None
        self._color_bar_panel: ColorBarPanel | None = None  # as drawn this frame
        self._color_bar_drawing: tuple[object, Any] | None = None  # key, primitives
        # The primitives last drawn for the legend's panel and each of its
        # swatches, with their keys.
        self._legend_drawings: dict[object, tuple[object, Any]] = {}
        self._colors = ColorCycle(self._theme.palette)
        self._sampling = sampling if sampling is not None else DEFAULT_SAMPLING
        self._last_scene: GridScene | None = None
        self._grid_key: tuple[object, ...] | None = None
        self._plots = PlotCache(enabled=cache)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def view(self) -> View:
        """The plane's :class:`~pygridplot.View`.

        Use it to read or change the centre, scale and visible bounds
        directly.
        """
        return self._view

    @property
    def rect(self) -> ScreenRect:
        """The screen rectangle the plane occupies.

        Assign a rectangle (for example a ``pygame.Rect``) to move or resize
        the plane, or ``None`` to make it fill the surface it is drawn on.
        Resizing keeps the centre and scale, so more or less of the plane
        becomes visible.
        """
        return self._view.rect

    @rect.setter
    def rect(self, rect: object | None) -> None:
        self._auto_rect = rect is None
        if rect is not None:
            self._view.rect = rect
            self._framing.settle()

    @property
    def log_x(self) -> bool:
        """Whether the x axis is logarithmic.

        Assign to switch it: the visible x range is kept where possible (see
        :meth:`~pygridplot.View.set_log`) and becomes part of the new home
        view.
        """
        return self._view.log_x

    @log_x.setter
    def log_x(self, log: bool) -> None:
        self._set_log("x", log)

    @property
    def log_y(self) -> bool:
        """Whether the y axis is logarithmic (assign to switch it, as for x)."""
        return self._view.log_y

    @log_y.setter
    def log_y(self, log: bool) -> None:
        self._set_log("y", log)

    @property
    def theme(self) -> GridTheme:
        """The grid's appearance. Assign a new theme to restyle the plane."""
        return self._theme

    @theme.setter
    def theme(self, theme: GridTheme) -> None:
        self._theme = theme
        self._grid_state.reset()
        self._colors.palette = theme.palette

    @property
    def sampling(self) -> FunctionSampling:
        """How :meth:`plot` samples functions (assign to change it)."""
        return self._sampling

    @sampling.setter
    def sampling(self, config: FunctionSampling) -> None:
        self._sampling = config

    @property
    def cache(self) -> bool:
        """Whether plots that have not changed since the previous frame are reused.

        On by default. A plotting call is reused when it is the same kind of
        call, in the same place among the frame's calls, with the same view,
        style and inputs as in the previous frame, and its functions still
        give exactly the values they gave then (they are checked at every
        point where they were evaluated). The drawing is then exactly what
        drawing again would give, so turning the cache off changes only the
        speed; do so to measure how long drawing takes, or if your functions
        are not *pointwise* (their value at ``x`` depends on the other points
        they are called with).
        """
        return self._plots.enabled

    @cache.setter
    def cache(self, enabled: bool) -> None:
        self._plots.enabled = bool(enabled)
        if not enabled:
            self.clear_cache()

    @property
    def cache_stats(self) -> CacheStats:
        """How many plotting calls were reused (hits) and built again (misses)."""
        return self._plots.stats

    def clear_cache(self) -> None:
        """Forget every reused plot and grid, and the counts in :attr:`cache_stats`."""
        self._plots.clear()
        self._grid_key = None

    @property
    def interaction(self) -> InteractionConfig:
        """Settings of the built-in pan/zoom controls and of dragging points."""
        return self._controller.config

    @interaction.setter
    def interaction(self, config: InteractionConfig) -> None:
        self._controller.config = config
        self._drag.config = config
        self._panel_input.config = config

    @property
    def renderer(self) -> PygameRenderer:
        """The renderer the plane draws with."""
        return self._renderer

    @property
    def last_scene(self) -> GridScene | None:
        """The grid drawn by the most recent :meth:`draw` call, if any.

        Useful for debugging overlays (it records the chosen spacings).
        """
        return self._last_scene

    # ------------------------------------------------------------------
    # Per-frame API
    # ------------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Let the plane react to an input event (dragging points, pan, zoom, keys).

        Pass every event, including mouse motion: the plane tracks the cursor
        to zoom about it and to highlight draggable points and sliders under
        it. A press on a slider sets it, a press over a draggable point drags
        the point, and anywhere else it pans; the wheel over a slider moves
        it, anywhere else it zooms.

        Args:
            event: A pygame event.

        Returns:
            ``True`` if the plane used the event, so your code can skip
            handling it again.
        """
        if self._panel_input.handle_event(event) or self._drag.handle_event(event):
            if event.type == pygame.MOUSEMOTION and self.interactive:
                # The pan/zoom controls still track the cursor, to zoom about it.
                self._controller.handle_event(event)
            return True
        if not self.interactive:
            return False
        return self._controller.handle_event(event)

    def update(self) -> bool:
        """Advance eased pan/zoom movement, and keep a dragged point under the cursor.

        :meth:`draw` calls this automatically, so you only need it if you
        stop drawing the grid but still want eased movement (and dragging
        while the view eases) to continue.

        Returns:
            ``True`` if the view changed.
        """
        changed = self._controller.update() if self.interactive else False
        self._drag.update()
        self._panel_input.update()
        return changed

    def begin_frame(self) -> None:
        """Start a new frame: the palette, the legend, points, sliders and checkboxes.

        The next object drawn without a colour gets the first palette colour
        again, the legend starts with no entries, playing sliders move on by
        the time since the previous frame (by pygame's clock,
        :func:`pygame.time.get_ticks`, at most a quarter of a second, so a
        stalled frame does not make them jump), and each draggable point, slider and
        checkbox updates its ``moved`` or ``changed`` flag. :meth:`draw`
        calls this automatically. Call it yourself only if you draw plots without
        drawing the grid, so that every object keeps the same palette colour
        from frame to frame, and the legend shows only the frame's plots.
        """
        self._frame += 1
        self._legend_entries.clear()
        self._legend_panel = None
        self._color_bar_panel = None
        self._colors.reset()
        self._plots.begin_frame()
        for point in self._draggables:
            point.begin_frame()
        clock = self._play_clock
        now = float(pygame.time.get_ticks()) if clock is None else clock()
        last, self._last_tick = self._last_tick, now
        seconds = 0.0 if last is None else min(max(now - last, 0), _MAX_PLAY_MS) / 1000
        for control in self._controls:
            if isinstance(control, Slider):
                control.advance(seconds)
            control.begin_frame()
        self._panels = ()  # until the controls are drawn again
        self._drawing = []

    def draw(
        self,
        surface: pygame.Surface,
        *,
        x_title: str | None = None,
        y_title: str | None = None,
        x_labels: Sequence[str] | Mapping[float, str] | None = None,
        y_labels: Sequence[str] | Mapping[float, str] | None = None,
    ) -> None:
        """Draw the background, grid lines, axes, tick labels and axis titles.

        Drawing is clipped to :attr:`rect`, so the plane can share a window
        with other content. Call it once per frame, before drawing anything
        that should appear on top of the grid. It also starts a new frame
        for the colour palette and the legend (see :meth:`begin_frame`).

        The titles sit at the axes' positive ends, as in textbook diagrams:
        the x axis' title at the right, just above the axis, and the y axis'
        title at the top, just right of it::

            grid.draw(screen, x_title="t (s)", y_title="v (m/s)")

        They stay on screen when their axis is not, pinned to the edge like
        the tick labels, and tick labels fade out where they meet one. Their
        size is the theme's :attr:`~pygridplot.GridTheme.title_size`; an
        arrowhead at each axis' end is the theme's
        :attr:`~pygridplot.GridTheme.show_arrows`.

        ``x_labels`` names positions along the x axis in place of its
        numbers, for categories (a bar chart's bars at 0, 1, 2, ...)::

            grid.bars(screen, [3, 1, 4])
            grid.draw(screen, x_labels=["red", "green", "blue"])  # at 0, 1, 2
            grid.draw(screen, x_labels={2019: "before", 2023: "after"})

        The names behave like tick labels: pinned to the edge, fading where
        they meet a title, and thinned to every 2nd, 4th, ... name when the
        axis is zoomed out so far that they would overlap. ``y_labels`` does
        the same for the y axis. A categorical axis has no shared ``0`` at
        the origin; the other axis labels its own zero.

        Args:
            surface: The surface to draw on (usually the display surface).
            x_title: A title for the x axis (``None`` for none).
            y_title: A title for the y axis (``None`` for none).
            x_labels: Names for x positions: a sequence of names (at 0, 1,
                2, ...) or a mapping from positions to names (``None`` for
                numbers).
            y_labels: Names for y positions, likewise.

        Raises:
            TypeError: If a title or name is not a string.
            ValueError: If a named position is not a finite number.
        """
        titles = (_title(x_title, "x_title"), _title(y_title, "y_title"))
        names = (
            check_categories(x_labels, "x_labels"),
            check_categories(y_labels, "y_labels"),
        )
        self._sync_rect(surface)
        self.update()
        self.begin_frame()
        scene = self._grid_scene(titles, names)
        self._renderer.draw_grid(surface, scene)
        self._drawing.append(("scene", scene))
        self._last_scene = scene

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(
        self,
        surface: pygame.Surface,
        func: Function,
        *,
        color: ColorLike | None = None,
        width: float | None = None,
        opacity: float | None = None,
        dash: Dash | None = None,
        domain: tuple[float, float] | None = None,
        style: LineStyle | None = None,
        sampling: FunctionSampling | None = None,
        label: str | None = None,
    ) -> None:
        """Plot the graph of an explicit function ``y = func(x)``.

        ``func`` can be vectorised (written with NumPy, the fastest) or work
        on single numbers only (for example written with :mod:`math`). Points
        where it raises an error or returns NaN or a complex number are left
        out, and jumps such as the asymptotes of ``tan`` are not joined up.

        Args:
            surface: The surface to draw on.
            func: The function to plot.
            color: Line colour (default: the next palette colour).
            width: Line width in pixels (default 2.5).
            opacity: Line opacity in ``[0, 1]``.
            dash: ``"dashed"``, ``"dotted"``, ``"solid"`` (default), or the
                lengths of the dashes and gaps in pixels, alternately, as
                ``(12, 6)``. The dashes stay put while panning.
            domain: Only plot where ``domain[0] <= x <= domain[1]``.
            style: A reusable :class:`~pygridplot.LineStyle`. The other
                keyword arguments override its fields.
            sampling: Override the plane's sampling settings for this call.
            label: A name for this plot in the legend (see :meth:`legend`).

        Note:
            When creating functions in a loop, bind the loop variable as a
            default argument: ``[lambda x, k=k: x**k for k in range(3)]``.
            Plain ``lambda x: x**k`` would make every function use the last
            ``k``.
        """
        self._sync_rect(surface)
        line = with_overrides(
            style or _DEFAULT_LINE,
            color=color,
            width=width,
            opacity=opacity,
            dash=dash,
        )
        color_ = self._colors.resolve(line.color)
        self._add_entry(label, line, color_)
        config = sampling if sampling is not None else self._sampling
        bounds = None if domain is None else (float(domain[0]), float(domain[1]))
        stroke = self._cached(
            "plot",
            (line, color_, bounds, config),
            [func],
            lambda fs: function_stroke(
                self._view, fs[0], line, color_, domain=domain, sampling=config
            ),
        )
        self._draw(surface, stroke)

    def parametric(
        self,
        surface: pygame.Surface,
        x: CurveComponent,
        y: CurveComponent,
        t_range: tuple[float, float],
        *,
        color: ColorLike | None = None,
        width: float | None = None,
        opacity: float | None = None,
        dash: Dash | None = None,
        style: LineStyle | None = None,
        sampling: CurveSampling | None = None,
        label: str | None = None,
    ) -> None:
        """Plot the parametric curve ``(x(t), y(t))`` for ``t`` in ``t_range``.

        ``x`` and ``y`` are functions of ``t``, written like the functions
        given to :meth:`plot` (vectorised or scalar-only, and allowed to fail
        for some ``t``), or plain numbers. The curve is sampled adaptively on
        screen, so tight turns, loops and spirals stay smooth at any zoom,
        and jumps and poles are not joined up.

        Args:
            surface: The surface to draw on.
            x: The x coordinate as a function of ``t`` (or a number).
            y: The y coordinate as a function of ``t`` (or a number).
            t_range: ``(t_min, t_max)``, for example ``(0, 2 * np.pi)``.
            color: Line colour (default: the next palette colour).
            width: Line width in pixels (default 2.5).
            opacity: Line opacity in ``[0, 1]``.
            dash: ``"dashed"``, ``"dotted"``, ``"solid"`` (default), or the
                lengths of the dashes and gaps in pixels, alternately, as
                ``(12, 6)``. The dashes stay put while panning.
            style: A reusable :class:`~pygridplot.LineStyle`. The other
                keyword arguments override its fields.
            sampling: How the curve is sampled (see
                :class:`~pygridplot.CurveSampling`; the defaults suit almost
                every curve).
            label: A name for this plot in the legend (see :meth:`legend`).

        Raises:
            ValueError: If ``t_range`` is not an increasing pair of numbers.
        """
        self._sync_rect(surface)
        line = with_overrides(
            style or _DEFAULT_LINE,
            color=color,
            width=width,
            opacity=opacity,
            dash=dash,
        )
        color_ = self._colors.resolve(line.color)
        self._add_entry(label, line, color_)
        config = sampling if sampling is not None else DEFAULT_CURVE_SAMPLING
        stroke = self._cached(
            "parametric",
            (line, color_, _pair(t_range), config),
            [x, y],
            lambda fs: parametric_stroke(
                self._view, fs[0], fs[1], t_range, line, color_, sampling=config
            ),
        )
        self._draw(surface, stroke)

    def polar(
        self,
        surface: pygame.Surface,
        r: CurveComponent,
        theta_range: tuple[float, float] = (0.0, 2 * math.pi),
        *,
        color: ColorLike | None = None,
        width: float | None = None,
        opacity: float | None = None,
        dash: Dash | None = None,
        style: LineStyle | None = None,
        sampling: CurveSampling | None = None,
        label: str | None = None,
    ) -> None:
        """Plot the polar curve ``r = r(θ)``.

        The point at angle ``θ`` (radians, anticlockwise from the positive x
        axis) is ``(r cos θ, r sin θ)``; a negative ``r`` lies on the opposite
        side of the origin. Angles where ``r`` is undefined are left out.

        Args:
            surface: The surface to draw on.
            r: The radius as a function of ``θ`` (or a number, for a circle).
            theta_range: ``(θ_min, θ_max)``. The default is one full turn;
                spirals and curves such as ``r = cos(θ / 3)`` need more.
            color: Line colour (default: the next palette colour).
            width: Line width in pixels (default 2.5).
            opacity: Line opacity in ``[0, 1]``.
            dash: ``"dashed"``, ``"dotted"``, ``"solid"`` (default), or the
                lengths of the dashes and gaps in pixels, alternately, as
                ``(12, 6)``. The dashes stay put while panning.
            style: A reusable :class:`~pygridplot.LineStyle`. The other
                keyword arguments override its fields.
            sampling: How the curve is sampled (see
                :class:`~pygridplot.CurveSampling`).
            label: A name for this plot in the legend (see :meth:`legend`).

        Raises:
            ValueError: If ``theta_range`` is not an increasing pair of numbers.
        """
        self._sync_rect(surface)
        line = with_overrides(
            style or _DEFAULT_LINE,
            color=color,
            width=width,
            opacity=opacity,
            dash=dash,
        )
        color_ = self._colors.resolve(line.color)
        self._add_entry(label, line, color_)
        config = sampling if sampling is not None else DEFAULT_CURVE_SAMPLING
        stroke = self._cached(
            "polar",
            (line, color_, _pair(theta_range), config),
            [r],
            lambda fs: polar_stroke(
                self._view, fs[0], theta_range, line, color_, sampling=config
            ),
        )
        self._draw(surface, stroke)

    def implicit(
        self,
        surface: pygame.Surface,
        func: XYFunction,
        level: ArrayLike = 0.0,
        *,
        color: ColorLike | None = None,
        width: float | None = None,
        opacity: float | None = None,
        dash: Dash | None = None,
        style: LineStyle | None = None,
        sampling: ImplicitSampling | None = None,
        label: str | None = None,
    ) -> None:
        """Plot the implicit curve ``func(x, y) = level``.

        The curve is found on a grid of samples a couple of pixels apart
        (marching squares), refined wherever it crosses the grid, so it can
        be any shape: closed loops, several separate pieces, branches that
        cross. Where ``func`` changes sign without passing through ``level``
        (a pole such as ``y - tan(x)``, or a jump such as ``y - floor(x)``)
        no line is drawn, and the curve stops where ``func`` is undefined.

        ``func`` must change sign across the curve: a curve where it only
        touches ``level`` (``(x - y)**2 = 0``) is not found, so write such
        equations without the square. Features smaller than about two
        pixels may be missed.

        Args:
            surface: The surface to draw on.
            func: ``F(x, y)``, taking and returning NumPy arrays (or plain
                numbers, at some cost in speed). It may raise errors or
                return NaN where it is undefined.
            level: The value of ``func`` along the curve, so that
                ``grid.implicit(screen, lambda x, y: x**2 + y**2, 9)``
                draws the circle ``x² + y² = 9``. A sequence or array of values
                draws a curve for each, all in one style (contour lines,
                like the height lines of a map); this is much faster than
                one call per value, as ``func`` is evaluated only once.
            color: Line colour (default: the next palette colour).
            width: Line width in pixels (default 2.5).
            opacity: Line opacity in ``[0, 1]``.
            dash: ``"dashed"``, ``"dotted"``, ``"solid"`` (default), or the
                lengths of the dashes and gaps in pixels, alternately, as
                ``(12, 6)``. The dashes stay put while panning.
            style: A reusable :class:`~pygridplot.LineStyle`. The other
                keyword arguments override its fields.
            sampling: How the curve is searched for (see
                :class:`~pygridplot.ImplicitSampling`).
            label: A name for this plot in the legend (see :meth:`legend`).

        Raises:
            ValueError: If a level is not a finite number.
        """
        self._sync_rect(surface)
        line = with_overrides(
            style or _DEFAULT_LINE,
            color=color,
            width=width,
            opacity=opacity,
            dash=dash,
        )
        color_ = self._colors.resolve(line.color)
        self._add_entry(label, line, color_)
        config = sampling if sampling is not None else DEFAULT_IMPLICIT_SAMPLING
        stroke = self._cached(
            "implicit",
            (line, color_, _numbers(level), config),
            [func],
            lambda fs: implicit_stroke(
                self._view, fs[0], line, color_, level=level, sampling=config
            ),
            arity=2,
        )
        self._draw(surface, stroke)

    def fill_between(
        self,
        surface: pygame.Surface,
        f: Boundary,
        g: Boundary = 0.0,
        *,
        color: ColorLike | None = None,
        opacity: float | None = None,
        domain: tuple[float, float] | None = None,
        style: FillStyle | None = None,
        sampling: FunctionSampling | None = None,
        label: str | None = None,
    ) -> None:
        """Shade the area between the graphs ``y = f(x)`` and ``y = g(x)``.

        Both sides are sampled exactly as :meth:`plot` samples a graph, so
        the shading meets the curves drawn with :meth:`plot` exactly. It
        covers only the x intervals where both sides are defined, and stops
        with a straight edge wherever either one jumps (as ``tan`` does at
        its asymptotes) or ends. Where the graphs cross, both parts are
        shaded.

        The area has no outline: plot the curves too, if they should show::

            grid.fill_between(screen, np.sin, 0, domain=(0, np.pi))
            grid.plot(screen, np.sin)

        Args:
            surface: The surface to draw on.
            f: One side: a function of ``x`` (vectorised or scalar-only,
                like for :meth:`plot`), or a number for a horizontal line.
            g: The other side, likewise (default ``0``: the area between
                ``f`` and the x axis). ``-math.inf`` or ``math.inf`` shades
                all the way to the bottom or top of the plane; on a log y
                axis, so does any number ``<= 0``.
            color: Fill colour (default: the next palette colour).
            opacity: Fill opacity in ``[0, 1]`` (default 0.25).
            domain: Only shade where ``domain[0] <= x <= domain[1]``.
            style: A reusable :class:`~pygridplot.FillStyle`. The other
                keyword arguments override its fields.
            sampling: Override the plane's sampling settings for this call
                (used for both sides).
            label: A name for this plot in the legend (see :meth:`legend`).

        Raises:
            TypeError: If ``f`` or ``g`` is neither callable nor a number.
            ValueError: If ``domain`` is not an increasing pair of numbers.
        """
        self._sync_rect(surface)
        fill = with_overrides(style or _DEFAULT_FILL, color=color, opacity=opacity)
        color_ = self._colors.resolve(fill.color)
        self._add_entry(label, fill, color_)
        config = sampling if sampling is not None else self._sampling
        bounds = None if domain is None else (float(domain[0]), float(domain[1]))
        area = self._cached(
            "fill_between",
            (fill, color_, bounds, config),
            [f, g],
            lambda fs: area_fill(
                self._view, fs[0], fs[1], fill, color_, domain=domain, sampling=config
            ),
        )
        self._draw(surface, area)

    def inequality(
        self,
        surface: pygame.Surface,
        condition: XYFunction,
        *,
        color: ColorLike | None = None,
        opacity: float | None = None,
        style: FillStyle | None = None,
        sampling: ImplicitSampling | None = None,
        label: str | None = None,
    ) -> None:
        """Shade the region where ``condition(x, y)`` holds.

        The condition is an inequality written in Python, such as
        ``lambda x, y: x**2 + y**2 < 4`` or ``lambda x, y: y < np.sin(x)``.
        Conditions combine with NumPy's ``&`` (and), ``|`` (or) and ``~``
        (not), each part in brackets::

            grid.inequality(screen, lambda x, y: (y > x**2) & (y < 4))

        (Python's ``and``, ``or`` and chained comparisons such as
        ``0 < x < 1`` work too, but only on single numbers, so they are
        much slower.) Any value that is not zero counts as true, so a
        function returning 0 or 1 works as well. Where the condition is
        undefined (it raises an error or returns NaN), the point is outside.

        The region is found like an implicit curve, on a grid of samples
        a couple of pixels apart, with its edges refined onto the true
        boundary, so it can be any shape: several pieces, holes, regions
        that end at an asymptote. Islands and holes smaller than about
        eight pixels across may be missed.

        The region has no outline: draw its boundary with :meth:`implicit`
        or :meth:`plot`, if it should show::

            grid.inequality(screen, lambda x, y: x**2 + y**2 < 4)
            grid.implicit(screen, lambda x, y: x**2 + y**2, 4)

        Args:
            surface: The surface to draw on.
            condition: ``condition(x, y)``, taking NumPy arrays and
                returning booleans (or plain numbers, at some cost in
                speed).
            color: Fill colour (default: the next palette colour).
            opacity: Fill opacity in ``[0, 1]`` (default 0.25).
            style: A reusable :class:`~pygridplot.FillStyle`. The other
                keyword arguments override its fields.
            sampling: How the region is searched for (the same settings as
                implicit curves: see :class:`~pygridplot.ImplicitSampling`).
            label: A name for this plot in the legend (see :meth:`legend`).
        """
        self._sync_rect(surface)
        fill = with_overrides(style or _DEFAULT_FILL, color=color, opacity=opacity)
        color_ = self._colors.resolve(fill.color)
        self._add_entry(label, fill, color_)
        config = sampling if sampling is not None else DEFAULT_IMPLICIT_SAMPLING
        area = self._cached(
            "inequality",
            (fill, color_, config),
            [condition],
            lambda fs: region_fill(self._view, fs[0], fill, color_, sampling=config),
            arity=2,
        )
        self._draw(surface, area)

    def arrow(
        self,
        surface: pygame.Surface,
        start: ArrayLike,
        end: ArrayLike,
        *,
        color: ColorLike | None = None,
        width: float | None = None,
        opacity: float | None = None,
        head_size: float | None = None,
        style: ArrowStyle | None = None,
        label: str | None = None,
    ) -> None:
        """Draw an arrow from ``start`` to ``end``, or many arrows at once.

        The arrow runs straight from ``start`` to its tip at ``end``, both
        in world units, so it moves and grows with the view; its width and
        head are in pixels, so it looks the same at every zoom. To draw a
        vector ``v`` at a point ``p``, give ``p`` and ``np.add(p, v)``.

        Each end is one point ``(x, y)`` or many (a sequence of points, or
        an ``(N, 2)`` array), for many arrows in one call, which is much
        faster than one call each. A single start is shared by all the
        ends, and a single end by all the starts::

            grid.arrow(screen, (0, 0), (2, 1))
            grid.arrow(screen, (0, 0), [(1, 0), (0, 1)])      # two, from the origin

        Arrows with a NaN coordinate, or whose ends are in the same place,
        are left out.

        Args:
            surface: The surface to draw on.
            start: Where the arrow starts (its tail).
            end: Where it points to (its tip).
            color: Arrow colour (default: the next palette colour).
            width: Shaft width in pixels (default 2.5).
            opacity: Opacity in ``[0, 1]``.
            head_size: Length of the head in pixels (default: in proportion
                to the width; ``0`` for no head).
            style: A reusable :class:`~pygridplot.ArrowStyle`. The other
                keyword arguments override its fields.
            label: A name for this plot in the legend (see :meth:`legend`).

        Raises:
            ValueError: If ``start`` or ``end`` is not one or more points, or
                they are both several points but not as many.
        """
        self._sync_rect(surface)
        arrow_style = with_overrides(
            style or _DEFAULT_ARROW,
            color=color,
            width=width,
            opacity=opacity,
            head_size=head_size,
        )
        color_ = self._colors.resolve(arrow_style.color)
        self._add_entry(label, arrow_style, color_)
        starts = as_points(start).copy()  # the caller may change them in place
        ends = as_points(end).copy()
        fill = self._cached(
            "arrow",
            (arrow_style, color_, starts, ends),
            [],
            lambda _: arrow_fill(self._view, starts, ends, arrow_style, color_),
        )
        self._draw(surface, fill)

    def vector_field(
        self,
        surface: pygame.Surface,
        field: VectorFunction,
        *,
        spacing: float | None = None,
        scale: float | None = None,
        color: ColorLike | None = None,
        width: float | None = None,
        opacity: float | None = None,
        head_size: float | None = None,
        style: ArrowStyle | None = None,
        label: str | None = None,
    ) -> None:
        """Draw the vector field ``(u, v) = field(x, y)`` as a lattice of arrows.

        ``field`` takes ``x`` and ``y`` and returns the vector there as a
        pair, written with NumPy like the functions given to
        :meth:`implicit`::

            grid.vector_field(screen, lambda x, y: (-y, x))  # a rotation
            grid.vector_field(screen, lambda x, y: (1, np.sin(x)))  # u is a number

        An arrow is centred on each point of a lattice that keeps about the
        same density on screen, ``spacing`` pixels apart. The lattice is
        fixed to the plane, so panning moves the arrows with it; while
        zooming, new arrows fade and grow in between the others (or shrink
        and fade out), so that their density stays the same and nothing
        jumps.

        By default every arrow has the same length, so the picture shows
        the field's direction. Give ``scale`` to show its magnitude too:
        each arrow is then ``scale`` pixels long per unit of magnitude, up
        to a cap of 0.8 × ``spacing``, so that arrows never run into each
        other. Where ``field`` raises an error, returns NaN or an infinite
        component, or is zero, no arrow is drawn.

        Arrows are tangent to the field's flow lines on screen, on log axes
        too (a vector ``(u, v)`` is taken as a direction in world units at
        its point).

        Args:
            surface: The surface to draw on.
            field: ``field(x, y) -> (u, v)``, taking NumPy arrays (or plain
                numbers, at some cost in speed).
            spacing: Distance between neighbouring arrows in pixels (default
                40, at least 10). While zooming it varies between this and
                twice this, as arrows fade in and out.
            scale: ``None`` for arrows of one length, or pixels of length
                per unit of the vector's magnitude.
            color: Arrow colour (default: the next palette colour).
            width: Shaft width in pixels (default 1.5).
            opacity: Opacity in ``[0, 1]``.
            head_size: Length of the heads in pixels (default: in proportion
                to the width).
            style: A reusable :class:`~pygridplot.ArrowStyle`. The other
                keyword arguments override its fields.
            label: A name for this plot in the legend (see :meth:`legend`).

        Raises:
            ValueError: If ``spacing`` or ``scale`` is not valid.
        """
        self._sync_rect(surface)
        arrow_style = with_overrides(
            style or _DEFAULT_FIELD_ARROW,
            color=color,
            width=width,
            opacity=opacity,
            head_size=head_size,
        )
        color_ = self._colors.resolve(arrow_style.color)
        self._add_entry(label, arrow_style, color_)
        gap = float(spacing) if spacing is not None else DEFAULT_SPACING
        magnitude = None if scale is None else float(scale)
        fills = self._cached(
            "vector_field",
            (arrow_style, color_, gap, magnitude),
            [field],
            lambda fs: vector_field_fills(
                self._view, fs[0], arrow_style, color_, spacing=gap, scale=magnitude
            ),
            arity=2,
            outputs=2,
        )
        self._draw(surface, *fills)

    def slope_field(
        self,
        surface: pygame.Surface,
        slope: XYFunction,
        *,
        spacing: float | None = None,
        color: ColorLike | None = None,
        width: float | None = None,
        opacity: float | None = None,
        style: LineStyle | None = None,
        label: str | None = None,
    ) -> None:
        """Draw the slope field of the differential equation ``dy/dx = slope(x, y)``.

        A short segment is drawn at each point of a lattice, with the slope
        the equation gives there, so the solution curves can be seen
        following the segments::

            grid.slope_field(screen, lambda x, y: x - y)
            grid.plot(screen, lambda x: x - 1 + 2 * np.exp(-x))     # one solution

        The lattice behaves as for :meth:`vector_field`: fixed to the plane
        when panning, with segments fading in and out while zooming. An
        infinite slope gives a vertical segment; where ``slope`` raises an
        error or returns NaN, no segment is drawn. For arrows instead of
        segments, draw the vector field ``lambda x, y: (1, slope(x, y))``.

        Args:
            surface: The surface to draw on.
            slope: ``dy/dx`` as a function of ``x`` and ``y``, taking NumPy
                arrays (or plain numbers, at some cost in speed).
            spacing: Distance between neighbouring segments in pixels
                (default 40, at least 10).
            color: Segment colour (default: the next palette colour).
            width: Line width in pixels (default 1.5).
            opacity: Opacity in ``[0, 1]``.
            style: A reusable :class:`~pygridplot.LineStyle`. The other
                keyword arguments override its fields.
            label: A name for this plot in the legend (see :meth:`legend`).

        Raises:
            ValueError: If ``spacing`` is not valid, or ``style`` is dashed
                (the segments are too short for a pattern).
        """
        self._sync_rect(surface)
        line = with_overrides(
            style or _DEFAULT_SLOPE, color=color, width=width, opacity=opacity
        )
        if line.dash != "solid":
            msg = f"slope fields are drawn solid, got a style with dash={line.dash!r}"
            raise ValueError(msg)
        color_ = self._colors.resolve(line.color)
        self._add_entry(label, line, color_)
        gap = float(spacing) if spacing is not None else DEFAULT_SPACING
        strokes = self._cached(
            "slope_field",
            (line, color_, gap),
            [slope],
            lambda fs: slope_field_strokes(
                self._view, fs[0], line, color_, spacing=gap
            ),
            arity=2,
        )
        self._draw(surface, *strokes)

    def sequence(
        self,
        surface: pygame.Surface,
        terms: Function | ArrayLike,
        start: int = 0,
        *,
        baseline: float = 0.0,
        color: ColorLike | None = None,
        width: float | None = None,
        radius: float | None = None,
        marker: MarkerShape | None = None,
        opacity: float | None = None,
        style: SequenceStyle | None = None,
        label: str | None = None,
    ) -> None:
        """Draw the terms ``aₙ`` of a sequence, each a stem from the x axis to a marker.

        The terms are a finite sequence (a list or array, whose first term
        has the index ``start``) or an infinite one (a function of ``n``,
        drawn for every visible ``n >= start``)::

            grid.sequence(screen, [3, 1, 4, 1, 5, 9, 2, 6])             # a₀ ... a₇
            grid.sequence(screen, lambda n: (-1) ** n / n, start=1)     # every n ≥ 1
            grid.sequence(screen, pgp.fourier(np.sign, 15).amplitudes) # a spectrum
            grid.sequence(screen, [1, 1 / 2, 1 / 3], width=0)          # markers alone

        A function receives a float array of whole numbers (so NumPy
        arithmetic cannot overflow), or a Python ``int`` if it only works on
        one index at a time (so ``math.factorial`` and list indexing work),
        and may fail or return NaN or infinity for some ``n``: those terms
        are left out. Where several terms fall in one pixel column, their
        stems are drawn as one, which looks the same and costs one stem per
        pixel. Where an infinite sequence's terms are closer than a quarter
        of a pixel apart (zoomed far out), only the multiples of 3, 9, 27,
        ... are drawn, the others fading out as you zoom out, so that the
        work stays bounded; powers of 3 keep odd and even terms alike, so an
        alternating sequence stays two-sided. A finite sequence is always
        drawn in full. The stems start at ``baseline`` (0, the x axis, by
        default; a limit, say, to show the terms closing in on it); on a log
        y axis a baseline of 0 is at the bottom edge.

        Args:
            surface: The surface to draw on.
            terms: The terms: a sequence or 1-D array of numbers (NaN for a
                term not to draw), or a function of ``n``.
            start: The index of the first term (a whole number).
            baseline: Where the stems start: a y value (default 0).
            color: Colour of the stems and markers (default: the next
                palette colour).
            width: Stem thickness in pixels (default 2; 0 for markers alone).
            radius: Marker size in pixels (default 3.5; 0 for stems alone).
            marker: ``"circle"`` (default), ``"square"``, ``"diamond"``,
                ``"cross"`` or ``"plus"``.
            opacity: Opacity in ``[0, 1]``.
            style: A reusable :class:`~pygridplot.SequenceStyle`. The other
                keyword arguments override its fields.
            label: A name for this plot in the legend (see :meth:`legend`).

        Raises:
            ValueError: If ``start`` is not a whole number, ``baseline`` is not
                a finite number, or ``terms`` is neither a function nor a 1-D
                sequence of real numbers.
        """
        self._sync_rect(surface)
        stems = with_overrides(
            style or _DEFAULT_SEQUENCE,
            color=color,
            width=width,
            radius=radius,
            marker=marker,
            opacity=opacity,
        )
        color_ = self._colors.resolve(stems.color)
        self._add_entry(label, stems, color_)
        first = check_start(start)
        base = check_baseline(baseline)
        if callable(terms):
            primitives = self._cached(
                "sequence",
                (stems, color_, first, base),
                [integer_arguments(terms)],
                lambda fs: sequence_primitives(
                    self._view, fs[0], first, stems, color_, baseline=base
                ),
            )
        else:
            data = as_terms(terms)  # a copy: the caller may change it in place
            primitives = self._cached(
                "sequence",
                (stems, color_, first, base, data),
                [],
                lambda _: sequence_primitives(
                    self._view, data, first, stems, color_, baseline=base
                ),
            )
        self._draw(surface, *primitives)

    def points(
        self,
        surface: pygame.Surface,
        points: ArrayLike,
        *,
        color: ColorLike | Sequence[ColorLike] | ArrayLike | None = None,
        radius: float | ArrayLike | None = None,
        marker: MarkerShape | None = None,
        edge_color: ColorLike | None = None,
        edge_width: float | None = None,
        opacity: float | None = None,
        x_error: float | ArrayLike | None = None,
        y_error: float | ArrayLike | None = None,
        error_width: float | None = None,
        error_cap: float | None = None,
        style: PointStyle | None = None,
        label: str | None = None,
    ) -> None:
        """Draw points as markers that keep the same size in pixels at every zoom.

        Points are data as well as marks: each can have its own colour and
        size (a scatter plot coloured or sized by a value), and error bars::

            grid.points(screen, data)                              # a scatter plot
            grid.points(screen, data, y_error=0.2)                 # ± 0.2 each
            grid.points(screen, data, y_error=errors, x_error=dx)  # one each
            grid.points(screen, data, color=colors, radius=sizes)  # per point

        Error bars are in world units (they grow with the zoom), in the
        points' colour, under the markers; with a colour per point they are
        in the theme's label colour. ``radius=0`` draws the error bars alone
        (on top of bars, say).

        Args:
            surface: The surface to draw on.
            points: A single point ``(x, y)``, a sequence of points, or an
                ``(N, 2)`` array. Points with a NaN coordinate are skipped.
            color: Marker colour (default: the next palette colour), or a
                colour for each point: a sequence of colours, or an
                ``(N, 3)`` or ``(N, 4)`` array of values in ``[0, 255]``.
            radius: Marker size in pixels (default 4.5; ``0`` for none), or
                a size for each point (NaN for none).
            marker: ``"circle"`` (default), ``"square"``, ``"diamond"``,
                ``"cross"`` or ``"plus"``.
            edge_color: Colour of an outline around each marker.
            edge_width: Outline thickness in pixels (default 1.5).
            opacity: Opacity in ``[0, 1]``.
            x_error: Error bars across x: one length for every point, one
                per point, or two rows (left lengths, then right). NaN for
                no bar.
            y_error: Error bars across y: likewise (below, then above).
            error_width: Thickness of the error bars in pixels (default 1.5).
            error_cap: Width of the caps on the error bars in pixels
                (default 8; 0 for none).
            style: A reusable :class:`~pygridplot.PointStyle`. The other
                keyword arguments override its fields.
            label: A name for this plot in the legend (see :meth:`legend`).

        Raises:
            ValueError: If there is not one colour, size or error for each
                point, or a size or error is negative.
        """
        self._sync_rect(surface)
        data = as_points(points).copy()  # the caller may change it in place
        count = data.shape[0]
        colors = per_point_colors(color, count)
        radii = per_point_radii(radius, count)
        errors_x = as_errors(x_error, count, "x_error")
        errors_y = as_errors(y_error, count, "y_error")
        point = with_overrides(
            style or _DEFAULT_POINT,
            color=None if colors is not None else color,
            radius=None if radii is not None else radius,
            marker=marker,
            edge_color=edge_color,
            edge_width=edge_width,
            opacity=opacity,
            error_width=error_width,
            error_cap=error_cap,
        )
        if colors is not None and len(colors) > 0:
            first = colors[0]
            color_ = tuple(first.tolist()) if isinstance(first, np.ndarray) else first
            bar_color = self._theme.label_color
        else:
            color_ = self._colors.resolve(point.color)
            bar_color = color_
        has_errors = errors_x is not None or errors_y is not None
        self._add_entry(label, point, color_, errors=has_errors)
        bars, markers = self._cached(
            "points",
            (point, color_, data, colors, radii, errors_x, errors_y, bar_color),
            [],
            lambda _: (
                error_bar_stroke(
                    self._view,
                    data,
                    point,
                    bar_color,
                    x_error=errors_x,
                    y_error=errors_y,
                )
                if has_errors
                else None,
                point_markers(
                    self._view, data, point, color_, colors=colors, radii=radii
                ),
            ),
        )
        self._draw(surface, bars, markers)

    def polyline(
        self,
        surface: pygame.Surface,
        points: ArrayLike,
        *,
        color: ColorLike | None = None,
        width: float | None = None,
        opacity: float | None = None,
        dash: Dash | None = None,
        closed: bool = False,
        step: Step | None = None,
        style: LineStyle | None = None,
        label: str | None = None,
    ) -> None:
        """Draw straight lines through a sequence of world-space vertices.

        With ``step=``, the line goes in steps between the vertices instead,
        horizontally and vertically, as for a count, a sampled-and-held
        signal or an empirical distribution function::

            grid.polyline(screen, data)               # straight lines
            grid.polyline(screen, data, step="post")  # each value until the next x

        Args:
            surface: The surface to draw on.
            points: The vertices, in order: a sequence of points or an
                ``(N, 2)`` array. A vertex with a NaN coordinate breaks the
                line.
            color: Line colour (default: the next palette colour).
            width: Line width in pixels (default 2.5).
            opacity: Line opacity in ``[0, 1]``.
            dash: ``"dashed"``, ``"dotted"``, ``"solid"`` (default), or the
                lengths of the dashes and gaps in pixels, alternately, as
                ``(12, 6)``. The dashes stay put while panning.
            closed: Connect the last vertex back to the first.
            step: ``None`` for straight lines (default), or steps:
                ``"post"`` (each value holds from its x until the next),
                ``"pre"`` (from the previous x up to its own) or ``"mid"``
                (the steps half way between the xs).
            style: A reusable :class:`~pygridplot.LineStyle`. The other
                keyword arguments override its fields.
            label: A name for this plot in the legend (see :meth:`legend`).

        Raises:
            ValueError: If ``step`` is unknown, or given with ``closed``.
        """
        self._sync_rect(surface)
        line = with_overrides(
            style or _DEFAULT_LINE,
            color=color,
            width=width,
            opacity=opacity,
            dash=dash,
        )
        color_ = self._colors.resolve(line.color)
        self._add_entry(label, line, color_)
        stairs = check_step(step)
        if stairs is not None and closed:
            msg = "a polyline cannot be both closed and in steps"
            raise ValueError(msg)
        data = as_points(points).copy()  # the caller may change it in place
        stroke = self._cached(
            "polyline",
            (line, color_, data, closed, stairs),
            [],
            lambda _: polyline_stroke(
                self._view, data, line, color_, closed=closed, step=stairs
            ),
        )
        self._draw(surface, stroke)

    def polygon(
        self,
        surface: pygame.Surface,
        points: ArrayLike,
        *,
        color: ColorLike | None = None,
        width: float | None = None,
        opacity: float | None = None,
        fill: bool | None = None,
        fill_color: ColorLike | None = None,
        fill_opacity: float | None = None,
        fill_rule: FillRule | None = None,
        dash: Dash | None = None,
        style: PolygonStyle | None = None,
        label: str | None = None,
    ) -> None:
        """Draw a polygon: a translucent fill with an outline.

        The polygon's vertices are in world units, so it grows and shrinks
        with the zoom. Its outline width is in pixels.

        Args:
            surface: The surface to draw on.
            points: The vertices of the ring, in order (it closes itself).
            color: Outline colour, also used for the fill unless
                ``fill_color`` is given (default: the next palette colour).
            width: Outline width in pixels (default 2.5; ``0`` for none).
            opacity: Outline opacity in ``[0, 1]``.
            fill: Whether to fill the polygon (default ``True``).
            fill_color: Fill colour (default: the outline colour).
            fill_opacity: Fill opacity in ``[0, 1]`` (default 0.25).
            fill_rule: ``"nonzero"`` (default) or ``"evenodd"``, for polygons
                whose edges cross.
            dash: How the outline is drawn: ``"dashed"``, ``"dotted"``,
                ``"solid"`` (default), or dash and gap lengths in pixels (as
                for :meth:`plot`).
            style: A reusable :class:`~pygridplot.PolygonStyle`. The other
                keyword arguments override its fields.
            label: A name for this plot in the legend (see :meth:`legend`).
        """
        self._sync_rect(surface)
        shape = with_overrides(
            style or _DEFAULT_POLYGON,
            color=color,
            width=width,
            opacity=opacity,
            fill=fill,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            fill_rule=fill_rule,
            dash=dash,
        )
        color_ = self._colors.resolve(shape.color)
        self._add_entry(label, shape, color_)
        data = as_points(points).copy()  # the caller may change it in place
        area, outline = self._cached(
            "polygon",
            (shape, color_, data),
            [],
            lambda _: polygon_shapes(self._view, data, shape, color_),
        )
        self._draw(surface, area, outline)

    def bars(
        self,
        surface: pygame.Surface,
        data: ArrayLike,
        start: float = 0,
        *,
        width: float | None = None,
        baseline: float | ArrayLike = 0.0,
        horizontal: bool = False,
        color: ColorLike | None = None,
        opacity: float | None = None,
        edge_width: float | None = None,
        style: BarStyle | None = None,
        label: str | None = None,
    ) -> None:
        """Draw bars standing on the x axis (a bar chart), or along it.

        ``data`` is the bars' heights, standing at ``start``, ``start + 1``,
        ... (at 0, 1, 2, ... by default), or their ``(x, top)`` pairs::

            grid.bars(screen, [3, 1, 4, 1, 5])                     # at x = 0 ... 4
            grid.bars(screen, np.bincount(rolls), start=0)         # counts of outcomes
            grid.bars(screen, [(2019, 3.1), (2021, 2.4)], width=1.5)
            grid.bars(screen, b, baseline=a)       # stacked: b's bars on a's
            grid.bars(screen, [2, 5, 3], horizontal=True)          # to x = 2, 5, 3

        Bars are in world units, so they grow with the zoom; by default each
        is 0.8 of the smallest gap between neighbouring bars, so bars never
        touch (give ``width`` to choose; bars as wide as their spacing
        touch). A bar runs from its baseline to its top; a top below the
        baseline hangs down. One baseline per bar stacks bars (each series
        standing on the tops of the one before: pass ``a + b`` with
        ``baseline=a``) or floats them (from a start to an end).
        ``horizontal=True`` lays the bars along x: ``data`` is then their
        ends (x values, or ``(y, end)`` pairs) and the baseline an x. On a
        log axis across the bars a baseline of 0 is the plane's edge; on a
        log axis along them, ``width`` is in decades, so bars look alike.

        Args:
            surface: The surface to draw on.
            data: The heights (a sequence of numbers), or ``(x, top)`` pairs
                (a sequence of pairs, or an ``(N, 2)`` array). A NaN leaves a
                bar out.
            start: The position of the first bar, when ``data`` is heights.
            width: The bars' width in x units (y units when horizontal;
                default: 0.8 of the smallest gap between bars).
            baseline: Where the bars stand: a y value (an x when horizontal),
                default 0, or one per bar.
            horizontal: Lay the bars along x, standing on a vertical
                baseline, rather than upright.
            color: Colour of the bars (default: the next palette colour).
            opacity: Opacity of their fill in ``[0, 1]`` (default 0.5).
            edge_width: Width of their outline in pixels (default 1.5; 0
                for none).
            style: A reusable :class:`~pygridplot.BarStyle`. The other
                keyword arguments override its fields.
            label: A name for this plot in the legend (see :meth:`legend`).

        Raises:
            ValueError: If ``data`` is not heights or pairs, ``start`` or
                ``width`` is not a finite number (and a width positive), or
                ``baseline`` is not a number or one per bar.
        """
        self._sync_rect(surface)
        look = with_overrides(
            style or _DEFAULT_BAR, color=color, opacity=opacity, edge_width=edge_width
        )
        table = as_bars(data, start)  # a copy: the caller may change it in place
        size = check_width(width)
        base = as_baselines(baseline, table.shape[0])
        flat = bool(horizontal)
        color_ = self._colors.resolve(look.color)
        self._add_entry(label, look, color_)
        area, outline = self._cached(
            "bars",
            (look, color_, table, size, base, flat),
            [],
            lambda _: bar_shapes(
                self._view,
                table,
                look,
                color_,
                width=size,
                baseline=base,
                horizontal=flat,
            ),
        )
        self._draw(surface, area, outline)

    def histogram(
        self,
        surface: pygame.Surface,
        samples: ArrayLike,
        *,
        width: float | None = None,
        edges: ArrayLike | None = None,
        density: bool = False,
        cumulative: bool = False,
        color: ColorLike | None = None,
        opacity: float | None = None,
        edge_width: float | None = None,
        style: BarStyle | None = None,
        label: str | None = None,
    ) -> None:
        """Draw a histogram of samples: how many fall in each bin.

        ``samples`` are raw values, such as measurements or the results of a
        simulation; the plane bins and counts them::

            samples = rng.normal(size=1000)
            grid.histogram(screen, samples)                  # automatic bins
            grid.histogram(screen, samples, width=0.25)      # [0, 0.25), ...
            grid.histogram(screen, samples, density=True)    # area 1
            grid.plot(screen, lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi))

        The bins are anchored: a bin of width ``w`` is always ``[k w, (k + 1)
        w)``, wherever the samples are, so the bars stay put while samples
        are added frame by frame, and panning and zooming never change them.
        Without a width, it is chosen from the samples (the Freedman-Diaconis
        rule) and rounded to 1, 2 or 5 times a power of ten, so the edges
        fall on round numbers. ``edges`` gives the bins' edges instead, as
        in NumPy (the last bin includes its right edge).

        With ``density=True``, each bar is as high as the density the
        samples estimate there (its count over the number of samples and
        the bin's width), so the bars have area 1 and a probability density
        drawn with :meth:`plot` lies over them. With ``cumulative=True``,
        each bar is as high as all the samples up to its right edge (the
        fraction of them, with ``density``), a stepped estimate of the
        distribution function. Neighbouring bars are filled
        as one shape with one outline, and the lines between them fade out
        as bins narrow below a few pixels. On a log x axis the bins are
        equal on screen: ``width`` is in decades.

        Args:
            surface: The surface to draw on.
            samples: The samples (a sequence or array of numbers). NaN and
                infinite ones are left out (on a log x axis, ones not
                positive too).
            width: The bins' width in x units (default: automatic).
            edges: The bins' edges instead: at least two numbers, increasing.
            density: Draw densities rather than counts (with ``cumulative``,
                fractions of the samples, rising to 1).
            cumulative: Draw each bin as high as all the samples up to its
                right edge: the counts (or fractions) add up along x.
            color: Colour of the bars (default: the next palette colour).
            opacity: Opacity of their fill in ``[0, 1]`` (default 0.5).
            edge_width: Width of their outline in pixels (default 1.5; 0
                for none).
            style: A reusable :class:`~pygridplot.BarStyle`. The other
                keyword arguments override its fields.
            label: A name for this plot in the legend (see :meth:`legend`).

        Raises:
            ValueError: If the samples are not numbers, ``width`` is not
                positive, ``edges`` are not increasing, or both are given.
        """
        self._sync_rect(surface)
        look = with_overrides(
            style or _DEFAULT_BAR, color=color, opacity=opacity, edge_width=edge_width
        )
        values = as_samples(samples)  # a copy: the caller may change it in place
        size = check_width(width)
        bounds = check_edges(edges)
        if size is not None and bounds is not None:
            msg = "give the bins' width or their edges, not both"
            raise ValueError(msg)
        color_ = self._colors.resolve(look.color)
        self._add_entry(label, look, color_)
        dense, rising = bool(density), bool(cumulative)
        area, outline, lines = self._cached(
            "histogram",
            (look, color_, values, size, bounds, dense, rising),
            [],
            lambda _: histogram_shapes(
                self._view,
                values,
                look,
                color_,
                width=size,
                edges=bounds,
                density=dense,
                cumulative=rising,
            ),
        )
        self._draw(surface, area, lines, outline)

    def box_plot(
        self,
        surface: pygame.Surface,
        samples: ArrayLike,
        start: float = 0,
        *,
        positions: ArrayLike | None = None,
        width: float | None = None,
        whisker: float = 1.5,
        horizontal: bool = False,
        color: ColorLike | None = None,
        opacity: float | None = None,
        edge_width: float | None = None,
        style: BarStyle | None = None,
        label: str | None = None,
    ) -> None:
        """Draw box plots: the quartiles, median and spread of groups of samples.

        ``samples`` is one group of samples, or several (a list of groups,
        which may differ in size, or a 2-D array, one group per row), drawn
        side by side at ``start``, ``start + 1``, ... or at ``positions``::

            grid.box_plot(screen, rng.normal(size=200))            # one box at 0
            grid.box_plot(screen, [before, after], start=1)        # at 1 and 2
            grid.box_plot(screen, groups, horizontal=True)         # along x

        Each box spans the first to the third quartile, with the median as a
        line across it; whiskers reach the furthest samples within
        ``whisker`` interquartile ranges of the box (Tukey's 1.5 by
        default), and samples beyond them are drawn as points. On a log axis
        the statistics are those of the logarithms, as the axis shows them.

        Args:
            surface: The surface to draw on.
            samples: One group of samples, or a list of groups. NaN and
                infinite samples are left out.
            start: The position of the first group, without ``positions``.
            positions: Each group's position (x, or y when horizontal).
            width: The boxes' width in x units (default: half the smallest
                gap between groups).
            whisker: How far whiskers reach beyond the box, in interquartile
                ranges (``inf`` for every sample).
            horizontal: Lay the boxes along x.
            color: Colour of the boxes (default: the next palette colour).
            opacity: Opacity of their fill in ``[0, 1]`` (default 0.5).
            edge_width: Width of their lines in pixels (default 1.5).
            style: A reusable :class:`~pygridplot.BarStyle`. The other
                keyword arguments override its fields.
            label: A name for this plot in the legend (see :meth:`legend`).

        Raises:
            ValueError: If the samples are not numbers, the positions not
                one per group, or ``width`` or ``whisker`` not valid.
        """
        self._sync_rect(surface)
        look = with_overrides(
            style or _DEFAULT_BAR, color=color, opacity=opacity, edge_width=edge_width
        )
        groups = as_groups(samples)  # copies: the caller may change them in place
        places = check_positions(positions, start, len(groups))
        size = check_width(width)
        reach = check_whisker(whisker)
        flat = bool(horizontal)
        color_ = self._colors.resolve(look.color)
        self._add_entry(label, look, color_)
        shapes = self._cached(
            "box_plot",
            (look, color_, tuple(groups), places, size, reach, flat),
            [],
            lambda _: box_plot_primitives(
                self._view,
                groups,
                places,
                look,
                color_,
                width=size,
                whisker=reach,
                horizontal=flat,
            ),
        )
        self._draw(surface, *shapes)

    def violin_plot(
        self,
        surface: pygame.Surface,
        samples: ArrayLike,
        start: float = 0,
        *,
        positions: ArrayLike | None = None,
        width: float | None = None,
        bandwidth: float | None = None,
        horizontal: bool = False,
        color: ColorLike | None = None,
        opacity: float | None = None,
        edge_width: float | None = None,
        style: BarStyle | None = None,
        label: str | None = None,
    ) -> None:
        """Draw violin plots: the density of groups of samples, side by side.

        ``samples`` is one group or several, as for :meth:`box_plot`::

            grid.violin_plot(screen, [rng.normal(size=500), rng.gamma(2, size=500)])
            grid.box_plot(screen, groups, width=0.1)   # a narrow box inside each

        Each violin is the density of its samples, estimated with a Gaussian
        kernel (its bandwidth by Silverman's rule unless given), mirrored
        either side of the group's position, ``width`` wide where the density
        is greatest, from the smallest sample to the largest, with the median
        as a line across. Densities are of the logarithms on a log axis.

        Args:
            surface: The surface to draw on.
            samples: One group of samples, or a list of groups. NaN and
                infinite samples are left out; a group needs two different
                values to have a shape.
            start: The position of the first group, without ``positions``.
            positions: Each group's position (x, or y when horizontal).
            width: The violins' greatest width in x units (default: 0.8 of
                the smallest gap between groups).
            bandwidth: The kernel's standard deviation, in value units
                (default: Silverman's rule).
            horizontal: Lay the violins along x.
            color: Colour of the violins (default: the next palette colour).
            opacity: Opacity of their fill in ``[0, 1]`` (default 0.5).
            edge_width: Width of their outline in pixels (default 1.5).
            style: A reusable :class:`~pygridplot.BarStyle`. The other
                keyword arguments override its fields.
            label: A name for this plot in the legend (see :meth:`legend`).

        Raises:
            ValueError: If the samples are not numbers, the positions not
                one per group, or ``width`` or ``bandwidth`` not positive.
        """
        self._sync_rect(surface)
        look = with_overrides(
            style or _DEFAULT_BAR, color=color, opacity=opacity, edge_width=edge_width
        )
        groups = as_groups(samples)
        places = check_positions(positions, start, len(groups))
        size = check_width(width)
        kernel = check_bandwidth(bandwidth)
        flat = bool(horizontal)
        color_ = self._colors.resolve(look.color)
        self._add_entry(label, look, color_)
        shapes = self._cached(
            "violin_plot",
            (look, color_, tuple(groups), places, size, kernel, flat),
            [],
            lambda _: violin_primitives(
                self._view,
                groups,
                places,
                look,
                color_,
                width=size,
                bandwidth=kernel,
                horizontal=flat,
            ),
        )
        self._draw(surface, *shapes)

    # ------------------------------------------------------------------
    # Text and the legend
    # ------------------------------------------------------------------
    def text(
        self,
        surface: pygame.Surface,
        text: str,
        position: tuple[float, float],
        *,
        anchor: Anchor = "center",
        gap: float = 0.0,
        color: ColorLike | None = None,
        size: int | None = None,
        opacity: float | None = None,
        halo: bool | None = None,
        style: TextStyle | None = None,
    ) -> None:
        r"""Write text at a point of the plane, which moves with the point.

        The text keeps its size in pixels while the point follows panning
        and zooming, like a marker. ``anchor`` is the point of the text's box
        placed at ``position``, and ``gap`` moves the text that many pixels
        further from it, on the anchored sides::

            grid.text(screen, "A", (1, 2))                          # centred
            grid.text(screen, "max", (np.pi / 2, 1), anchor="bottom", gap=8)
            grid.text(screen, "y = sin x", (2, 1), anchor="left", color=RED)

        ``anchor="bottom"`` puts the text above the point, ``"left"`` to its
        right (the point is at the text's left), ``"top-right"`` below and
        to its left, and so on. ``"\n"`` starts a new line; lines are aligned
        to the anchor's side. The text is drawn with the theme's label font,
        in its label colour unless given one (text never takes palette
        colours), and with the label halo, which keeps it readable over
        lines and curves.

        Args:
            surface: The surface to draw on.
            text: The text.
            position: The world point ``(x, y)`` the text is anchored to.
                Nothing is drawn if it is NaN (or not positive on a log
                axis).
            anchor: ``"center"`` (default), ``"left"``, ``"right"``,
                ``"top"``, ``"bottom"``, ``"top-left"``, ``"top-right"``,
                ``"bottom-left"`` or ``"bottom-right"``.
            gap: Pixels between the point and the text (default 0).
            color: Text colour (default: the theme's label colour).
            size: Font size in points (default: the theme's label size).
            opacity: Opacity in ``[0, 1]``.
            halo: Whether to outline the glyphs in the background colour
                (default ``True``).
            style: A reusable :class:`~pygridplot.TextStyle`. The other
                keyword arguments override its fields.

        Raises:
            TypeError: If ``text`` is not a string.
            ValueError: If ``position`` is not one point, or ``anchor`` or
                ``gap`` is invalid.
        """
        self._sync_rect(surface)
        look = with_overrides(
            style or _DEFAULT_TEXT, color=color, size=size, opacity=opacity, halo=halo
        )
        words = _checked_text(text)
        point = as_points(position)
        if point.shape[0] != 1:
            msg = f"position must be one point (x, y), got {point.shape[0]} points"
            raise ValueError(msg)
        where = check_anchor(anchor)
        space = check_gap(gap)
        theme = self._theme
        halo_color = theme.resolved_halo_color if look.halo else None
        items = text_items(
            self._view,
            words,
            (float(point[0, 0]), float(point[0, 1])),
            look,
            look.color if look.color is not None else theme.label_color,
            font=FontSpec(theme.label_font, look.size or theme.label_size),
            halo=None if halo_color is None else (halo_color, theme.label_halo_width),
            measurer=self._renderer,
            anchor=where,
            gap=space,
        )
        if items:
            self._draw_texts(surface, items)

    def legend(self, surface: pygame.Surface, *, corner: Corner = "top-right") -> None:
        """Draw a legend naming the plots drawn with a ``label=`` in this frame.

        Every plotting method takes a ``label=``; the legend lists the
        labelled calls since the frame began (:meth:`draw`), in order,
        each beside a small sample of the plot in its style and colour::

            grid.plot(screen, np.sin, label="sin x")
            grid.plot(screen, np.cos, label="cos x")
            grid.fill_between(screen, np.sin, np.cos, label="between")
            grid.legend(screen)

        Call it after the plots, so it is on top of them. It is drawn in a
        panel in a corner of the plane, like the sliders' panels: presses on
        it never pan the plane, and a legend in the same corner as sliders
        sits beyond their panel. A call repeated exactly (the same label,
        style and colour) is listed once, so a plot drawn in pieces can
        label every piece. With no labelled calls, nothing is drawn.

        Args:
            surface: The surface to draw on.
            corner: ``"top-right"`` (default), ``"top-left"``,
                ``"bottom-left"`` or ``"bottom-right"``.

        Raises:
            ValueError: If the corner is unknown.
        """
        self._sync_rect(surface)
        if corner not in CORNERS:
            msg = f"corner must be one of {CORNERS}, got {corner!r}"
            raise ValueError(msg)
        theme = self._theme
        entries = tuple(self._legend_entries)
        panel = layout_legend(
            entries,
            corner,
            self.rect,
            theme,
            self._renderer,
            occupied=self._occupied(corner, legend=False),
        )
        self._legend_panel = panel
        if panel is None:
            return
        self._legend_layout = (self._frame, corner, panel.rect)
        # The panel and each swatch are drawn from last frame's primitives
        # when nothing about them changed, so the renderer reuses their
        # coverage: a legend at rest costs only blending, and a label that
        # changes (showing a value, say) only its text.
        previous = self._legend_drawings
        drawn: dict[object, tuple[object, Any]] = {}

        def reuse(
            name: object,
            key: tuple[object, ...],
            make: Callable[[], Sequence[Fill | Stroke | Markers]],
        ) -> Sequence[Fill | Stroke | Markers]:
            old = previous.get(name)
            shapes = old[1] if old is not None and old[0] == key else make()
            drawn[name] = (key, shapes)
            return shapes

        background = panel_color(theme)
        parts = list(
            reuse(
                "panel",
                (panel.rect, theme),
                lambda: background_primitives(panel.rect, theme),
            )
        )
        for index, row in enumerate(panel.rows):
            entry, left, middle = row.entry, row.swatch_left, row.center_y
            key = (entry.style, entry.color, entry.errors, left, middle, background)
            make = partial(swatch_primitives, entry, left, middle, background)
            parts.extend(reuse(index, key, make))
        self._legend_drawings = drawn
        self._draw(surface, *parts)
        self._draw_texts(surface, legend_texts(panel, theme))

    def color_bar(
        self,
        surface: pygame.Surface,
        colors: ColorMapLike,
        value_range: tuple[float, float],
        *,
        log: bool = False,
        title: str | None = None,
        corner: Corner = "top-right",
    ) -> None:
        """Draw a colour bar: which colour is which value.

        The key to points coloured with :func:`pygridplot.color_map`: give it
        the same colour map, range and scale::

            colors = pgp.color_map(speed, "plasma", (0, 5))
            grid.points(screen, positions, color=colors)
            grid.color_bar(screen, "plasma", (0, 5), title="speed")

        The map is drawn upright, low at the bottom, with ticks at round
        values (multiples of 1, 2 or 5 times a power of ten, or powers of ten
        on a log scale) and a title above. It is a panel in a corner, like
        the legend: it stays put while you pan and zoom, presses on it never
        pan the plane, and in a corner with sliders or the legend it goes
        beyond them.

        Args:
            surface: The surface to draw on.
            colors: The colour map, as for :func:`~pygridplot.color_map`.
            value_range: ``(low, high)``: the values at the map's ends.
            log: Whether the values are on a log scale.
            title: A title above the bar (``None`` for none).
            corner: ``"top-right"`` (default), ``"top-left"``,
                ``"bottom-left"`` or ``"bottom-right"``.

        Raises:
            ValueError: If the map, range or corner is not valid.
        """
        self._sync_rect(surface)
        if corner not in CORNERS:
            msg = f"corner must be one of {CORNERS}, got {corner!r}"
            raise ValueError(msg)
        stops = resolve_stops(colors)
        bounds = check_range(value_range, log=log)
        if bounds is None:
            msg = "a colour bar needs a range, (low, high)"
            raise ValueError(msg)
        words = None if title is None else _checked_text(title)
        theme = self._theme
        panel = layout_color_bar(
            bounds[0],
            bounds[1],
            log=bool(log),
            title=words,
            corner=corner,
            rect=self.rect,
            theme=theme,
            measurer=self._renderer,
            occupied=self._occupied(corner, legend=True),
        )
        self._color_bar_panel = panel
        if panel is None:
            return
        # The same colour bar as last frame is drawn from the same
        # primitives, so the renderer reuses their coverage and its image.
        key = (stops, bounds, bool(log), words, panel.rect, theme)
        old = self._color_bar_drawing
        if old is not None and same_inputs(old[0], key):
            shapes, texts = old[1]
        else:
            shapes, texts = color_bar_primitives(panel, stops, theme)
            self._color_bar_drawing = (key, (shapes, texts))
        self._draw(surface, *shapes)
        self._draw_texts(surface, texts)

    def _add_entry(
        self,
        label: object,
        style: SwatchStyle,
        color: ColorLike,
        *,
        errors: bool = False,
    ) -> None:
        """Note a plotting call's label for the legend.

        Args:
            label: The call's ``label=`` (``None`` for none).
            style: Its resolved style.
            color: Its resolved colour.
            errors: Whether it drew error bars (points only).

        Raises:
            TypeError: If the label is not a string.
            ValueError: If it has more than one line.
        """
        name = check_label(label)
        if name is None:
            return
        entry = LegendEntry(name, style, color, errors)
        if entry not in self._legend_entries:
            self._legend_entries.append(entry)

    def _controls_panel_in(self, corner: Corner) -> ScreenRect | None:
        """Return the sliders' panel in a corner, if drawn in this frame or the last.

        Args:
            corner: The corner.

        Returns:
            Its rectangle, or ``None``.
        """
        frame, panels = self._panel_layout
        if frame == 0 and self._frame <= 1:
            # The first frame, before the sliders are drawn: expect them all.
            panels = layout_panels(
                self._controls, self.rect, self._theme, self._renderer
            )
        elif self._frame - frame > 1:
            return None  # the sliders are no longer drawn
        for panel in panels:
            if panel.rows and panel.rows[0].control.corner == corner:
                return panel.rect
        return None

    def _occupied(self, corner: Corner, *, legend: bool) -> ScreenRect | None:
        """Return the area of the panels in a corner that a new panel goes beyond.

        Args:
            corner: The corner.
            legend: Whether to count the legend too (a colour bar goes beyond
                it), if it was drawn there in this frame or the last.

        Returns:
            The panels' bounding rectangle, or ``None`` if there are none.
        """
        rects = []
        sliders = self._controls_panel_in(corner)
        if sliders is not None:
            rects.append(sliders)
        if legend and self._legend_layout is not None:
            frame, where, rect = self._legend_layout
            if where == corner and self._frame - frame <= 1:
                rects.append(rect)
        if not rects:
            return None
        left = min(r.left for r in rects)
        top = min(r.top for r in rects)
        right = max(r.right for r in rects)
        bottom = max(r.bottom for r in rects)
        return ScreenRect(left, top, right - left, bottom - top)

    def _overlays(self) -> tuple[Panel, ...]:
        """Return the panels on top of the plane: presses there never reach it.

        Returns:
            The sliders' panels, the legend's and the colour bar's, as last
            drawn.
        """
        extra = [
            Panel(panel.rect, ())
            for panel in (self._legend_panel, self._color_bar_panel)
            if panel is not None
        ]
        return (*self._panels, *extra) if extra else self._panels

    # ------------------------------------------------------------------
    # Draggable points
    # ------------------------------------------------------------------
    def draggable(
        self,
        position: tuple[float, float] | None = None,
        *,
        lock: Lock | None = None,
        curve: CurveSpec | None = None,
        t: float | None = None,
        x_range: Range | None = None,
        y_range: Range | None = None,
        color: ColorLike | None = None,
        style: PointStyle | None = None,
        coordinates: Coordinates = "hover",
        enabled: bool = True,
    ) -> DraggablePoint:
        """Create a point that the user can drag with the mouse.

        Create the point once, before the main loop. :meth:`handle_event`
        then drags it, your code reads its :attr:`~DraggablePoint.x` and
        :attr:`~DraggablePoint.y` every frame, and :meth:`draw_controls`
        draws it::

            a = grid.draggable((1, 2))                       # free
            s = grid.draggable((-3, 0.5), lock="x", y_range=(0, 1))  # a slider
            c = grid.draggable(curve=(np.cos, np.sin, (0, 2 * np.pi)))

        Args:
            position: Where the point starts (needed for free and locked
                points). A point on a curve starts at the point of the curve
                nearest to it, or at ``t`` if no position is given.
            lock: ``"x"`` keeps x fixed so the point slides vertically,
                ``"y"`` keeps y fixed so it slides horizontally; ``None``
                (the default) lets it move freely.
            curve: ``(x, y, t_range)``: keep the point on the parametric
                curve ``(x(t), y(t))``, written as for :meth:`parametric`.
                Dragging follows the curve from where the point is, jumping
                to another part of the curve only if it is clearly closer to
                the cursor.
            t: Starting parameter of a point on a curve (default: the start
                of ``t_range``).
            x_range: Keep x within ``(low, high)`` (free and locked points).
            y_range: Keep y within ``(low, high)`` (free and locked points).
            color: Fill colour. The default gives each draggable point its own
                palette colour, in the order the points were created.
            style: A :class:`~pygridplot.PointStyle`. The default is a larger
                marker than plain points, outlined in the background colour.
            coordinates: Show the point's coordinates ``"hover"`` (while
                hovered or dragged, the default), ``"always"`` or ``"never"``.
            enabled: Whether the point can be dragged.

        Returns:
            The point, registered with this plane until its
            :meth:`~DraggablePoint.remove` is called.

        Raises:
            ValueError: If the options conflict (``lock`` with ``curve``,
                ranges with ``curve``), a free or locked point has no
                position, or a range or position is invalid.
        """
        # The plane owns its points, not the other way round: a point only
        # holds a weak reference to it, so a plane can be freed while the
        # program still holds its points.
        forget = weakref.WeakMethod(self._forget_draggable)

        def on_remove(removed: DraggablePoint) -> None:
            method = forget()
            if method is not None:
                method(removed)

        point = DraggablePoint(
            self._view,
            position,
            lock=lock,
            curve=curve,
            t=t,
            x_range=x_range,
            y_range=y_range,
            style=style,
            color=color,
            coordinates=coordinates,
            enabled=enabled,
            on_remove=on_remove,
        )
        self._draggables.append(point)
        self._draggable_colors[point] = self._draggables_created
        self._draggables_created += 1
        return point

    @overload
    def slider(  # an int step: an int value
        self,
        label: str,
        value_range: tuple[int, int],
        *,
        value: int | None = None,
        step: int,
        log: bool = False,
        color: ColorLike | None = None,
        corner: Corner = "bottom-left",
        enabled: bool = True,
        play: PlayMode | None = None,
        duration: float = DEFAULT_DURATION,
    ) -> Slider[int]: ...
    @overload
    def slider(
        self,
        label: str,
        value_range: tuple[float, float],
        *,
        value: float | None = None,
        step: float | None = None,
        log: bool = False,
        color: ColorLike | None = None,
        corner: Corner = "bottom-left",
        enabled: bool = True,
        play: PlayMode | None = None,
        duration: float = DEFAULT_DURATION,
    ) -> Slider[float]: ...
    def slider(
        self,
        label: str,
        value_range: tuple[float, float],
        *,
        value: float | None = None,
        step: float | None = None,
        log: bool = False,
        color: ColorLike | None = None,
        corner: Corner = "bottom-left",
        enabled: bool = True,
        play: PlayMode | None = None,
        duration: float = DEFAULT_DURATION,
    ) -> Slider[Any]:
        """Create a slider: a value the user sets by dragging a handle along a track.

        Create the slider once, before the main loop. It is drawn in a
        panel in a corner of the plane, with its label and value, and stays
        put while the plane is panned and zoomed. :meth:`handle_event` drags
        it, your code reads its :attr:`~pygridplot.Slider.value` every frame,
        and :meth:`draw_controls` draws it::

            a = grid.slider("a", (0, 5), value=1)              # any value
            n = grid.slider("n", (1, 50), value=10, step=1)    # whole numbers
            r = grid.slider("r", (0.01, 100), value=1, log=True)

            grid.plot(screen, lambda x: a.value * np.sin(x))
            grid.plot(screen, pgp.taylor(np.sin, 0, n.value))
            grid.draw_controls(screen)

        Pressing the track moves the handle there; the mouse wheel over a
        slider moves it by a step. Sliders (and checkboxes) in the same
        corner are stacked in the order they are drawn.

        With ``play=``, the slider gets a play button, as in Desmos: pressed,
        the handle sweeps along the track by itself, taking ``duration``
        seconds a sweep, until pressed again. Set
        :attr:`~pygridplot.Slider.playing` to start or stop it from code::

            t = grid.slider("t", (0, 2 * np.pi), play="loop", duration=4)
            t.playing = True                                    # from the start

        Args:
            label: The name shown before the value (``"a"`` shows
                ``a = 1.5``; ``""`` shows the value alone).
            value_range: ``(low, high)``, the values at the ends of the track.
            value: The starting value (default: ``low``), kept within the
                range and on the step.
            step: The spacing of the allowed values, from ``low``. An ``int``
                (``step=1``) makes an integer slider, whose value is an
                ``int``; ``None`` (the default) allows any value, rounded to
                what one pixel of the track can set.
            log: Make the track geometric, for values spanning several
                powers of ten (``low`` must be positive).
            color: The handle's colour. The default gives each slider its own
                palette colour, in the order the sliders were created.
            corner: ``"bottom-left"`` (the default), ``"bottom-right"``,
                ``"top-left"`` or ``"top-right"``: where the slider's panel is.
            enabled: Whether the slider can be dragged and its play button
                pressed (a disabled slider is drawn faded).
            play: Give the slider a play button, and say what playing does at
                the end of the track: ``"loop"`` starts again from the low
                end, ``"bounce"`` turns round (back and forth), ``"once"``
                stops. ``None`` (the default) for no play button.
            duration: How many seconds playing takes to sweep the whole
                track (default 5).

        Returns:
            The slider, registered with this plane until its
            :meth:`~pygridplot.Slider.remove` is called.

        Raises:
            TypeError: If the label is not a string, or a number is not a
                number.
            ValueError: If the range is not two finite numbers in increasing
                order (positive for a log slider), the step is not positive
                or wider than the range, an integer step has ends that are
                not whole numbers, the corner or play mode is unknown, or the
                duration is not positive.
        """
        slider: Slider[Any] = Slider(
            label,
            value_range,
            value=value,
            step=step,
            log=log,
            color=color,
            corner=corner,
            enabled=enabled,
            play=play,
            duration=duration,
            on_remove=self._removal(),
        )
        self._register(slider)
        return slider

    def checkbox(
        self,
        label: str,
        *,
        value: bool = False,
        color: ColorLike | None = None,
        corner: Corner = "bottom-left",
        enabled: bool = True,
    ) -> Checkbox:
        """Create a checkbox: a switch the user turns on and off with a click.

        Create the checkbox once, before the main loop. It is drawn in a
        panel in a corner of the plane, with sliders in the same corner, and
        stays put while the plane is panned and zoomed. A click on the box or
        its label ticks or unticks it, your code reads its
        :attr:`~pygridplot.Checkbox.value` every frame, and
        :meth:`draw_controls` draws it::

            show = grid.checkbox("show the derivative")

            grid.plot(screen, f)
            if show.value:
                grid.plot(screen, pgp.derivative(f))
            grid.draw_controls(screen)

        Args:
            label: The text beside the box.
            value: Whether it starts ticked (default ``False``).
            color: The colour of the ticked box. The default gives each
                control (slider or checkbox) its own palette colour, in the
                order they were created.
            corner: ``"bottom-left"`` (the default), ``"bottom-right"``,
                ``"top-left"`` or ``"top-right"``: where the checkbox's panel
                is.
            enabled: Whether it can be clicked (a disabled checkbox is drawn
                faded).

        Returns:
            The checkbox, registered with this plane until its
            :meth:`~pygridplot.Checkbox.remove` is called.

        Raises:
            TypeError: If the label is not a string, or the value not a bool.
            ValueError: If the corner is unknown.
        """
        box = Checkbox(
            label,
            value=value,
            color=color,
            corner=corner,
            enabled=enabled,
            on_remove=self._removal(),
        )
        self._register(box)
        return box

    def _removal(self) -> Callable[[Any], None]:
        """Return what a control calls when removed (holding the plane weakly)."""
        forget = weakref.WeakMethod(self._forget_control)

        def on_remove(removed: Control) -> None:
            method = forget()
            if method is not None:
                method(removed)

        return on_remove

    def _register(self, control: Control) -> None:
        """Add a control to the plane, with the next palette colour."""
        self._controls.append(control)
        self._control_colors[control] = self._controls_created
        self._controls_created += 1

    @property
    def sliders(self) -> tuple[Slider[Any], ...]:
        """The plane's sliders, in the order they were created."""
        return tuple(c for c in self._controls if isinstance(c, Slider))

    @property
    def checkboxes(self) -> tuple[Checkbox, ...]:
        """The plane's checkboxes, in the order they were created."""
        return tuple(c for c in self._controls if isinstance(c, Checkbox))

    @property
    def draggables(self) -> tuple[DraggablePoint, ...]:
        """The plane's draggable points, in the order they were created."""
        return tuple(self._draggables)

    @property
    def dragged(self) -> DraggablePoint | None:
        """The draggable point being dragged, if any."""
        return self._drag.dragged

    def draw_controls(
        self,
        surface: pygame.Surface,
        *items: DraggablePoint | Slider[Any] | Checkbox,
    ) -> None:
        """Draw draggable points, sliders and checkboxes, with their highlights.

        Call it after the plots, so the points sit on top of the curves they
        control and the panels of sliders and checkboxes on top of
        everything. A hovered or dragged point gets a translucent ring, and
        its coordinates are shown as set by its
        :attr:`~DraggablePoint.coordinates`; sliders and checkboxes are drawn
        in panels in the corners of the plane (see :meth:`slider` and
        :meth:`checkbox`).

        Only sliders and checkboxes drawn here can be used until the next
        frame, so leaving one out (``grid.draw_controls(screen, a, n)``)
        hides it and switches it off.

        Args:
            surface: The surface to draw on.
            *items: The points, sliders and checkboxes to draw (default: all
                of this plane's points, then all its sliders and checkboxes,
                in creation order).

        Raises:
            ValueError: If a point, slider or checkbox does not belong to
                this plane.
        """
        self._sync_rect(surface)
        if items:
            points = [item for item in items if isinstance(item, DraggablePoint)]
            controls: list[Control] = [
                item for item in items if not isinstance(item, DraggablePoint)
            ]
        else:
            points, controls = list(self._draggables), list(self._controls)
        for point in points:
            if point not in self._draggable_colors:
                msg = f"{point!r} is not a draggable point of this plane"
                raise ValueError(msg)
        for control in controls:
            if control not in self._control_colors:
                kind = "slider" if isinstance(control, Slider) else "checkbox"
                msg = f"{control!r} is not a {kind} of this plane"
                raise ValueError(msg)
        theme = self._theme
        markers: list[Markers] = []
        for point in points:
            color = point.color
            if color is None:
                index = self._draggable_colors[point]
                color = theme.palette[index % len(theme.palette)]
            markers.extend(point_primitives(point, color, theme.background))
        self._draw(surface, *markers)
        labels = [coordinate_item(point, theme, self._renderer) for point in points]
        texts = [label for label in labels if label is not None]
        if texts:
            self._draw_texts(surface, texts)
        if controls:
            self._draw_panels(surface, controls)

    def _draw_panels(self, surface: pygame.Surface, controls: list[Control]) -> None:
        """Lay out and draw sliders and checkboxes, keeping the layout for events.

        Args:
            surface: The surface to draw on.
            controls: The sliders and checkboxes, in order.
        """
        theme = self._theme
        self._panels = layout_panels(controls, self.rect, theme, self._renderer)
        self._panel_layout = (self._frame, self._panels)
        # The previous frame's primitives are drawn again when nothing about
        # them changed, so the renderer reuses their coverage: a panel at rest
        # costs only blending, and a drag only redraws the dragged slider (a
        # playing slider redraws itself every frame).
        previous = self._panel_drawings
        drawn: dict[object, tuple[object, Any]] = {}
        for panel in self._panels:
            key: tuple[object, ...] = (panel.rect, theme)
            name: object = panel.rect
            old = previous.get(name)
            background = old[1] if old is not None and old[0] == key else None
            if background is None:
                background = (background_primitives(panel.rect, theme), [])
            drawn[name] = (key, background)
            shapes: list[Fill | Stroke | Markers] = list(background[0])
            texts: list[TextItem] = []
            for row in panel.rows:
                control = row.control
                color = control.color
                if color is None:
                    index = self._control_colors[control]
                    color = theme.palette[index % len(theme.palette)]
                key = (row_key(row, color), theme)
                old = previous.get(control)
                if old is not None and old[0] == key:
                    parts = old[1]
                else:
                    row_shapes, text = row_primitives(row, theme, color)
                    parts = (row_shapes, [text])
                drawn[control] = (key, parts)
                shapes.extend(parts[0])
                texts.extend(parts[1])
            self._draw(surface, *shapes)
            self._draw_texts(surface, texts)
        self._panel_drawings = drawn

    def _draw_texts(self, surface: pygame.Surface, texts: Sequence[TextItem]) -> None:
        """Draw labels, clipped to the plane.

        Args:
            surface: The surface to draw on.
            texts: The labels.
        """
        self._drawing.append(("texts", tuple(texts)))
        with self._renderer.clipped(surface, self.rect) as visible:
            if visible is None:
                return
            for text in texts:
                self._renderer.draw_text(surface, text)

    def _forget_control(self, control: Control) -> None:
        """Unregister a removed slider or checkbox (see :meth:`Slider.remove`).

        Args:
            control: The slider or checkbox.
        """
        if control in self._control_colors:
            self._controls.remove(control)
            del self._control_colors[control]
            self._panels = tuple(
                Panel(
                    panel.rect,
                    tuple(row for row in panel.rows if row.control is not control),
                )
                for panel in self._panels
            )
            self._panel_input.forget(control)

    def _forget_draggable(self, point: DraggablePoint) -> None:
        """Unregister a removed point (see :meth:`DraggablePoint.remove`).

        Args:
            point: The point.
        """
        if point in self._draggable_colors:
            self._draggables.remove(point)
            del self._draggable_colors[point]
            self._drag.forget(point)

    # ------------------------------------------------------------------
    # Coordinates and view control
    # ------------------------------------------------------------------
    @overload
    def to_world(self, px: float, py: float) -> tuple[float, float]: ...
    @overload
    def to_world(
        self, px: ArrayLike, py: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def to_world(self, px: Any, py: Any) -> Any:
        """Convert screen coordinates (for example the mouse) to world coordinates.

        Args:
            px: Screen x coordinate(s) in pixels.
            py: Screen y coordinate(s) in pixels.

        Returns:
            The world coordinates ``(x, y)``.
        """
        return self._view.to_world(px, py)

    @overload
    def to_screen(self, x: float, y: float) -> tuple[float, float]: ...
    @overload
    def to_screen(
        self, x: ArrayLike, y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def to_screen(self, x: Any, y: Any) -> Any:
        """Convert world coordinates to continuous screen coordinates.

        Args:
            x: World x coordinate(s).
            y: World y coordinate(s).

        Returns:
            The screen coordinates ``(p_x, p_y)`` in pixels.
        """
        return self._view.to_screen(x, y)

    def set_view(
        self,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
    ) -> None:
        """Fit the view to show the given world ranges.

        With equal aspect on, the tighter range decides the scale and the
        other axis shows extra space. On a log axis the range must be
        positive. An automatic plane that has not been drawn yet fits again
        once it knows its real size (unless the view is changed before).

        Args:
            x_range: ``(x_min, x_max)`` to show across the plane's width.
            y_range: ``(y_min, y_max)`` to show across the plane's height.

        Raises:
            ValueError: If a range is empty, reversed, not finite, or not
                positive on a log axis.
        """
        self._controller.cancel()
        self._framing.fit((x_range, y_range), home=False)

    def reset_view(self) -> None:
        """Return to the home view (initially the view the plane was created with)."""
        self._controller.cancel()
        self._framing.reset()

    def set_home(self, state: ViewState | None = None) -> None:
        """Change the view that :meth:`reset_view` (and the Home key) returns to.

        Args:
            state: The new home view, or ``None`` to use the current view.
        """
        self._framing.home = state if state is not None else self._view.state()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def save(self, path: str | os.PathLike[str], *, scale: float = 1.0) -> None:
        """Save the plane, as drawn in this frame, to a picture file.

        Call it after drawing the frame (the grid, the plots, the controls)
        and before the next :meth:`draw`. The picture is the plane alone,
        its rectangle and nothing outside it, drawn again at ``scale`` times
        as many pixels along each side: ``scale=2`` gives a picture twice as
        wide and twice as tall, with every line, point and letter twice the
        size, so it looks the same, only sharper::

            grid.draw(screen)
            grid.plot(screen, np.sin)
            grid.save("sine.png")                # as on screen
            grid.save("sine@3x.png", scale=3)    # for a slide or for print

        Only what the plane drew is in the picture: draw your own things
        (captions, buttons) with pygame, and save the whole window with
        :func:`pygame.image.save` instead. Curves are sampled for the
        screen, to a quarter of a pixel, so they stay smooth up to about
        ``scale=4``.

        Args:
            path: Where to save, with the file type as its extension:
                ``.png`` (lossless; a plane without a background, see
                :attr:`GridTheme.background`, is transparent), ``.jpg``,
                ``.bmp`` or ``.tga``. Missing folders are created.
            scale: How many times as many pixels along each side (``0.5``
                for a smaller picture). The picture can be at most 16384
                pixels on a side.

        Raises:
            ValueError: If nothing was drawn on the plane in this frame, the
                extension is not one of those above, or the scale is not
                positive or makes too large a picture.
            TypeError: If the scale is not a number.
        """
        target = Path(path)
        suffix = target.suffix.lower()
        if suffix not in _PICTURE_TYPES:
            msg = f"save a picture as {', '.join(_PICTURE_TYPES)}, got {str(path)!r}"
            raise ValueError(msg)
        picture = self._snapshot(scale, opaque=suffix != ".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(picture, str(target))

    def _snapshot(self, scale: float, *, opaque: bool) -> pygame.Surface:
        """Draw this frame again onto a surface of its own (see :meth:`save`).

        Args:
            scale: How many times as many pixels along each side.
            opaque: Fill it first with the plane's colour (its background,
                or the label halo's colour where it has none), for videos and
                picture types without transparency.

        Returns:
            The picture.

        Raises:
            ValueError: If nothing was drawn in this frame, or the picture
                would be too large.
        """
        factor = check_scale(scale)
        if not self._drawing:
            msg = "nothing was drawn on the plane in this frame: draw it, then save"
            raise ValueError(msg)
        if self.rect.is_empty:
            msg = "the plane has no area to save"
            raise ValueError(msg)
        if self._replayer is None:
            self._replayer = Replayer()
        # Transparent only where the plane drew no background of its own.
        has_scene = any(kind == "scene" for kind, _ in self._drawing)
        transparent = not opaque and (self._theme.background is None or not has_scene)
        return self._replayer.render(
            self._drawing,
            self.rect,
            self._renderer,
            factor,
            None if transparent else panel_color(self._theme),
            transparent=transparent,
        )

    def _use_clock(self, clock: Callable[[], float] | None) -> None:
        """Make playing sliders follow another clock (a recorder's), or pygame's.

        The time between frames starts again from the new clock, so nothing
        jumps when switching.

        Args:
            clock: Returns the time in milliseconds, or ``None`` for pygame's.
        """
        self._play_clock = clock
        self._last_tick = None if clock is None else clock()

    def __repr__(self) -> str:
        return f"Grid(view={self._view!r})"

    def _cached(
        self,
        kind: str,
        key: tuple[object, ...],
        functions: list[Any],
        make: Callable[[list[Any]], Any],
        *,
        arity: int = 1,
        outputs: int = 1,
    ) -> Any:
        """Build a plotting call's primitives, or reuse the previous frame's.

        Args:
            kind: The kind of plot.
            key: Everything besides the view and the functions that the
                result depends on.
            functions: The call's functions (or constants in their place).
            make: Builds the primitives from the functions.
            arity: The functions' number of arguments.
            outputs: The number of values they return (2 for vector fields).

        Returns:
            The primitives (see :class:`~pygridplot.core.plot.memo.PlotCache`).
        """
        return self._plots.build(
            kind,
            (self._view.fingerprint, key),
            functions,
            make,
            arity=arity,
            outputs=outputs,
        )

    def _grid_scene(
        self,
        titles: tuple[str | None, str | None],
        names: tuple[Categories | None, Categories | None] = (None, None),
    ) -> GridScene:
        """Build the grid, or reuse the last one if nothing it depends on changed.

        The grid is a pure function of the view, the theme and the titles,
        except in the ``"snap"`` transition, whose hysteresis remembers
        earlier frames; that one is always built.

        Args:
            titles: The x and y axis' titles.
            names: The x and y axis' category names, or ``None``.

        Returns:
            The grid scene.
        """
        key = (self._view.fingerprint, self._theme, titles, names)
        reusable = (
            self._plots.enabled
            and self._theme.transition == "fade"
            and self._last_scene is not None
            and self._grid_key is not None
            and same_inputs(self._grid_key, key)
        )
        if reusable and self._last_scene is not None:
            return self._last_scene
        scene = build_grid(
            self._view,
            self._theme,
            self._renderer,
            self._grid_state,
            x_title=titles[0],
            y_title=titles[1],
            x_labels=names[0],
            y_labels=names[1],
        )
        if scene.arrows and self._last_scene is not None:
            # An arrowhead that has not moved (the x axis' while panning
            # sideways) is drawn from the same primitive, so the renderer
            # reuses its coverage instead of rasterising it again.
            old = self._last_scene.arrows
            scene = replace(
                scene, arrows=tuple(_same(head, old) for head in scene.arrows)
            )
        self._grid_key = key
        return scene

    def _draw(
        self,
        surface: pygame.Surface,
        *primitives: Stroke | Fill | Markers | Image | None,
    ) -> None:
        """Draw primitives in order, clipped to the plane.

        Every plotting method ends here, so a new kind of plot only needs to
        build primitives (see :mod:`pygridplot.core.plot.builders`) and pass them on.

        Args:
            surface: The surface to draw on.
            *primitives: What to draw, in painter's order; ``None`` entries
                (nothing visible) are skipped.
        """
        shapes = [shape for shape in primitives if shape is not None]
        if not shapes:
            return
        self._drawing.append(("shapes", tuple(shapes)))
        with self._renderer.clipped(surface, self.rect) as visible:
            if visible is None:
                return
            for shape in shapes:
                if isinstance(shape, Stroke):
                    self._renderer.draw_stroke(surface, shape)
                elif isinstance(shape, Fill):
                    self._renderer.draw_fill(surface, shape)
                elif isinstance(shape, Image):
                    self._renderer.draw_image(surface, shape)
                else:
                    self._renderer.draw_markers(surface, shape)

    def _sync_rect(self, surface: pygame.Surface) -> None:
        """Make an automatic plane fill ``surface`` (no-op for explicit rects).

        Args:
            surface: The surface being drawn on.
        """
        if self._auto_rect:
            self._view.rect = surface.get_rect()
            self._framing.settle()

    def _set_log(self, axis: Literal["x", "y"], log: bool) -> None:
        """Switch one axis between linear and logarithmic.

        Args:
            axis: ``"x"`` or ``"y"``.
            log: Whether the axis should be logarithmic.
        """
        self._controller.cancel()
        if self._framing.set_log(axis, log):
            self._grid_state.reset()


def _display_rect() -> ScreenRect:
    """Return the rectangle of the current display surface, if one exists.

    Returns:
        The display's rectangle, or an empty rectangle before a window exists.
    """
    if pygame.display.get_init():
        display = pygame.display.get_surface()
        if display is not None:
            return ScreenRect.from_any(display.get_rect())
    return ScreenRect(0, 0, 0, 0)


def _same(fill: Fill, previous: Sequence[Fill]) -> Fill:
    """Return an equal fill drawn before, if there is one, else ``fill``.

    Args:
        fill: A fill.
        previous: Fills drawn in the last frame.

    Returns:
        The fill to draw.
    """
    for old in previous:
        if (
            old.color == fill.color
            and len(old.rings) == len(fill.rings)
            and all(
                np.array_equal(a, b) for a, b in zip(old.rings, fill.rings, strict=True)
            )
        ):
            return old
    return fill


def _checked_text(text: object) -> str:
    """Check the text given to :meth:`Grid.text`.

    Args:
        text: The text given.

    Returns:
        The text.

    Raises:
        TypeError: If it is not a string.
    """
    if not isinstance(text, str):
        msg = f"text must be a string, got {type(text).__name__}"
        raise TypeError(msg)
    return text


def _title(title: object, name: str) -> str | None:
    """Check an axis title.

    Args:
        title: The title given.
        name: The argument's name, for the error message.

    Returns:
        The title, or ``None`` for none (an empty title is none).

    Raises:
        TypeError: If the title is neither a string nor ``None``.
    """
    if title is None or title == "":
        return None
    if not isinstance(title, str):
        msg = f"{name} must be a string or None, got {type(title).__name__}"
        raise TypeError(msg)
    return title


def _pair(span: Any) -> tuple[float, float] | Any:
    """Return a range as two floats for a cache key (or as given, if it is not one)."""
    try:
        return (float(span[0]), float(span[1]))
    except (TypeError, ValueError, IndexError):
        return span


def _numbers(value: Any) -> Any:
    """Return a number or sequence of numbers as a float array for a cache key."""
    try:
        return np.atleast_1d(np.array(value, dtype=np.float64))
    except (TypeError, ValueError):
        return value
