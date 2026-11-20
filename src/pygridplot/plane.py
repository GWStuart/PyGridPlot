"""The public :class:`Plane` class: an interactive Cartesian plane for pygame.

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
    plane = pgp.Plane()
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            plane.handle_event(event)
        plane.draw_grid(screen)
        plane.plot(screen, np.sin)
        plane.polar(screen, lambda t: 1 + np.cos(t))
        plane.points(screen, [(0, 0), (np.pi, 0)])
        plane.polygon(screen, [(1, 1), (3, 1), (2, 2.5)])
        pygame.display.flip()
        clock.tick(60)

Drawing is *immediate mode*: everything is recomputed from the current view
every frame, so plots follow panning and zooming automatically and animating
them needs no extra work.
"""

from __future__ import annotations

import math
from typing import Any, Literal, overload

import numpy as np
import pygame
from numpy.typing import ArrayLike, NDArray

from pygridplot.core.coords import ColorLike, ScreenRect
from pygridplot.core.framing import ViewFraming, aspect_choice, equal_aspect_for
from pygridplot.core.grid.scene import GridScene, GridState, build_grid
from pygridplot.core.grid.theme import LIGHT, GridTheme
from pygridplot.core.plot.builders import (
    CurveComponent,
    function_stroke,
    parametric_stroke,
    point_markers,
    polar_stroke,
    polygon_shapes,
    polyline_stroke,
)
from pygridplot.core.plot.curves import DEFAULT_CURVE_SAMPLING, CurveSampling
from pygridplot.core.plot.evaluation import Function
from pygridplot.core.plot.sampling import DEFAULT_SAMPLING, SamplingConfig
from pygridplot.core.plot.style import (
    ColorCycle,
    LineStyle,
    PointStyle,
    PolygonStyle,
    with_overrides,
)
from pygridplot.core.primitives import Fill, FillRule, Markers, MarkerShape, Stroke
from pygridplot.core.view import View, ViewLimits, ViewState
from pygridplot.interaction import InteractionConfig, InteractionController
from pygridplot.render.pygame_renderer import PygameRenderer

_DEFAULT_LINE = LineStyle()
_DEFAULT_POINT = PointStyle()
_DEFAULT_POLYGON = PolygonStyle()


class Plane:
    """An interactive Cartesian plane drawn into a rectangle of a pygame surface.

    The initial view can be given like a graphing calculator's window
    settings, ``x_range=(x_min, x_max)`` and ``y_range=(y_min, y_max)``, or as a
    ``center`` and ``scale``. Either axis can be logarithmic.

    Args:
        rect: The screen rectangle to draw into. ``None`` (the default) fills
            whatever surface :meth:`draw_grid` is given and follows it when
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
        sampling: SamplingConfig | None = None,
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
        """Whether :meth:`handle_event` reacts to input at all."""
        self._renderer = renderer if renderer is not None else PygameRenderer()
        self._grid_state = GridState()
        self._controller = InteractionController(
            self._view, interaction, on_reset=self.reset_view
        )
        self._colors = ColorCycle(self._theme.palette)
        self._sampling = sampling if sampling is not None else DEFAULT_SAMPLING
        self._last_grid: GridScene | None = None

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
    def sampling(self) -> SamplingConfig:
        """How :meth:`plot` samples functions (assign to change it)."""
        return self._sampling

    @sampling.setter
    def sampling(self, config: SamplingConfig) -> None:
        self._sampling = config

    @property
    def interaction(self) -> InteractionConfig:
        """Settings of the built-in pan/zoom controls."""
        return self._controller.config

    @interaction.setter
    def interaction(self, config: InteractionConfig) -> None:
        self._controller.config = config

    @property
    def renderer(self) -> PygameRenderer:
        """The renderer the plane draws with."""
        return self._renderer

    @property
    def last_grid(self) -> GridScene | None:
        """The grid drawn by the most recent :meth:`draw_grid` call, if any.

        Useful for debugging overlays (it records the chosen spacings).
        """
        return self._last_grid

    # ------------------------------------------------------------------
    # Per-frame API
    # ------------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Let the plane react to an input event (pan, zoom, keyboard).

        Pass every event, including mouse motion: the plane tracks the cursor
        to zoom about it.

        Args:
            event: A pygame event.

        Returns:
            ``True`` if the plane used the event, so your code can skip
            handling it again.
        """
        if not self.interactive:
            return False
        return self._controller.handle_event(event)

    def update(self) -> bool:
        """Advance eased pan/zoom movement.

        :meth:`draw_grid` calls this automatically, so you only need it if you
        stop drawing the grid but still want eased movement to continue.

        Returns:
            ``True`` if the view changed.
        """
        if not self.interactive:
            return False
        return self._controller.update()

    def begin_frame(self) -> None:
        """Start a new frame for the colour palette.

        The next object drawn without a colour gets the first palette colour
        again. :meth:`draw_grid` calls this automatically. Call it yourself only if
        you draw plots without drawing the grid, so that every object keeps
        the same palette colour from frame to frame.
        """
        self._colors.reset()

    def draw_grid(self, surface: pygame.Surface) -> None:
        """Draw the background, grid lines, axes and tick labels.

        Drawing is clipped to :attr:`rect`, so the plane can share a window
        with other content. Call it once per frame, before drawing anything
        that should appear on top of the grid. It also starts a new frame
        for the colour palette (see :meth:`begin_frame`).

        Args:
            surface: The surface to draw on (usually the display surface).
        """
        self._sync_rect(surface)
        self.update()
        self.begin_frame()
        scene = build_grid(self._view, self._theme, self._renderer, self._grid_state)
        self._renderer.draw_grid(surface, scene)
        self._last_grid = scene

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
        domain: tuple[float, float] | None = None,
        style: LineStyle | None = None,
        sampling: SamplingConfig | None = None,
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
            domain: Only plot where ``domain[0] <= x <= domain[1]``.
            style: A reusable :class:`~pygridplot.LineStyle`. The other
                keyword arguments override its fields.
            sampling: Override the plane's sampling settings for this call.

        Note:
            When creating functions in a loop, bind the loop variable as a
            default argument: ``[lambda x, k=k: x**k for k in range(3)]``.
            Plain ``lambda x: x**k`` would make every function use the last
            ``k``.
        """
        self._sync_rect(surface)
        line = with_overrides(
            style or _DEFAULT_LINE, color=color, width=width, opacity=opacity
        )
        stroke = function_stroke(
            self._view,
            func,
            line,
            self._colors.resolve(line.color),
            domain=domain,
            sampling=sampling if sampling is not None else self._sampling,
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
        style: LineStyle | None = None,
        sampling: CurveSampling | None = None,
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
            style: A reusable :class:`~pygridplot.LineStyle`. The other
                keyword arguments override its fields.
            sampling: How the curve is sampled (see
                :class:`~pygridplot.CurveSampling`; the defaults suit almost
                every curve).

        Raises:
            ValueError: If ``t_range`` is not an increasing pair of numbers.
        """
        self._sync_rect(surface)
        line = with_overrides(
            style or _DEFAULT_LINE, color=color, width=width, opacity=opacity
        )
        stroke = parametric_stroke(
            self._view,
            x,
            y,
            t_range,
            line,
            self._colors.resolve(line.color),
            sampling=sampling if sampling is not None else DEFAULT_CURVE_SAMPLING,
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
        style: LineStyle | None = None,
        sampling: CurveSampling | None = None,
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
            style: A reusable :class:`~pygridplot.LineStyle`. The other
                keyword arguments override its fields.
            sampling: How the curve is sampled (see
                :class:`~pygridplot.CurveSampling`).

        Raises:
            ValueError: If ``theta_range`` is not an increasing pair of numbers.
        """
        self._sync_rect(surface)
        line = with_overrides(
            style or _DEFAULT_LINE, color=color, width=width, opacity=opacity
        )
        stroke = polar_stroke(
            self._view,
            r,
            theta_range,
            line,
            self._colors.resolve(line.color),
            sampling=sampling if sampling is not None else DEFAULT_CURVE_SAMPLING,
        )
        self._draw(surface, stroke)

    def points(
        self,
        surface: pygame.Surface,
        points: ArrayLike,
        *,
        color: ColorLike | None = None,
        radius: float | None = None,
        marker: MarkerShape | None = None,
        edge_color: ColorLike | None = None,
        edge_width: float | None = None,
        opacity: float | None = None,
        style: PointStyle | None = None,
    ) -> None:
        """Draw points as markers that keep the same size in pixels at every zoom.

        Args:
            surface: The surface to draw on.
            points: A single point ``(x, y)``, a sequence of points, or an
                ``(N, 2)`` array. Points with a NaN coordinate are skipped.
            color: Marker colour (default: the next palette colour).
            radius: Marker size in pixels (default 4.5).
            marker: ``"circle"`` (default), ``"square"``, ``"diamond"``,
                ``"cross"`` or ``"plus"``.
            edge_color: Colour of an outline around each marker.
            edge_width: Outline thickness in pixels (default 1.5).
            opacity: Opacity in ``[0, 1]``.
            style: A reusable :class:`~pygridplot.PointStyle`. The other
                keyword arguments override its fields.
        """
        self._sync_rect(surface)
        point = with_overrides(
            style or _DEFAULT_POINT,
            color=color,
            radius=radius,
            marker=marker,
            edge_color=edge_color,
            edge_width=edge_width,
            opacity=opacity,
        )
        markers = point_markers(
            self._view, points, point, self._colors.resolve(point.color)
        )
        self._draw(surface, markers)

    def polyline(
        self,
        surface: pygame.Surface,
        points: ArrayLike,
        *,
        color: ColorLike | None = None,
        width: float | None = None,
        opacity: float | None = None,
        closed: bool = False,
        style: LineStyle | None = None,
    ) -> None:
        """Draw straight lines through a sequence of world-space vertices.

        Args:
            surface: The surface to draw on.
            points: The vertices, in order: a sequence of points or an
                ``(N, 2)`` array. A vertex with a NaN coordinate breaks the
                line.
            color: Line colour (default: the next palette colour).
            width: Line width in pixels (default 2.5).
            opacity: Line opacity in ``[0, 1]``.
            closed: Connect the last vertex back to the first.
            style: A reusable :class:`~pygridplot.LineStyle`. The other
                keyword arguments override its fields.
        """
        self._sync_rect(surface)
        line = with_overrides(
            style or _DEFAULT_LINE, color=color, width=width, opacity=opacity
        )
        stroke = polyline_stroke(
            self._view, points, line, self._colors.resolve(line.color), closed=closed
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
        style: PolygonStyle | None = None,
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
            style: A reusable :class:`~pygridplot.PolygonStyle`. The other
                keyword arguments override its fields.
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
        )
        area, outline = polygon_shapes(
            self._view, points, shape, self._colors.resolve(shape.color)
        )
        self._draw(surface, area, outline)

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

    def __repr__(self) -> str:
        return f"Plane(view={self._view!r})"

    def _draw(
        self, surface: pygame.Surface, *primitives: Stroke | Fill | Markers | None
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
        with self._renderer.clipped(surface, self.rect) as visible:
            if visible is None:
                return
            for shape in shapes:
                if isinstance(shape, Stroke):
                    self._renderer.draw_stroke(surface, shape)
                elif isinstance(shape, Fill):
                    self._renderer.draw_fill(surface, shape)
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
