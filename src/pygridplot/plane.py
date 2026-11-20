"""The public :class:`Plane` class: an interactive Cartesian plane for pygame.

A plane ties together a :class:`~pygridplot.core.view.View` (what is
visible), a :class:`~pygridplot.core.theme.GridTheme` (how it looks), an
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
        plane.points(screen, [(0, 0), (np.pi, 0)])
        plane.polygon(screen, [(1, 1), (3, 1), (2, 2.5)])
        pygame.display.flip()
        clock.tick(60)

Drawing is *immediate mode*: everything is recomputed from the current view
every frame, so plots follow panning and zooming automatically and animating
them needs no extra work.
"""

from __future__ import annotations

from typing import Any, overload

import numpy as np
import pygame
from numpy.typing import ArrayLike, NDArray

from pygridplot.core.coords import ColorLike, ScreenRect
from pygridplot.core.evaluation import Function
from pygridplot.core.grid import GridScene, GridState, build_grid
from pygridplot.core.plotting import (
    function_stroke,
    point_markers,
    polygon_shapes,
    polyline_stroke,
)
from pygridplot.core.sampling import DEFAULT_SAMPLING, SamplingConfig
from pygridplot.core.style import (
    ColorCycle,
    FillRule,
    LineStyle,
    MarkerShape,
    PointStyle,
    PolygonStyle,
    with_overrides,
)
from pygridplot.core.theme import LIGHT, GridTheme
from pygridplot.core.view import View, ViewLimits, ViewState
from pygridplot.interaction import InteractionConfig, InteractionController
from pygridplot.render.pygame_renderer import PygameRenderer

_DEFAULT_LINE = LineStyle()
_DEFAULT_POINT = PointStyle()
_DEFAULT_POLYGON = PolygonStyle()


class Plane:
    """An interactive Cartesian plane drawn into a rectangle of a pygame surface.

    Args:
        rect: The screen rectangle to draw into. ``None`` (the default) fills
            whatever surface :meth:`draw_grid` is given and follows it when
            the window is resized.
        center: World point initially shown at the middle of the plane.
        scale: Initial pixels per world unit: one number, or an ``(x, y)``
            pair when ``equal_aspect`` is off.
        equal_aspect: Keep both axes at the same scale (circles stay round).
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
        center: tuple[float, float] = (0.0, 0.0),
        scale: float | tuple[float, float] = 50.0,
        equal_aspect: bool = True,
        theme: GridTheme | None = None,
        interactive: bool = True,
        interaction: InteractionConfig | None = None,
        limits: ViewLimits | None = None,
        renderer: PygameRenderer | None = None,
        sampling: SamplingConfig | None = None,
    ) -> None:
        self._auto_rect = rect is None
        initial = _display_rect() if rect is None else ScreenRect.from_any(rect)
        self._view = View(
            initial, center, scale, equal_aspect=equal_aspect, limits=limits
        )
        self._home = self._view.state()
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
        if stroke is not None:
            with self._renderer.clipped(surface, self.rect) as visible:
                if visible is not None:
                    self._renderer.draw_stroke(surface, stroke)

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
        if markers is not None:
            with self._renderer.clipped(surface, self.rect) as visible:
                if visible is not None:
                    self._renderer.draw_markers(surface, markers)

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
        if stroke is not None:
            with self._renderer.clipped(surface, self.rect) as visible:
                if visible is not None:
                    self._renderer.draw_stroke(surface, stroke)

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
        if area is None and outline is None:
            return
        with self._renderer.clipped(surface, self.rect) as visible:
            if visible is None:
                return
            if area is not None:
                self._renderer.draw_fill(surface, area)
            if outline is not None:
                self._renderer.draw_stroke(surface, outline)

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

        Args:
            x_range: ``(x_min, x_max)`` to show across the plane's width.
            y_range: ``(y_min, y_max)`` to show across the plane's height.
        """
        self._controller.cancel()
        self._view.fit(x_range, y_range)

    def reset_view(self) -> None:
        """Return to the home view (initially the view the plane was created with)."""
        self._controller.cancel()
        self._view.restore(self._home)

    def set_home(self, state: ViewState | None = None) -> None:
        """Change the view that :meth:`reset_view` (and the Home key) returns to.

        Args:
            state: The new home view, or ``None`` to use the current view.
        """
        self._home = state if state is not None else self._view.state()

    def __repr__(self) -> str:
        return f"Plane(view={self._view!r})"

    def _sync_rect(self, surface: pygame.Surface) -> None:
        """Make an automatic plane fill ``surface`` (no-op for explicit rects).

        Args:
            surface: The surface being drawn on.
        """
        if self._auto_rect:
            self._view.rect = surface.get_rect()


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
