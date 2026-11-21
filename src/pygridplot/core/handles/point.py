"""The :class:`DraggablePoint`: a point on the plane that the user can drag.

A draggable point is the one object in PyGridPlot that keeps state between
frames: its position, which the user changes with the mouse. It is created
once with :meth:`pygridplot.Plane.draggable`, which registers it with the
plane so that :meth:`~pygridplot.Plane.handle_event` can drag it; your code
reads :attr:`~DraggablePoint.x` and :attr:`~DraggablePoint.y` every frame
and uses them like any other number, and draws the points with
:meth:`~pygridplot.Plane.draw_draggables`.

There are four kinds, set when the point is created:

* **free**: dragged anywhere;
* **x locked** (``lock="x"``): x stays fixed and the point slides vertically;
* **y locked** (``lock="y"``): y stays fixed and the point slides horizontally;
* **on a curve** (``curve=(x, y, t_range)``): the point stays on a parametric
  curve, following it as the cursor moves (see
  :mod:`pygridplot.core.handles.tracking`).

Free and locked points can be kept within ``x_range`` and ``y_range``; a
point on a curve is kept within the curve's ``t_range``. Constraints apply
to dragging and to positions set from code alike.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, TypeAlias

from pygridplot.core.coords import ColorLike
from pygridplot.core.handles.constraints import (
    AxisConstraint,
    CurveConstraint,
    Lock,
    Placement,
    Range,
    check_bounds,
)
from pygridplot.core.plot.builders import CurveComponent
from pygridplot.core.plot.style import PointStyle
from pygridplot.core.view import View

Coordinates: TypeAlias = Literal["hover", "always", "never"]
"""When a draggable point shows its coordinates: while hovered or dragged,
always, or never."""

CurveSpec: TypeAlias = tuple[CurveComponent, CurveComponent, tuple[float, float]]
"""A parametric curve for a point to stay on: ``(x, y, t_range)``, with ``x``
and ``y`` functions of ``t`` (or numbers), as for
:meth:`pygridplot.Plane.parametric`."""

COORDINATE_MODES: tuple[Coordinates, ...] = ("hover", "always", "never")
"""The allowed values of :attr:`DraggablePoint.coordinates`."""

DEFAULT_DRAGGABLE_STYLE = PointStyle(radius=6.0, edge_width=2.0)
"""How draggable points look unless given a style: larger than plain points,
with an outline in the theme's background colour."""


class DraggablePoint:
    """A point that can be dragged with the mouse, freely or along a constraint.

    Create points with :meth:`pygridplot.Plane.draggable` rather than
    directly: the plane then routes mouse events to them and draws them.

    Args:
        view: The view the point lives in (the plane's).
        position: The starting position. For a point on a curve, the point
            starts at the nearest point of the curve (as seen on screen);
            without it, at ``t``.
        lock: ``"x"`` keeps x fixed, ``"y"`` keeps y fixed, ``None`` (the
            default) lets the point move freely.
        curve: ``(x, y, t_range)``: keep the point on this parametric curve.
        t: Starting parameter of a point on a curve (default: the start of
            ``t_range``).
        x_range: Bounds for x, for free and locked points.
        y_range: Bounds for y, for free and locked points.
        style: How the point looks (default :data:`DEFAULT_DRAGGABLE_STYLE`).
        color: Fill colour; ``None`` gives each point a palette colour, in
            the order the points were created.
        coordinates: When to show the coordinates: ``"hover"`` (while hovered
            or dragged), ``"always"`` or ``"never"``.
        enabled: Whether the point can be dragged.
        on_remove: Called by :meth:`remove` (the plane forgets the point).

    Raises:
        ValueError: If the settings conflict or are invalid (for example
            ``lock`` together with ``curve``, or a position outside a log
            axis).
        TypeError: If a curve component is neither callable nor a number.
    """

    def __init__(
        self,
        view: View,
        position: tuple[float, float] | None = None,
        *,
        lock: Lock | None = None,
        curve: CurveSpec | None = None,
        t: float | None = None,
        x_range: Range | None = None,
        y_range: Range | None = None,
        style: PointStyle | None = None,
        color: ColorLike | None = None,
        coordinates: Coordinates = "hover",
        enabled: bool = True,
        on_remove: Callable[[DraggablePoint], None] | None = None,
    ) -> None:
        if curve is not None and lock is not None:
            msg = "a point on a curve cannot also be locked to an axis"
            raise ValueError(msg)
        if curve is not None and (x_range is not None or y_range is not None):
            msg = "a point on a curve is bounded by its t_range, not x/y ranges"
            raise ValueError(msg)
        if curve is None and t is not None:
            msg = "t is only for points on a curve"
            raise ValueError(msg)
        self._view = view
        self._axes = AxisConstraint(
            lock, check_bounds("x_range", x_range), check_bounds("y_range", y_range)
        )
        self._curve: CurveConstraint | None = None
        self._placement = Placement(math.nan, math.nan)
        self._version = 0
        self._seen_version = 0
        self.style = style if style is not None else DEFAULT_DRAGGABLE_STYLE
        """How the point looks (a :class:`~pygridplot.PointStyle`)."""
        self.color = color
        """Fill colour, or ``None`` for the point's palette colour."""
        self.coordinates = coordinates
        self.enabled = enabled
        """Whether the point can be dragged (it is drawn either way)."""
        self._on_remove = on_remove
        self._dragging = False
        self._hovered = False
        self._moved = False
        if curve is not None:
            self._curve = _curve_constraint(curve)
            if position is None:
                start = self._curve.t_range[0] if t is None else t
                self._set(self._curve.at(start))
            else:
                self._set(self._nearest_on_curve(*position))
        else:
            if position is None:
                msg = "free and locked points need a starting position"
                raise ValueError(msg)
            x, y = float(position[0]), float(position[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                msg = f"position must be finite, got {position!r}"
                raise ValueError(msg)
            if (view.log_x and x <= 0) or (view.log_y and y <= 0):
                msg = f"position {position!r} cannot be shown on a log axis"
                raise ValueError(msg)
            self._set(self._axes.place(view, x, y, Placement(x, y)))
        self._seen_version = self._version  # not "moved" in its first frame

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------
    @property
    def x(self) -> float:
        """World x (NaN while a point's curve is undefined at its parameter)."""
        return self._placement.x

    @property
    def y(self) -> float:
        """World y (NaN while a point's curve is undefined at its parameter)."""
        return self._placement.y

    @property
    def position(self) -> tuple[float, float]:
        """The world position ``(x, y)``.

        Setting it moves the point as near as its constraint allows: a free
        point goes there (within its bounds), a locked point too (locks
        restrict only dragging), and a point on a curve goes to the nearest
        point of the curve as seen on screen.
        """
        return (self._placement.x, self._placement.y)

    @position.setter
    def position(self, value: tuple[float, float]) -> None:
        x, y = float(value[0]), float(value[1])
        if self._curve is not None:
            self._set(self._nearest_on_curve(x, y))
        else:
            self._set(self._axes.place(self._view, x, y, self._placement))

    @property
    def t(self) -> float | None:
        """The curve parameter of a point on a curve (``None`` for other points).

        Setting it moves the point to that parameter (clamped to the curve's
        ``t_range``).
        """
        return self._placement.t

    @t.setter
    def t(self, value: float) -> None:
        if self._curve is None:
            msg = "only points on a curve have a parameter"
            raise ValueError(msg)
        self._set(self._curve.at(float(value)))

    # ------------------------------------------------------------------
    # Constraint
    # ------------------------------------------------------------------
    @property
    def lock(self) -> Lock | None:
        """``"x"`` or ``"y"`` for a locked point, ``None`` otherwise."""
        return self._axes.lock

    @property
    def x_range(self) -> Range | None:
        """Bounds for x, if any."""
        return self._axes.x_range

    @property
    def y_range(self) -> Range | None:
        """Bounds for y, if any."""
        return self._axes.y_range

    @property
    def curve(self) -> CurveSpec | None:
        """The curve ``(x, y, t_range)`` of a point on a curve (``None`` otherwise).

        Assign a new curve (every frame, for an animated one) to move the
        point with it: the point keeps its parameter ``t`` (clamped to the
        new range). A free or locked point cannot be given a curve.
        """
        if self._curve is None:
            return None
        x, y = self._curve.components
        return (x, y, self._curve.t_range)

    @curve.setter
    def curve(self, value: CurveSpec) -> None:
        if self._curve is None:
            msg = "only points created on a curve can be given a new curve"
            raise ValueError(msg)
        self._curve = _curve_constraint(value)
        t = self._placement.t
        self._set(self._curve.at(self._curve.t_range[0] if t is None else t))

    # ------------------------------------------------------------------
    # Appearance and state
    # ------------------------------------------------------------------
    @property
    def coordinates(self) -> Coordinates:
        """When the coordinates are shown: ``"hover"``, ``"always"`` or ``"never"``."""
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value: Coordinates) -> None:
        if value not in COORDINATE_MODES:
            msg = f"coordinates must be one of {COORDINATE_MODES}, got {value!r}"
            raise ValueError(msg)
        self._coordinates: Coordinates = value

    @property
    def dragging(self) -> bool:
        """Whether the point is being dragged."""
        return self._dragging

    @property
    def hovered(self) -> bool:
        """Whether the cursor is over the point (and it can be dragged)."""
        return self._hovered

    @property
    def moved(self) -> bool:
        """Whether the point moved between the previous frame and this one.

        Frames start at :meth:`pygridplot.Plane.draw_grid` (or
        :meth:`~pygridplot.Plane.begin_frame`), so the value stays the same
        while you draw; use it to recompute something only when needed.
        """
        return self._moved

    @property
    def visible(self) -> bool:
        """Whether the point has a position that can be drawn."""
        return math.isfinite(self._placement.x) and math.isfinite(self._placement.y)

    @property
    def view(self) -> View:
        """The view the point lives in (its plane's)."""
        return self._view

    @property
    def screen_position(self) -> tuple[float, float]:
        """The point's position on screen, in pixels."""
        return self._view.to_screen(self._placement.x, self._placement.y)

    # ------------------------------------------------------------------
    # Dragging (used by the plane)
    # ------------------------------------------------------------------
    def drag_to(self, px: float, py: float) -> bool:
        """Move the point towards a screen position, as a drag would.

        The plane calls this while the point is dragged; it can also be used
        to drive a point from code with screen coordinates.

        Args:
            px: Screen x of the target.
            py: Screen y of the target.

        Returns:
            Whether the point moved.
        """
        target = (float(px), float(py))
        if self._curve is not None:
            placement = self._curve.drag(self._view, target, self._placement)
            if placement is None:
                return False
        else:
            placement = self._axes.drag(self._view, target, self._placement)
        return self._set(placement)

    def set_interaction(self, *, hovered: bool, dragging: bool) -> None:
        """Record whether the point is hovered and dragged (by the plane).

        Args:
            hovered: Whether the cursor is over the point.
            dragging: Whether the point is being dragged.
        """
        self._hovered = hovered
        self._dragging = dragging

    def begin_frame(self) -> None:
        """Start a new frame: update :attr:`moved` and resample a curve when needed.

        :meth:`pygridplot.Plane.begin_frame` calls this for every point.
        """
        self._moved = self._version != self._seen_version
        self._seen_version = self._version
        if self._curve is not None:
            self._curve.invalidate()

    def remove(self) -> None:
        """Remove the point from its plane: it is no longer dragged or drawn."""
        self._dragging = self._hovered = False
        if self._on_remove is not None:
            callback, self._on_remove = self._on_remove, None
            callback(self)

    def __repr__(self) -> str:
        """Return a short description with the position."""
        kind = (
            "curve"
            if self._curve is not None
            else f"lock={self.lock!r}"
            if self.lock
            else "free"
        )
        return f"DraggablePoint(({self.x:.6g}, {self.y:.6g}), {kind})"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _set(self, placement: Placement) -> bool:
        """Move to a placement, counting the change.

        Args:
            placement: The new position.

        Returns:
            Whether the position changed.
        """
        old = self._placement
        self._placement = placement
        if (
            _same(placement.x, old.x)
            and _same(placement.y, old.y)
            and placement.t == old.t
        ):
            return False
        self._version += 1
        return True

    def _nearest_on_curve(self, x: float, y: float) -> Placement:
        """Return the curve point nearest ``(x, y)``, or the curve's start.

        Args:
            x: World x.
            y: World y.

        Returns:
            The placement.
        """
        assert self._curve is not None
        found = self._curve.nearest(self._view, x, y)
        return found if found is not None else self._curve.at(self._curve.t_range[0])


def _curve_constraint(curve: CurveSpec) -> CurveConstraint:
    """Build a curve constraint from ``(x, y, t_range)``.

    Args:
        curve: The curve.

    Returns:
        The constraint.

    Raises:
        ValueError: If ``curve`` is not a triple.
    """
    try:
        x, y, t_range = curve
    except (TypeError, ValueError):
        msg = f"curve must be (x, y, t_range), got {curve!r}"
        raise ValueError(msg) from None
    return CurveConstraint(x, y, t_range)


def _same(a: float, b: float) -> bool:
    """Return whether two coordinates are equal, counting NaN as equal to NaN.

    Args:
        a: One value.
        b: The other.

    Returns:
        Whether they are the same.
    """
    return a == b or (math.isnan(a) and math.isnan(b))
