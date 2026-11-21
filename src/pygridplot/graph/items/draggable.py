"""Draggable points: points the user moves, kept by the graph and saved."""

from __future__ import annotations

import math
import weakref
from typing import Any, ClassVar, TypeAlias

from pygridplot.core.expr import (
    Binding,
    PointBinding,
)
from pygridplot.core.handles.constraints import Lock, Range, check_bounds
from pygridplot.core.handles.point import (
    COORDINATE_MODES,
    DEFAULT_DRAGGABLE_STYLE,
    Coordinates,
    DraggablePoint,
)
from pygridplot.core.plot.style import (
    PointStyle,
)
from pygridplot.graph.items.base import (
    Expression,
    Formula,
    Maker,
    PointLikeItem,
    check_name,
    component,
    finite,
    finite_range,
    two,
)

CurveTarget: TypeAlias = (
    "str | tuple[Expression | float, Expression | float, tuple[float, float]]"
)
"""What a draggable point follows: a curve item's name, or ``(x, y, t_range)``."""


class DraggableItem(PointLikeItem):
    """A point the user drags with the mouse, like :meth:`Plane.draggable`.

    The item holds the point's position, which dragging changes, setting
    :attr:`position` changes, and saving keeps. Other items use it as
    ``P.x`` and ``P.y`` (and follow it as it moves). The point can be free,
    locked to an axis, or kept on a curve: another item's curve (a plot, a
    parametric or a polar curve, by name) or one given here as
    ``(x(t), y(t), t_range)``.

    Args:
        position: Where it starts (needed unless it follows a curve).
        name: The point's name (optional).
        lock: ``"x"`` keeps x fixed, ``"y"`` keeps y fixed.
        curve: A curve item's name, or ``(x(t), y(t), t_range)``.
        t: Starting parameter on the curve.
        x_range: Bounds for x (free and locked points).
        y_range: Bounds for y (free and locked points).
        color: Fill colour (default: the item's palette colour).
        style: A :class:`~pygridplot.PointStyle` (default: the plane's
            style for draggable points).
        coordinates: Show the coordinates ``"hover"``, ``"always"`` or
            ``"never"``.
        enabled: Whether it can be dragged.
        visible: Whether it is drawn (a hidden point cannot be dragged).
        palette: Palette slot (normally chosen by the graph).
    """

    kind: ClassVar[str] = "draggable"
    default_style: ClassVar[Any] = DEFAULT_DRAGGABLE_STYLE

    def __init__(
        self,
        position: tuple[float, float] | None = None,
        *,
        name: str | None = None,
        lock: Lock | None = None,
        curve: CurveTarget | None = None,
        t: float | None = None,
        x_range: Range | None = None,
        y_range: Range | None = None,
        color: Any = None,
        style: PointStyle | None = None,
        coordinates: Coordinates = "hover",
        enabled: bool = True,
        visible: bool = True,
        palette: int | None = None,
    ) -> None:
        super().__init__(
            style=style,
            overrides={"color": color},
            palette=palette,
            name=name,
            visible=visible,
        )
        self._curve_name: str | None = None
        self._curve: tuple[Formula, Formula, tuple[float, float]] | None = None
        self._lock: Lock | None = None
        self._x_range: Range | None = None
        self._y_range: Range | None = None
        self._configure(lock, curve, x_range, y_range)
        if position is None and curve is None:
            msg = "a draggable point needs a position unless it follows a curve"
            raise ValueError(msg)
        self._position: tuple[float, float] | None = (
            None if position is None else _position(position)
        )
        self._t = None if t is None else finite(t, "t")
        self.coordinates = coordinates
        self._enabled = bool(enabled)
        self._points: weakref.WeakSet[DraggablePoint] = weakref.WeakSet()

    # --- Position ----------------------------------------------------------
    @property
    def position(self) -> tuple[float, float] | None:
        """The current position (``None`` before a curve point is first drawn).

        Setting it moves the point; a point on a curve goes to the nearest
        point of the curve when next drawn.
        """
        return self._position

    @position.setter
    def position(self, position: tuple[float, float]) -> None:
        self._position = _position(position)

    @property
    def x(self) -> float:
        """The current x (NaN before a curve point is first drawn)."""
        return math.nan if self._position is None else self._position[0]

    @property
    def y(self) -> float:
        """The current y (NaN before a curve point is first drawn)."""
        return math.nan if self._position is None else self._position[1]

    @property
    def t(self) -> float | None:
        """The parameter on the curve (``None`` for free and locked points).

        Setting it moves a curve point along its curve when next drawn.
        """
        return self._t

    @t.setter
    def t(self, t: float) -> None:
        if self._curve is None and self._curve_name is None:
            msg = "only points on a curve have a parameter"
            raise ValueError(msg)
        self._t = finite(t, "t")

    @property
    def dragging(self) -> bool:
        """Whether the point is being dragged on any plane."""
        return any(point.dragging for point in self._points)

    # --- Constraint -----------------------------------------------------
    @property
    def lock(self) -> Lock | None:
        """``"x"`` or ``"y"`` for a locked point, ``None`` otherwise."""
        return self._lock

    @lock.setter
    def lock(self, lock: Lock | None) -> None:
        self._configure(lock, self.curve, self._x_range, self._y_range)
        self._changed()

    @property
    def curve(self) -> CurveTarget | None:
        """The curve the point follows: a curve item's name, or ``(x, y, t_range)``."""
        if self._curve_name is not None:
            return self._curve_name
        if self._curve is not None:
            x, y, t_range = self._curve
            return (x.source, y.source, t_range)
        return None

    @curve.setter
    def curve(self, curve: CurveTarget | None) -> None:
        self._configure(self._lock, curve, self._x_range, self._y_range)
        if curve is None:
            self._t = None
        self._changed()

    @property
    def x_range(self) -> Range | None:
        """Bounds for x, if any."""
        return self._x_range

    @x_range.setter
    def x_range(self, x_range: Range | None) -> None:
        self._configure(self._lock, self.curve, x_range, self._y_range)
        self._changed()

    @property
    def y_range(self) -> Range | None:
        """Bounds for y, if any."""
        return self._y_range

    @y_range.setter
    def y_range(self, y_range: Range | None) -> None:
        self._configure(self._lock, self.curve, self._x_range, y_range)
        self._changed()

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
    def enabled(self) -> bool:
        """Whether the point can be dragged."""
        return self._enabled

    @enabled.setter
    def enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    # --- For the graph ----------------------------------------------------
    def _configure(
        self,
        lock: object,
        curve: object,
        x_range: object,
        y_range: object,
    ) -> None:
        """Validate and store the constraint (all of it, as its parts interact)."""
        if lock not in (None, "x", "y"):
            msg = f"lock must be None, 'x' or 'y', got {lock!r}"
            raise ValueError(msg)
        if curve is not None and lock is not None:
            msg = "a point on a curve cannot also be locked to an axis"
            raise ValueError(msg)
        if curve is not None and (x_range is not None or y_range is not None):
            msg = "a point on a curve is bounded by the curve's range, not x/y ranges"
            raise ValueError(msg)
        bounds_x = check_bounds("x_range", x_range)  # type: ignore[arg-type]
        bounds_y = check_bounds("y_range", y_range)  # type: ignore[arg-type]
        name: str | None = None
        spec: tuple[Formula, Formula, tuple[float, float]] | None = None
        if isinstance(curve, str):
            name = check_name(curve, "a curve's name")
        elif curve is not None:
            parts = tuple(curve) if isinstance(curve, (tuple, list)) else ()
            if len(parts) != 3:
                msg = f"curve must be a curve's name or (x, y, t_range), got {curve!r}"
                raise ValueError(msg)
            x, y, t_range = parts
            spec = (
                component(x, "x(t)"),
                component(y, "y(t)"),
                finite_range(t_range, "t_range"),
            )
        self._lock = lock  # narrowed by the check above
        self._curve_name, self._curve = name, spec
        self._x_range, self._y_range = bounds_x, bounds_y

    def _settings(self) -> tuple[object, ...]:
        """Return what the plane's point is made with (made again if this changes)."""
        return (
            self._lock,
            self._curve_name,
            self._curve is not None,
            self._x_range,
            self._y_range,
        )

    def references(self) -> set[str]:
        """Return the names of other items this point refers to (its curve's)."""
        names = super().references()
        if self._curve_name is not None:
            names.add(self._curve_name)
        return names

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        if self._curve is None:
            return []
        x, y, _ = self._curve
        return [(x, ("t",)), (y, ("t",))]

    def _runtime(self, make: Maker) -> Any:
        if self._curve is None:
            return None
        x, y, t_range = self._curve
        return (make(x, ("t",)), make(y, ("t",)), t_range)

    def _binding(self, runtime: Any) -> Binding | None:
        return PointBinding(lambda: (self.x, self.y))

    def _set_state(self, position: tuple[float, float], t: float | None) -> None:
        """Record where the plane's point is (by the graph, after dragging)."""
        self._position = position
        if t is not None:
            self._t = t

    def _describe(self) -> str:
        if self._position is None:
            return f"on {self._curve_name or 'a curve'}"
        return f"({self._position[0]:.6g}, {self._position[1]:.6g})"


def _position(value: object) -> tuple[float, float]:
    """Validate a numeric position."""
    x, y = two(value, "position")
    return (finite(x, "x"), finite(y, "y"))
