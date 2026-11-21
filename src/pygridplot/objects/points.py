"""Points: fixed, following other objects, or dragged by the user.

A :class:`Point` is one marker. Its position is

* **numbers**: ``Point((1, 2))``; set ``p.x = 3`` or ``p.position = (3, 4)``
  at any time;
* **derived**: ``Point(lambda: (q.x, f(q.x)))``, or per coordinate (a
  number, a function of nothing, or text without variables:
  ``Point(("a", "2a"))`` in a graph), evaluated whenever it is drawn or
  read, so the point follows what it uses;
* or **dragged**: ``Point((1, 2), draggable=True)``, free, locked to an
  axis, bounded, or kept on a curve (another object, a graph's curve by
  name, or ``(x(t), y(t), t_range)``).

A draggable point is the *model*; on each plane it is drawn on it keeps a
:class:`~pygridplot.DraggablePoint` *handle*, which the plane drags. The
two are kept in step every frame (:func:`_sync`): a drag moves the point,
and setting the point from code moves the handles, on every plane.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

from pygridplot.core.expr import Binding, PointBinding
from pygridplot.core.handles.constraints import Lock, Range, check_bounds
from pygridplot.core.handles.point import (
    COORDINATE_MODES,
    DEFAULT_DRAGGABLE_STYLE,
    Coordinates,
    DraggablePoint,
)
from pygridplot.core.plot.style import PointStyle
from pygridplot.objects.base import MarkerObject
from pygridplot.objects.curves import CurveObject, CurveSpec, curve_start
from pygridplot.objects.values import (
    Expression,
    Formula,
    Maker,
    as_float,
    call_quietly,
    check_name,
    component,
    coordinate,
    finite,
    finite_range,
    getter,
    show,
    two,
)

if TYPE_CHECKING:
    import pygame

    from pygridplot.plane import Plane

CurveTarget: TypeAlias = (
    "str | CurveObject"
    " | tuple[Expression | float, Expression | float, tuple[float, float]]"
)
"""What a draggable point follows: a curve object, a graph's curve by name, or
``(x(t), y(t), t_range)``."""

PositionLike: TypeAlias = "tuple[Any, Any] | Callable[[], tuple[float, float]]"
"""A position: two coordinates (numbers, functions of nothing or text), or a
function of nothing returning both."""


class Point(MarkerObject):
    """A point: fixed, following other objects, or dragged with the mouse.

    Read its position at any time as :attr:`x`, :attr:`y` or
    :attr:`position` (numbers, also for derived and dragged points), and use
    it in other objects: ``pgp.Function(lambda x: a.y * np.sin(x - a.x))``
    follows ``a`` as it moves. In a graph, a named point is ``P.x`` and
    ``P.y`` in expressions.

    Args:
        position: ``(x, y)``: numbers; or, for a point that follows others,
            coordinates that are functions of nothing or text, or one
            function returning ``(x, y)``. A draggable point needs numbers,
            and may leave it out on a curve.
        draggable: Whether the user can drag it.
        lock: ``"x"`` keeps x fixed, ``"y"`` keeps y fixed (draggable only).
        curve: Keep a draggable point on a curve: a :class:`Function`,
            :class:`ParametricCurve` or :class:`PolarCurve` object, a graph's
            curve by name, or ``(x(t), y(t), t_range)``.
        t: Starting parameter on the curve.
        x_range: Bounds for x of a draggable point.
        y_range: Bounds for y of a draggable point.
        name: The point's name (optional).
        color: Fill colour (default: the object's palette colour).
        radius: Marker size in pixels.
        marker: Marker shape.
        edge_color: Outline colour.
        edge_width: Outline thickness in pixels.
        opacity: Opacity.
        style: A :class:`~pygridplot.PointStyle` (default: the plane's style
            for draggable points if created draggable, else markers').
        coordinates: Show a draggable point's coordinates ``"hover"``,
            ``"always"`` or ``"never"``.
        enabled: Whether a draggable point can be dragged (it is drawn either
            way).
        visible: Whether it is drawn (a hidden point cannot be dragged).
        palette: Palette slot (normally chosen by the plane or graph).

    Raises:
        ValueError: For constraints on a point that is not draggable, or
            constraints that conflict (``lock`` with ``curve``).
    """

    kind: ClassVar[str] = "point"

    def __init__(
        self,
        position: PositionLike | None = None,
        *,
        draggable: bool = False,
        lock: Lock | None = None,
        curve: CurveTarget | None = None,
        t: float | None = None,
        x_range: Range | None = None,
        y_range: Range | None = None,
        name: str | None = None,
        color: Any = None,
        radius: float | None = None,
        marker: Any = None,
        edge_color: Any = None,
        edge_width: float | None = None,
        opacity: float | None = None,
        style: PointStyle | None = None,
        coordinates: Coordinates = "hover",
        enabled: bool = True,
        visible: bool = True,
        palette: int | None = None,
    ) -> None:
        super().__init__(
            style=style,
            overrides={
                "color": color,
                "radius": radius,
                "marker": marker,
                "edge_color": edge_color,
                "edge_width": edge_width,
                "opacity": opacity,
            },
            palette=palette,
            name=name,
            visible=visible,
            default=DEFAULT_DRAGGABLE_STYLE if draggable else None,
        )
        self._draggable = bool(draggable)
        if not self._draggable and (
            lock is not None
            or curve is not None
            or t is not None
            or x_range is not None
            or y_range is not None
        ):
            msg = (
                "lock, curve, t and bounds are for draggable points: "
                "pass draggable=True"
            )
            raise ValueError(msg)
        self._curve_name: str | None = None
        self._curve_object: CurveObject | None = None
        self._curve: tuple[Formula, Formula, tuple[float, float]] | None = None
        self._lock: Lock | None = None
        self._x_range: Range | None = None
        self._y_range: Range | None = None
        self._configure(lock, curve, x_range, y_range)
        self._whole: Callable[[], Any] | None = None
        self._coords: tuple[Formula, Formula] = (Formula(math.nan, None),) * 2
        self._placed = position is not None
        if position is None:
            if not (self._draggable and self._on_curve):
                msg = "a point needs a position (a draggable one, unless on a curve)"
                raise ValueError(msg)
        else:
            self._set_position(position)
        self._t = None if t is None else finite(t, "t")
        self.coordinates = coordinates
        self._enabled = bool(enabled)
        self._handles: weakref.WeakKeyDictionary[Plane, _Handle] = (
            weakref.WeakKeyDictionary()
        )

    # --- Position -----------------------------------------------------------
    @property
    def x(self) -> float:
        """The current x (NaN where undefined, or before a curve point is placed)."""
        return self._coordinate(0)

    @x.setter
    def x(self, x: Any) -> None:
        self._set_position((x, self._source_of(1)))

    @property
    def y(self) -> float:
        """The current y (NaN where undefined, or before a curve point is placed)."""
        return self._coordinate(1)

    @y.setter
    def y(self, y: Any) -> None:
        self._set_position((self._source_of(0), y))

    @property
    def position(self) -> tuple[float, float] | None:
        """The current position as numbers (``None`` before a curve point is placed).

        Setting it accepts what the constructor does: numbers, functions or
        text (numbers only for a draggable point, which goes to the nearest
        allowed position when next drawn).
        """
        if not self._placed:
            return None
        return (self.x, self.y)

    @position.setter
    def position(self, position: PositionLike) -> None:
        self._set_position(position)

    @property
    def source(self) -> Any:
        """The position as given: a pair, or a function returning one."""
        if self._whole is not None:
            return self._whole
        if not self._placed:
            return None
        return (self._coords[0].source, self._coords[1].source)

    @property
    def t(self) -> float | None:
        """The parameter on the curve (``None`` unless on a curve).

        Setting it moves the point along its curve when next drawn.
        """
        return self._t

    @t.setter
    def t(self, t: float) -> None:
        if not self._on_curve:
            msg = "only points on a curve have a parameter"
            raise ValueError(msg)
        self._t = finite(t, "t")

    @property
    def dragging(self) -> bool:
        """Whether the point is being dragged, on any plane."""
        return any(handle.point.dragging for handle in self._handles.values())

    # --- Dragging -----------------------------------------------------------
    @property
    def draggable(self) -> bool:
        """Whether the user can drag the point.

        Turning it on needs a numeric position (or a curve); turning it off
        takes its handles off every plane, keeping the position.
        """
        return self._draggable

    @draggable.setter
    def draggable(self, draggable: bool) -> None:
        draggable = bool(draggable)
        if draggable and not self._numeric and not self._on_curve:
            msg = "a draggable point's position must be numbers"
            raise ValueError(msg)
        self._draggable = draggable
        if not draggable:
            self._detach_all()

    @property
    def lock(self) -> Lock | None:
        """``"x"`` or ``"y"`` for a locked point, ``None`` otherwise."""
        return self._lock

    @lock.setter
    def lock(self, lock: Lock | None) -> None:
        self._require_draggable("lock")
        self._configure(lock, self.curve, self._x_range, self._y_range)

    @property
    def curve(self) -> CurveTarget | None:
        """The curve followed: an object, a name, or ``(x, y, t_range)`` as given."""
        if self._curve_object is not None:
            return self._curve_object
        if self._curve_name is not None:
            return self._curve_name
        if self._curve is not None:
            x, y, t_range = self._curve
            return (x.source, y.source, t_range)
        return None

    @curve.setter
    def curve(self, curve: CurveTarget | None) -> None:
        self._require_draggable("curve")
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
        self._require_draggable("x_range")
        self._configure(self._lock, self.curve, x_range, self._y_range)

    @property
    def y_range(self) -> Range | None:
        """Bounds for y, if any."""
        return self._y_range

    @y_range.setter
    def y_range(self, y_range: Range | None) -> None:
        self._require_draggable("y_range")
        self._configure(self._lock, self.curve, self._x_range, y_range)

    @property
    def coordinates(self) -> Coordinates:
        """When a draggable point shows its coordinates (``"hover"``, ...)."""
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value: Coordinates) -> None:
        if value not in COORDINATE_MODES:
            msg = f"coordinates must be one of {COORDINATE_MODES}, got {value!r}"
            raise ValueError(msg)
        self._coordinates: Coordinates = value

    @property
    def enabled(self) -> bool:
        """Whether a draggable point can be dragged (it is drawn either way)."""
        return self._enabled

    @enabled.setter
    def enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    # --- Drawing ------------------------------------------------------------
    def update(self, plane: Plane) -> None:
        """Bring a draggable point and its handle on ``plane`` into step."""
        super().update(plane)
        if not self._draggable:
            self.detach(plane)
            return
        runtime, error = self._live()
        if error is not None:
            self.detach(plane)
            return
        try:
            _sync(self, plane, self._curve_source(runtime))
        except (TypeError, ValueError) as failure:
            self.detach(plane)
            self._draw_error = str(failure)
            return
        handle = self._handles[plane].point
        handle.color = self._color(plane)
        handle.style = self.style
        handle.coordinates = self._coordinates
        handle.enabled = self._enabled and self._visible

    def handles(self, plane: Plane) -> tuple[DraggablePoint, ...]:
        """Return the point's handle on ``plane`` (if it is draggable and shown)."""
        entry = self._handles.get(plane)
        if entry is None or not self._draggable or not self._visible:
            return ()
        return (entry.point,)

    def detach(self, plane: Plane) -> None:
        """Take the point's handle off ``plane``."""
        entry = self._handles.pop(plane, None)
        if entry is not None:
            entry.point.remove()

    def _detach_all(self) -> None:
        """Take the point's handles off every plane."""
        for entry in list(self._handles.values()):
            entry.point.remove()
        self._handles.clear()

    def _render(self, plane: Plane, surface: pygame.Surface, runtime: Any) -> None:
        if self._draggable:
            return  # drawn as a handle, on top (see handles)
        x, y = runtime[0]
        plane.points(surface, [(x(), y())], style=self.style, color=self._color(plane))

    # --- For the owner -----------------------------------------------------
    def references(self) -> set[str]:
        """Return the names of other objects this point refers to (its curve's)."""
        names = super().references()
        if self._curve_name is not None:
            names.add(self._curve_name)
        return names

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        formulas: list[tuple[Formula, tuple[str, ...]]] = [
            (f, ()) for f in self._coords
        ]
        if self._curve is not None:
            x, y, _ = self._curve
            formulas += [(x, ("t",)), (y, ("t",))]
        return formulas

    def _runtime(self, make: Maker) -> Any:
        getters = tuple(
            getter(make(f, ())) if f.is_text else self._live_getter(index)
            for index, f in enumerate(self._coords)
        )
        curve = None
        if self._curve is not None:
            x, y, t_range = self._curve
            curve = (make(x, ("t",)), make(y, ("t",)), t_range)
        return (getters, curve)

    def _binding(self, runtime: Any) -> Binding | None:
        return PointBinding(lambda: (self.x, self.y))

    def _describe(self) -> str:
        if not self._placed:
            target = self._curve_name or "a curve"
            return f"on {target}"
        if self._whole is not None:
            return f"({show(self._whole)})"
        x, y = self._coords
        if self._numeric:
            return f"({x.source:.6g}, {y.source:.6g})"
        return f"({show(x)}, {show(y)})"

    def _set_state(self, position: tuple[float, float], t: float | None) -> None:
        """Record where the handle put the point (by the sync, after dragging)."""
        self._coords = (Formula(position[0], None), Formula(position[1], None))
        self._placed = True
        if t is not None:
            self._t = t

    def _settings(self) -> tuple[object, ...]:
        """Return what a handle is made with (made again if this changes)."""
        return (
            self._lock,
            self._curve_name,
            self._curve_object,
            self._curve is not None,
            self._x_range,
            self._y_range,
        )

    # --- Internals ----------------------------------------------------------
    @property
    def _on_curve(self) -> bool:
        return (
            self._curve is not None
            or self._curve_name is not None
            or self._curve_object is not None
        )

    @property
    def _numeric(self) -> bool:
        """Whether the position is plain numbers (or not placed yet)."""
        return self._whole is None and not any(
            f.is_text or callable(f.source) for f in self._coords
        )

    def _coordinate(self, index: int) -> float:
        """Return a coordinate's current value (only text needs compiling)."""
        if self._whole is not None:
            return _pick(self._whole, index)
        formula = self._coords[index]
        if not formula.is_text:
            source = formula.source
            if callable(source):
                return as_float(call_quietly(source))
            return float(source)
        runtime, error = self._live()
        if error is not None or runtime is None:
            return math.nan
        return float(runtime[0][index]())

    def _source_of(self, index: int) -> Any:
        """Return one coordinate as given (a number for a whole-position function)."""
        if self._whole is not None:
            return self._coordinate(index)
        return self._coords[index].source

    def _live_getter(self, index: int) -> Callable[[], float]:
        """Read a coordinate that is not text as it is when called."""
        return lambda: self._coordinate(index)

    def _set_position(self, position: PositionLike) -> None:
        """Set the position from what a program gave."""
        was_text = any(f.is_text for f in self._coords)
        if callable(position) and not isinstance(position, tuple):
            if self._draggable:
                msg = "a draggable point's position must be numbers"
                raise ValueError(msg)
            self._whole = position
            self._coords = (Formula(math.nan, None),) * 2
        else:
            x, y = two(position, "position")
            coords = (
                coordinate(x, "the x of position"),
                coordinate(y, "the y of position"),
            )
            if self._draggable and any(f.is_text or callable(f.source) for f in coords):
                msg = "a draggable point's position must be numbers"
                raise ValueError(msg)
            self._whole = None
            self._coords = coords
        self._placed = True
        if was_text or any(f.is_text for f in self._coords):
            self._changed()  # text is compiled; numbers and functions are read live
        else:
            self._own = None

    def _require_draggable(self, what: str) -> None:
        if not self._draggable:
            msg = f"{what} is for draggable points: set draggable = True first"
            raise ValueError(msg)

    def _configure(
        self, lock: object, curve: object, x_range: object, y_range: object
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
        target: CurveObject | None = None
        spec: tuple[Formula, Formula, tuple[float, float]] | None = None
        if isinstance(curve, str):
            name = check_name(curve, "a curve's name")
        elif isinstance(curve, CurveObject):
            target = curve
        elif curve is not None:
            parts = tuple(curve) if isinstance(curve, (tuple, list)) else ()
            if len(parts) != 3:
                msg = (
                    "curve must be a curve object, a curve's name or "
                    f"(x, y, t_range), got {curve!r}"
                )
                raise ValueError(msg)
            x, y, t_range = parts
            spec = (
                component(x, "x(t)"),
                component(y, "y(t)"),
                finite_range(t_range, "t_range"),
            )
        self._lock = lock  # narrowed by the check above
        self._curve_name, self._curve_object, self._curve = name, target, spec
        self._x_range, self._y_range = bounds_x, bounds_y

    def _curve_source(self, runtime: Any) -> _Source | None:
        """Return how to build the curve the point follows, if any."""
        if self._curve is not None:
            fx, fy, t_range = runtime[1]
            return _Source(lambda view, t: (fx, fy, t_range), False)
        target = self._curve_object
        if self._curve_name is not None:
            if self._owner is None:
                msg = (
                    f"the curve {self._curve_name!r} is given by name, which needs "
                    "the point to be in a graph (give the curve object instead)"
                )
                raise ValueError(msg)
            target = self._owner._curve_named(self)
        if target is None:
            return None
        return _Source(target._curve_spec, target.curve_from_zero)


def _pick(whole: Callable[[], Any], index: int) -> float:
    """Return one coordinate of a whole-position function's current value."""
    value = call_quietly(whole)
    try:
        return as_float(value[index])
    except (TypeError, IndexError, KeyError):
        return math.nan


# ----------------------------------------------------------------------
# Keeping a draggable point and its handles in step
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _Source:
    """How to build the curve a draggable point follows.

    Attributes:
        spec: Returns the curve for a view and the point's current parameter
            (a function's graph spans the visible x range, which changes).
        from_zero: Whether a point without a parameter starts at ``t = 0``.
    """

    spec: Callable[[Any, float | None], CurveSpec]
    from_zero: bool


@dataclass(slots=True, eq=False)
class _Handle:
    """A draggable point's handle on one plane.

    Attributes:
        point: The plane's point.
        settings: The point's constraint when the handle was made.
        synced: The point's ``(x, y, t)`` after the last synchronisation.
    """

    point: DraggablePoint
    settings: tuple[object, ...]
    synced: tuple[float, float, float | None]


def _sync(point: Point, plane: Plane, source: _Source | None) -> None:
    """Keep a draggable point and its handle on a plane in step.

    The handle is made the first time, and again when the point's
    constraint changed. A drag (or a changed curve) moves the point; the
    point changed from code (or on another plane) moves the handle. A drag
    wins over a change from code.
    """
    entry = point._handles.get(plane)
    if entry is None or entry.settings != point._settings():
        if entry is not None:
            entry.point.remove()
        point._handles[plane] = _create(point, plane, source)
        return
    handle = entry.point
    curve = None if source is None else source.spec(plane.view, point.t)
    changed_by_code = not _same(_state(point), entry.synced)
    if curve is not None:
        handle.curve = curve  # follows a changed curve, keeping its parameter
    if changed_by_code and not handle.dragging:
        before = entry.synced
        if (
            curve is not None
            and point.t is not None
            and not _same_number(point.t, before[2])
        ):
            handle.t = point.t
        elif point.position is not None:
            handle.position = point.position
    point._set_state(handle.position, handle.t)
    entry.synced = _state(point)


def _create(point: Point, plane: Plane, source: _Source | None) -> _Handle:
    """Make a point's handle on a plane."""
    curve = None if source is None else source.spec(plane.view, point.t)
    position = point.position
    t = point.t
    if curve is not None and t is None and position is None:
        assert source is not None
        t = curve_start(curve, source.from_zero)
    handle = plane.draggable(
        None if curve is not None and t is not None else position,
        lock=point.lock,
        curve=curve,
        t=t if curve is not None else None,
        x_range=point.x_range,
        y_range=point.y_range,
        style=point.style,
        coordinates=point.coordinates,
        enabled=point.enabled,
    )
    point._set_state(handle.position, handle.t)
    return _Handle(handle, point._settings(), _state(point))


def _state(point: Point) -> tuple[float, float, float | None]:
    return (point.x, point.y, point.t)


def _same_number(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return a == b or (math.isnan(a) and math.isnan(b))


def _same(
    a: tuple[float, float, float | None], b: tuple[float, float, float | None]
) -> bool:
    return all(_same_number(p, q) for p, q in zip(a, b, strict=True))
