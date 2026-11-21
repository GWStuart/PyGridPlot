"""Objects drawn from a list of points: points, polylines and polygons.

Their coordinates are numbers (an ``(N, 2)`` array of data is fine, and
fast: it is drawn as it is), or, coordinate by coordinate, Python functions
of nothing or text without variables, evaluated every time the object is
drawn, so a shape can follow other objects. A NaN coordinate is allowed, as
for the plane: a point that is not drawn, or a gap in a line.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.plot.style import LineStyle, PointStyle, PolygonStyle
from pygridplot.objects.base import (
    LineObject,
    MarkerObject,
    StyledObject,
    style_property,
)
from pygridplot.objects.values import (
    Coordinate,
    Formula,
    Maker,
    coordinate,
    getter,
    is_number,
    two,
)

if TYPE_CHECKING:
    import pygame

    from pygridplot.plane import Plane


class PathData:
    """The points of a ``Points``, ``Polyline`` or ``Polygon`` object.

    Plain numbers are kept in an array, so large data sets cost nothing to
    redraw; formulas (which may change) are evaluated each time.

    Attributes:
        numbers: The coordinates, NaN where a formula gives them.
        formulas: The formulas, by ``(row, column)``.
    """

    __slots__ = ("formulas", "numbers")

    def __init__(self, points: object, what: str) -> None:
        self.formulas: dict[tuple[int, int], Formula] = {}
        if isinstance(points, np.ndarray):
            array = np.asarray(points, dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != 2:
                msg = f"{what} must have shape (N, 2), got {array.shape}"
                raise ValueError(msg)
            if np.isinf(array).any():
                msg = f"{what} must not be infinite (NaN leaves a gap)"
                raise ValueError(msg)
            self.numbers = array.copy()
            return
        try:
            rows = list(points)  # type: ignore[call-overload]
        except TypeError:
            msg = f"{what} must be a sequence of (x, y) pairs, got {points!r}"
            raise TypeError(msg) from None
        self.numbers = np.zeros((len(rows), 2))
        for i, row in enumerate(rows):
            where = f"{what}[{i}]"
            for j, value in enumerate(two(row, where)):
                if is_number(value):
                    number = float(value)
                    if math.isinf(number):
                        msg = f"{where} must not be infinite (NaN leaves a gap)"
                        raise ValueError(msg)
                    self.numbers[i, j] = number
                else:
                    self.formulas[(i, j)] = coordinate(
                        value, f"a coordinate of {where}"
                    )
                    self.numbers[i, j] = np.nan

    def value(self) -> list[tuple[Coordinate, Coordinate]]:
        """Return the points as given (numbers, functions and text)."""
        result: list[tuple[Coordinate, Coordinate]] = []
        for i, (x, y) in enumerate(self.numbers.tolist()):
            fx = self.formulas.get((i, 0))
            fy = self.formulas.get((i, 1))
            result.append(
                (
                    x if fx is None else fx.source,
                    y if fy is None else fy.source,
                )
            )
        return result

    def evaluator(self, make: Maker) -> Callable[[], NDArray[np.float64]]:
        """Return a function giving the points' current coordinates."""
        getters = {key: getter(make(f, ())) for key, f in self.formulas.items()}
        numbers = self.numbers
        if not getters:
            return lambda: numbers

        def evaluate() -> NDArray[np.float64]:
            result: NDArray[np.float64] = numbers.copy()
            for (i, j), get in getters.items():
                result[i, j] = get()
            return result

        return evaluate


class _PathObject(StyledObject):
    """What objects made of a list of points share."""

    def _set_points(self, points: Any) -> None:
        """Replace the points; compile again only if formulas are involved."""
        new = PathData(points, "points")
        old: PathData | None = getattr(self, "_points", None)
        had_formulas = old is not None and bool(old.formulas)
        self._points: PathData = new
        if had_formulas or new.formulas:
            self._changed()
        else:
            self._own = None  # the runtime holds the old array

    @property
    def points(self) -> list[tuple[Coordinate, Coordinate]]:
        """The points as given: numbers, functions and text.

        Assign a new list or array to change them (every frame, for data
        that moves).
        """
        return self._points.value()

    @points.setter
    def points(self, points: Any) -> None:
        self._set_points(points)

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        return [(f, ()) for f in self._points.formulas.values()]

    def _runtime(self, make: Maker) -> Any:
        return self._points.evaluator(make)

    def _live(self) -> tuple[Any, str | None]:
        runtime, error = super()._live()
        if error is None and not self._points.formulas:
            return (lambda: self._points.numbers), None  # numbers: always current
        return runtime, error


class Points(_PathObject, MarkerObject):
    """Several points, drawn like :meth:`Plane.points`.

    Args:
        points: ``(x, y)`` pairs or an ``(N, 2)`` array; each coordinate a
            number, a function of nothing, or text without variables.
        name: The object's name (optional).
        color: Fill colour (default: the object's palette colour).
        radius: Marker size in pixels.
        marker: Marker shape.
        edge_color: Outline colour.
        edge_width: Outline thickness in pixels.
        opacity: Opacity.
        style: A :class:`~pygridplot.PointStyle`.
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the plane or graph).
    """

    kind: ClassVar[str] = "points"

    def __init__(
        self,
        points: Any,
        *,
        name: str | None = None,
        color: Any = None,
        radius: float | None = None,
        marker: Any = None,
        edge_color: Any = None,
        edge_width: float | None = None,
        opacity: float | None = None,
        style: PointStyle | None = None,
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
        )
        self._set_points(points)

    def _render(self, plane: Plane, surface: pygame.Surface, runtime: Any) -> None:
        plane.points(surface, runtime(), style=self.style, color=self._color(plane))

    def _describe(self) -> str:
        return f"{len(self._points.numbers)} points"


class Polyline(_PathObject, LineObject):
    """Points joined by straight lines, drawn like :meth:`Plane.polyline`.

    Args:
        points: The vertices: ``(x, y)`` pairs or an ``(N, 2)`` array; each
            coordinate a number, a function of nothing, or text.
        closed: Join the last point back to the first.
        name: The object's name (optional).
        color: Line colour (default: the object's palette colour).
        width: Line width in pixels.
        opacity: Line opacity.
        style: A :class:`~pygridplot.LineStyle`.
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the plane or graph).
    """

    kind: ClassVar[str] = "polyline"

    def __init__(
        self,
        points: Any,
        *,
        closed: bool = False,
        name: str | None = None,
        color: Any = None,
        width: float | None = None,
        opacity: float | None = None,
        style: LineStyle | None = None,
        visible: bool = True,
        palette: int | None = None,
    ) -> None:
        super().__init__(
            style=style,
            overrides={"color": color, "width": width, "opacity": opacity},
            palette=palette,
            name=name,
            visible=visible,
        )
        self._set_points(points)
        self._closed = bool(closed)

    @property
    def closed(self) -> bool:
        """Whether the last vertex is joined back to the first."""
        return self._closed

    @closed.setter
    def closed(self, closed: bool) -> None:
        self._closed = bool(closed)

    def _render(self, plane: Plane, surface: pygame.Surface, runtime: Any) -> None:
        plane.polyline(
            surface,
            runtime(),
            style=self.style,
            color=self._color(plane),
            closed=self._closed,
        )

    def _describe(self) -> str:
        return f"{len(self._points.numbers)} vertices"


class Polygon(_PathObject):
    """A filled polygon with an outline, drawn like :meth:`Plane.polygon`.

    Args:
        points: The vertices: ``(x, y)`` pairs or an ``(N, 2)`` array; each
            coordinate a number, a function of nothing, or text.
        name: The object's name (optional).
        color: Outline (and fill) colour (default: the palette colour).
        width: Outline thickness in pixels (0 for none).
        opacity: Outline opacity.
        fill: Whether to fill the inside.
        fill_color: Fill colour (default: ``color``).
        fill_opacity: Fill opacity.
        fill_rule: ``"nonzero"`` or ``"evenodd"``.
        style: A :class:`~pygridplot.PolygonStyle`.
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the plane or graph).
    """

    kind: ClassVar[str] = "polygon"
    style_type: ClassVar[type] = PolygonStyle
    default_style: ClassVar[Any] = PolygonStyle()

    def __init__(
        self,
        points: Any,
        *,
        name: str | None = None,
        color: Any = None,
        width: float | None = None,
        opacity: float | None = None,
        fill: bool | None = None,
        fill_color: Any = None,
        fill_opacity: float | None = None,
        fill_rule: Any = None,
        style: PolygonStyle | None = None,
        visible: bool = True,
        palette: int | None = None,
    ) -> None:
        super().__init__(
            style=style,
            overrides={
                "color": color,
                "width": width,
                "opacity": opacity,
                "fill": fill,
                "fill_color": fill_color,
                "fill_opacity": fill_opacity,
                "fill_rule": fill_rule,
            },
            palette=palette,
            name=name,
            visible=visible,
        )
        self._set_points(points)

    width = style_property("width", "float", "The outline's thickness, in pixels.")
    fill = style_property("fill", "bool", "Whether the inside is filled.")
    fill_color = style_property(
        "fill_color",
        "ColorLike | None",
        "The fill colour (``None`` for the outline's).",
    )
    fill_opacity = style_property("fill_opacity", "float", "The fill opacity.")

    def _render(self, plane: Plane, surface: pygame.Surface, runtime: Any) -> None:
        plane.polygon(surface, runtime(), style=self.style, color=self._color(plane))

    def _describe(self) -> str:
        return f"{len(self._points.numbers)} vertices"
