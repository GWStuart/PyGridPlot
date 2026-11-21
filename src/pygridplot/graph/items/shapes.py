"""Items drawn as points and shapes: points, polylines and polygons.

Their coordinates are numbers, or expressions without variables that are
evaluated when drawn, so a shape can follow parameters and points.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.expr import (
    Binding,
    PointBinding,
)
from pygridplot.core.plot.style import (
    LineStyle,
    PointStyle,
    PolygonStyle,
)
from pygridplot.graph.items.base import (
    Coordinate,
    Formula,
    LineItem,
    Maker,
    PointLikeItem,
    StyledItem,
    coordinates,
    getter,
    is_number,
    show,
    style_property,
    two,
)


class PointItem(PointLikeItem):
    """A single point, drawn like :meth:`Plane.points`.

    Its coordinates may be expressions, so the point moves with what it
    refers to (``("a", "a^2")``). A named point's coordinates are ``P.x``
    and ``P.y`` in other items.

    Args:
        position: ``(x, y)``: numbers, or expressions without variables.
        name: The point's name (optional).
        color: Fill colour (default: the item's palette colour).
        radius: Marker size in pixels.
        marker: Marker shape.
        edge_color: Outline colour.
        edge_width: Outline thickness in pixels.
        opacity: Opacity.
        style: A :class:`~pygridplot.PointStyle`.
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the graph).
    """

    kind: ClassVar[str] = "point"

    def __init__(
        self,
        position: tuple[Coordinate, Coordinate],
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
        self._position = coordinates(position, "position")

    @property
    def position(self) -> tuple[Coordinate, Coordinate]:
        """``(x, y)`` as given: numbers or expressions."""
        return (self._position[0].source, self._position[1].source)  # type: ignore[return-value]

    @position.setter
    def position(self, position: tuple[Coordinate, Coordinate]) -> None:
        self._position = coordinates(position, "position")
        self._changed()

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        return [(f, ()) for f in self._position]

    def _runtime(self, make: Maker) -> Any:
        x, y = (getter(make(f, ())) for f in self._position)
        return lambda: (x(), y())

    def _binding(self, runtime: Any) -> Binding | None:
        return PointBinding(runtime)

    def _describe(self) -> str:
        return f"({show(self._position[0])}, {show(self._position[1])})"


class _PathItem:
    """The points of a ``points``, ``polyline`` or ``polygon`` item.

    Plain numbers are kept in an array, so large data sets cost nothing to
    redraw; expressions (which may change) are evaluated each frame. A NaN
    coordinate is allowed, as on the plane: a point that is not drawn, or a
    gap in a line.
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
                    formula = Formula.make(value, f"a coordinate of {where}")
                    self.formulas[(i, j)] = formula
                    self.numbers[i, j] = np.nan

    def value(self) -> list[tuple[Coordinate, Coordinate]]:
        """Return the points as given (numbers and expressions)."""
        result: list[tuple[Coordinate, Coordinate]] = []
        for i, (x, y) in enumerate(self.numbers.tolist()):
            fx = self.formulas.get((i, 0))
            fy = self.formulas.get((i, 1))
            result.append(
                (
                    x if fx is None else fx.source,  # type: ignore[arg-type]
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


class PointsItem(PointLikeItem):
    """Several points, drawn like :meth:`Plane.points`.

    Args:
        points: ``(x, y)`` pairs (numbers or expressions without variables),
            or an ``(N, 2)`` array.
        name: The item's name (optional, to find it with ``graph[name]``).
        color: Fill colour (default: the item's palette colour).
        radius: Marker size in pixels.
        marker: Marker shape.
        edge_color: Outline colour.
        edge_width: Outline thickness in pixels.
        opacity: Opacity.
        style: A :class:`~pygridplot.PointStyle`.
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the graph).
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
        self._points = _PathItem(points, "points")

    @property
    def points(self) -> list[tuple[Coordinate, Coordinate]]:
        """The points as given: numbers and expressions."""
        return self._points.value()

    @points.setter
    def points(self, points: Any) -> None:
        self._points = _PathItem(points, "points")
        self._changed()

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        return [(f, ()) for f in self._points.formulas.values()]

    def _runtime(self, make: Maker) -> Any:
        return self._points.evaluator(make)

    def _describe(self) -> str:
        return f"{len(self._points.numbers)} points"


class PolylineItem(LineItem):
    """Points joined by straight lines, drawn like :meth:`Plane.polyline`.

    Args:
        points: ``(x, y)`` pairs (numbers or expressions without variables),
            or an ``(N, 2)`` array.
        closed: Join the last point back to the first.
        name: The item's name (optional).
        color: Line colour (default: the item's palette colour).
        width: Line width in pixels.
        opacity: Line opacity.
        style: A :class:`~pygridplot.LineStyle`.
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the graph).
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
        self._points = _PathItem(points, "points")
        self._closed = bool(closed)

    @property
    def points(self) -> list[tuple[Coordinate, Coordinate]]:
        """The vertices as given: numbers and expressions."""
        return self._points.value()

    @points.setter
    def points(self, points: Any) -> None:
        self._points = _PathItem(points, "points")
        self._changed()

    @property
    def closed(self) -> bool:
        """Whether the last vertex is joined back to the first."""
        return self._closed

    @closed.setter
    def closed(self, closed: bool) -> None:
        self._closed = bool(closed)

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        return [(f, ()) for f in self._points.formulas.values()]

    def _runtime(self, make: Maker) -> Any:
        return self._points.evaluator(make)

    def _describe(self) -> str:
        return f"{len(self._points.numbers)} vertices"


class PolygonItem(StyledItem):
    """A filled polygon with an outline, drawn like :meth:`Plane.polygon`.

    Args:
        points: The vertices: ``(x, y)`` pairs (numbers or expressions
            without variables), or an ``(N, 2)`` array.
        name: The item's name (optional).
        color: Outline (and fill) colour (default: the item's palette colour).
        width: Outline thickness in pixels (0 for none).
        opacity: Outline opacity.
        fill: Whether to fill the inside.
        fill_color: Fill colour (default: ``color``).
        fill_opacity: Fill opacity.
        fill_rule: ``"nonzero"`` or ``"evenodd"``.
        style: A :class:`~pygridplot.PolygonStyle`.
        visible: Whether it is drawn.
        palette: Palette slot (normally chosen by the graph).
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
        self._points = _PathItem(points, "points")

    width = style_property("width", "float", "The outline's thickness, in pixels.")
    fill = style_property("fill", "bool", "Whether the inside is filled.")
    fill_color = style_property(
        "fill_color",
        "ColorLike | None",
        "The fill colour (``None`` for the outline's).",
    )
    fill_opacity = style_property("fill_opacity", "float", "The fill opacity.")

    @property
    def points(self) -> list[tuple[Coordinate, Coordinate]]:
        """The vertices as given: numbers and expressions."""
        return self._points.value()

    @points.setter
    def points(self, points: Any) -> None:
        self._points = _PathItem(points, "points")
        self._changed()

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        return [(f, ()) for f in self._points.formulas.values()]

    def _runtime(self, make: Maker) -> Any:
        return self._points.evaluator(make)

    def _describe(self) -> str:
        return f"{len(self._points.numbers)} vertices"
