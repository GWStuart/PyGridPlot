"""The :class:`Graph`: a named list of objects, with text expressions and saving.

A graph holds the objects of a picture (functions, curves, points, the
:mod:`~pygridplot.objects`), plus parameters and definitions, by name and
in order. It adds to them what a graphing calculator's expression list
has: lines of maths (``graph.add("y = a sin(x)")``), names that other
objects' text can use in any order, errors per object, and saving to a
file. Drawing a graph draws its objects, which make the same plane calls a
program would make by hand, so everything (reuse of unchanged plots, log
axes, themes) works the same.

A graph does not own a plane. Draw it onto any plane, or several; the
program still owns the loop::

    graph = pgp.Graph()
    graph.add("a = 2")
    P = graph.draggable((1, 1), name="P")
    graph.add("y = a sin(x - P.x) + P.y")

    while running:
        for event in pygame.event.get():
            plane.handle_event(event)        # drags P
        plane.draw_grid(screen)
        graph.draw(plane, screen)
        pygame.display.flip()

    graph.save("wave.json")
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from os import PathLike
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from pygridplot.core.expr import FunctionBinding, compile_expression
from pygridplot.core.handles.constraints import Lock, Range
from pygridplot.core.handles.point import Coordinates, DraggablePoint
from pygridplot.core.plot.curves import CurveSampling
from pygridplot.core.plot.implicit import ImplicitSampling
from pygridplot.core.plot.sampling import SamplingConfig
from pygridplot.core.plot.style import LineStyle, PointStyle, PolygonStyle
from pygridplot.graph.definitions import Definition, Parameter, parse_signature
from pygridplot.graph.errors import GraphError
from pygridplot.graph.lines import interpret
from pygridplot.graph.namespace import Namespace, compile_items
from pygridplot.graph.view import GraphView
from pygridplot.objects import (
    FULL_TURN,
    CurveObject,
    CurveTarget,
    Function,
    ImplicitCurve,
    ParametricCurve,
    PlotObject,
    Point,
    Points,
    PolarCurve,
    Polygon,
    Polyline,
    StyledObject,
    check_name,
)
from pygridplot.objects.values import Expression

if TYPE_CHECKING:
    import pygame

    from pygridplot.plane import Plane


ItemT = TypeVar("ItemT", bound=PlotObject)


class Graph:
    """An ordered list of named objects: functions, curves, points, parameters, ...

    Add objects with :meth:`add` (one line of maths, ``"y = x^2"``), with
    the method for each kind (:meth:`plot`, :meth:`parametric`,
    :meth:`polar`, :meth:`implicit`, :meth:`point`, :meth:`points`,
    :meth:`polyline`, :meth:`polygon`, :meth:`draggable`,
    :meth:`parameter`, :meth:`define`), which take the same arguments as
    the plane's methods, or with :meth:`append` for an object made on its
    own (``graph.append(pgp.Function(np.sin))``). Each is returned, and can
    be changed afterwards; ``graph["name"]`` finds a named one.

    Expressions are text in the expression language (``"a sin(x - P.x)"``,
    see :mod:`pygridplot.core.expr`) and may use the names of other items,
    in any order, and Python functions registered with :meth:`register`
    (or ``functions=``). Python functions can also be given directly in
    place of an expression's text; such a graph can be drawn but only
    saved if the functions are registered.

    :meth:`draw` draws the graph onto a plane (or add it to one:
    ``plane.add(graph)``); :meth:`save` and :meth:`load` (or :meth:`to_json`
    and :meth:`from_json`) keep it in a file.

    Args:
        functions: Python functions that expressions can call, by name.
        view: What the graph is meant to be seen at (see :class:`GraphView`).
    """

    def __init__(
        self,
        *,
        functions: Mapping[str, Callable[..., Any]] | None = None,
        view: GraphView | None = None,
    ) -> None:
        self._items: list[PlotObject] = []
        self._functions: dict[str, Callable[..., Any]] = {}
        self._namespace: Namespace | None = None
        self._next_palette = 0
        self.view = view
        for name, func in (functions or {}).items():
            self.register(name, func)

    # ------------------------------------------------------------------
    # The items
    # ------------------------------------------------------------------
    @property
    def items(self) -> tuple[PlotObject, ...]:
        """The items, in drawing order."""
        return tuple(self._items)

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self._items)

    def __iter__(self) -> Iterator[PlotObject]:
        """Iterate over the items in drawing order."""
        return iter(tuple(self._items))

    def __getitem__(self, key: str | int) -> PlotObject:
        """Return the item with a name, or at a position.

        Args:
            key: A name, or an index into :attr:`items`.

        Returns:
            The item.

        Raises:
            KeyError: If no item has the name.
            IndexError: If the index is out of range.
        """
        if isinstance(key, str):
            for item in self._items:
                if item.name == key:
                    return item
            msg = f"no item is named {key!r}"
            raise KeyError(msg)
        return self._items[key]

    def __contains__(self, key: object) -> bool:
        """Return whether an item, or an item with a name, is in the graph."""
        if isinstance(key, str):
            return any(item.name == key for item in self._items)
        return any(item is key for item in self._items)

    def get(self, name: str, default: PlotObject | None = None) -> PlotObject | None:
        """Return the item with a name, or ``default``.

        Args:
            name: The name.
            default: What to return if no item has it.

        Returns:
            The item, or ``default``.
        """
        return next((item for item in self._items if item.name == name), default)

    def append(self, item: ItemT, *, index: int | None = None) -> ItemT:
        """Add an object made on its own (or taken from another graph).

        Args:
            item: The object.
            index: Its position (default: last).

        Returns:
            The object.

        Raises:
            GraphError: If it already belongs to a graph, or its name is
                taken.
        """
        if item._owner is not None:
            msg = f"{item!r} already belongs to a graph (remove it first)"
            raise GraphError(msg)
        if item.name is not None:
            self._check_free(item.name)
        if isinstance(item, StyledObject):
            if item.palette is None:
                item.palette = self._next_palette
            self._next_palette = max(self._next_palette, item.palette + 1)
        position = len(self._items) if index is None else index
        self._items.insert(position, item)
        item._owner = self
        self._invalidate()
        return item

    def remove(self, item: PlotObject | str) -> PlotObject:
        """Remove an item (by itself or by name).

        A draggable point's handles are taken off every plane at once.

        Args:
            item: The item or its name.

        Returns:
            The removed item.

        Raises:
            KeyError: If the item is not in the graph.
        """
        target = self[item] if isinstance(item, str) else item
        if target not in self:
            msg = f"{target!r} is not in this graph"
            raise KeyError(msg)
        self._items = [i for i in self._items if i is not target]
        target._owner = None
        target._changed()  # it compiles on its own from now on
        if isinstance(target, Point):
            target._detach_all()
        self._invalidate()
        return target

    def clear(self) -> None:
        """Remove every item (registered functions and the view are kept)."""
        for item in list(self._items):
            self.remove(item)
        self._next_palette = 0

    def move(self, item: PlotObject | str, index: int) -> None:
        """Move an item to another position (changing what is drawn on top).

        Args:
            item: The item or its name.
            index: Its new position in :attr:`items`.

        Raises:
            KeyError: If the item is not in the graph.
        """
        target = self[item] if isinstance(item, str) else item
        if target not in self:
            msg = f"{target!r} is not in this graph"
            raise KeyError(msg)
        self._items = [i for i in self._items if i is not target]
        self._items.insert(index, target)
        self._invalidate()

    @property
    def names(self) -> tuple[str, ...]:
        """The names of the named items, in order."""
        return tuple(item.name for item in self._items if item.name is not None)

    # ------------------------------------------------------------------
    # Python functions
    # ------------------------------------------------------------------
    @property
    def functions(self) -> Mapping[str, Callable[..., Any]]:
        """The registered Python functions, by name (see :meth:`register`)."""
        return MappingProxyType(self._functions)

    def register(self, name: str, func: Callable[..., Any]) -> None:
        """Let expressions call a Python function by name.

        The function takes NumPy arrays (or numbers) like any function given
        to a plane. A saved graph records the name only, so loading it needs
        the function again (``Graph.load(path, functions={...})``).

        Args:
            name: The name expressions use.
            func: The function.

        Raises:
            GraphError: If the name is invalid, or taken by an item.
            TypeError: If ``func`` is not callable.
        """
        check_name(name, "a function's name")
        if not callable(cast(object, func)):  # checked for callers without types
            msg = f"{name!r} must be registered with a function, got {func!r}"
            raise TypeError(msg)
        if name in self.names:
            msg = f"{name!r} is already the name of an item"
            raise GraphError(msg)
        self._functions[name] = func
        self._invalidate()

    def unregister(self, name: str) -> None:
        """Forget a registered Python function.

        Args:
            name: Its name.

        Raises:
            KeyError: If no function is registered under it.
        """
        del self._functions[name]
        self._invalidate()

    # ------------------------------------------------------------------
    # Adding items
    # ------------------------------------------------------------------
    def add(self, text: str, **options: Any) -> PlotObject:
        """Add an object from one line of maths, as a graphing calculator reads it.

        ==============================  ======================================
        Line                            PlotObject
        ==============================  ======================================
        ``y = x^2`` or ``x^2``          a plot
        ``f(x) = x^2``                  a plot named ``f``
        ``r = 1 + cos(θ)``              a polar curve
        ``(cos(t), sin(t))``            a parametric curve (it uses ``t``)
        ``C = (cos(t), sin(t))``        a parametric curve named ``C``
        ``(1, 2)`` or ``P = (a, 2a)``   a point (named ``P``)
        ``a = 2``                       a parameter
        ``g(u, v) = u v``               a definition
        ``x^2 + y^2 = 9``, ``x = 3``    an implicit curve (other equations)
        ==============================  ======================================

        The objects are those of :mod:`pygridplot.objects`
        (:class:`~pygridplot.Function`, :class:`~pygridplot.ParametricCurve`,
        ..., :class:`~pygridplot.Point`), plus :class:`Parameter` and
        :class:`Definition`.

        Args:
            text: The line.
            **options: Further arguments for the object, as for the method that
                creates it: styles (``color="red"``), ``t_range``,
                ``domain``, ``name``, ... A point with ``draggable=True``
                becomes a draggable point (its coordinates must then be
                numbers), which also accepts ``lock``, ``curve``, and so on.

        Returns:
            The new object.

        Raises:
            ExpressionError: If the line is not valid.
            GraphError: If the name is taken.
        """
        method, arguments = interpret(text)
        draggable = options.pop("draggable", False)
        for key, value in options.items():
            if key in arguments and arguments[key] != value:
                msg = f"{key} is given by the line and by the arguments"
                raise GraphError(msg)
        arguments.update(options)
        if draggable:
            if method != "point":
                msg = f"only a point can be draggable, not a {method}"
                raise GraphError(msg)
            x, y = arguments.pop("position")
            if isinstance(x, str) or isinstance(y, str):
                msg = "a draggable point's coordinates must be numbers"
                raise GraphError(msg)
            return self.point((x, y), draggable=True, **arguments)
        if method == "define":
            return self.define(arguments.pop("signature"), **arguments)
        creator: Callable[..., PlotObject] = getattr(self, method)
        return creator(**arguments)

    def plot(
        self,
        expression: Expression,
        *,
        name: str | None = None,
        domain: tuple[float, float] | None = None,
        color: Any = None,
        width: float | None = None,
        opacity: float | None = None,
        style: LineStyle | None = None,
        sampling: SamplingConfig | None = None,
        visible: bool = True,
    ) -> Function:
        """Add the graph of a function, ``y = f(x)`` (a :class:`~pygridplot.Function`).

        Returns:
            The new item.
        """
        return self._new(
            Function(
                expression,
                name=name,
                domain=domain,
                color=color,
                width=width,
                opacity=opacity,
                style=style,
                sampling=sampling,
                visible=visible,
            )
        )

    def parametric(
        self,
        x: Expression | float,
        y: Expression | float,
        t_range: tuple[float, float] = FULL_TURN,
        *,
        name: str | None = None,
        color: Any = None,
        width: float | None = None,
        opacity: float | None = None,
        style: LineStyle | None = None,
        sampling: CurveSampling | None = None,
        visible: bool = True,
    ) -> ParametricCurve:
        """Add a curve ``(x(t), y(t))`` (a :class:`~pygridplot.ParametricCurve`).

        Returns:
            The new item.
        """
        return self._new(
            ParametricCurve(
                x,
                y,
                t_range,
                name=name,
                color=color,
                width=width,
                opacity=opacity,
                style=style,
                sampling=sampling,
                visible=visible,
            )
        )

    def polar(
        self,
        r: Expression | float,
        theta_range: tuple[float, float] = FULL_TURN,
        *,
        name: str | None = None,
        color: Any = None,
        width: float | None = None,
        opacity: float | None = None,
        style: LineStyle | None = None,
        sampling: CurveSampling | None = None,
        visible: bool = True,
    ) -> PolarCurve:
        """Add a polar curve ``r = r(θ)`` (a :class:`~pygridplot.PolarCurve`).

        Returns:
            The new item.
        """
        return self._new(
            PolarCurve(
                r,
                theta_range,
                name=name,
                color=color,
                width=width,
                opacity=opacity,
                style=style,
                sampling=sampling,
                visible=visible,
            )
        )

    def implicit(
        self,
        expression: Expression,
        level: float | Sequence[float] = 0.0,
        *,
        name: str | None = None,
        color: Any = None,
        width: float | None = None,
        opacity: float | None = None,
        style: LineStyle | None = None,
        sampling: ImplicitSampling | None = None,
        visible: bool = True,
    ) -> ImplicitCurve:
        """Add a curve ``F(x, y) = level`` (a :class:`~pygridplot.ImplicitCurve`).

        Returns:
            The new item.
        """
        return self._new(
            ImplicitCurve(
                expression,
                level,
                name=name,
                color=color,
                width=width,
                opacity=opacity,
                style=style,
                sampling=sampling,
                visible=visible,
            )
        )

    def point(
        self,
        position: Any = None,
        *,
        draggable: bool = False,
        name: str | None = None,
        lock: Lock | None = None,
        curve: CurveTarget | None = None,
        t: float | None = None,
        x_range: Range | None = None,
        y_range: Range | None = None,
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
    ) -> Point:
        """Add a fixed, derived or draggable point (a :class:`~pygridplot.Point`).

        Returns:
            The new point.
        """
        return self._new(
            Point(
                position,
                draggable=draggable,
                name=name,
                lock=lock,
                curve=curve,
                t=t,
                x_range=x_range,
                y_range=y_range,
                color=color,
                radius=radius,
                marker=marker,
                edge_color=edge_color,
                edge_width=edge_width,
                opacity=opacity,
                style=style,
                coordinates=coordinates,
                enabled=enabled,
                visible=visible,
            )
        )

    def points(
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
    ) -> Points:
        """Add several points (a :class:`~pygridplot.Points`).

        Returns:
            The new item.
        """
        return self._new(
            Points(
                points,
                name=name,
                color=color,
                radius=radius,
                marker=marker,
                edge_color=edge_color,
                edge_width=edge_width,
                opacity=opacity,
                style=style,
                visible=visible,
            )
        )

    def polyline(
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
    ) -> Polyline:
        """Add vertices joined by straight lines (a :class:`~pygridplot.Polyline`).

        Returns:
            The new item.
        """
        return self._new(
            Polyline(
                points,
                closed=closed,
                name=name,
                color=color,
                width=width,
                opacity=opacity,
                style=style,
                visible=visible,
            )
        )

    def polygon(
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
    ) -> Polygon:
        """Add a filled polygon (a :class:`~pygridplot.Polygon`).

        Returns:
            The new item.
        """
        return self._new(
            Polygon(
                points,
                name=name,
                color=color,
                width=width,
                opacity=opacity,
                fill=fill,
                fill_color=fill_color,
                fill_opacity=fill_opacity,
                fill_rule=fill_rule,
                style=style,
                visible=visible,
            )
        )

    def draggable(
        self, position: tuple[float, float] | None = None, **options: Any
    ) -> Point:
        """Add a point the user can drag: :meth:`point` with ``draggable=True``.

        Args:
            position: Where it starts (needed unless it follows a curve).
            **options: As for :meth:`point`: ``name``, ``lock``, ``curve``
                (an object, a curve of the graph by name, or
                ``(x(t), y(t), t_range)``), ``t``, bounds, style, ...

        Returns:
            The new point.
        """
        return self.point(position, draggable=True, **options)

    def parameter(self, name: str, value: float | str) -> Parameter:
        """Add a named value (see :class:`Parameter`).

        Returns:
            The new item.
        """
        return self._new(Parameter(name, value))

    def define(self, signature: str, expression: str) -> Definition:
        """Add a function for other objects, not drawn (see :class:`Definition`).

        Args:
            signature: Its name and parameters, as in ``"g(u, v)"``.
            expression: Its value, in the parameters.

        Returns:
            The new item.
        """
        name, parameters = parse_signature(signature)
        return self._new(Definition(name, parameters, expression))

    # ------------------------------------------------------------------
    # Evaluating
    # ------------------------------------------------------------------
    def evaluate(self, expression: str, **variables: Any) -> Any:
        """Evaluate an expression with the graph's names.

        For example ``graph.evaluate("f(2) + a")``, or with variables,
        ``graph.evaluate("f(x)", x=np.linspace(0, 1, 5))``.

        Args:
            expression: The expression.
            **variables: Values for its variables (numbers or arrays).

        Returns:
            The value: a float for numbers, an array for arrays.

        Raises:
            ExpressionError: If the expression is not valid, or uses a name
                that is not defined (or has an error).
        """
        namespace = self._compiled()
        compiled = compile_expression(expression, tuple(variables), namespace.bindings)
        value = compiled(*variables.values())
        return float(value) if np.ndim(value) == 0 else value

    def function(self, name: str) -> Callable[..., Any]:
        """Return a named function (a plot, polar or implicit curve, a definition).

        The function works on numbers and arrays, like any function of the
        graph, and can be plotted or differentiated directly.

        Args:
            name: Its name.

        Returns:
            The function.

        Raises:
            GraphError: If no function has the name, or it has an error.
        """
        namespace = self._compiled()
        binding = namespace.bindings.get(name)
        if isinstance(binding, FunctionBinding):
            return binding.func
        item = self.get(name)
        if item is not None and item.error is not None:
            msg = f"{name!r} has an error: {item.error}"
            raise GraphError(msg)
        msg = f"{name!r} is not a function of this graph"
        raise GraphError(msg)

    @property
    def errors(self) -> dict[PlotObject, str]:
        """The objects that cannot be drawn, in order, with the reason for each.

        Errors in expressions (an unknown name, a circular definition, ...)
        are found without drawing; the rare errors found only when drawing
        (an invalid setting for a plane) are those of the last :meth:`draw`.
        """
        found = {item: item.error for item in self._items}
        return {item: error for item, error in found.items() if error is not None}

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    def capture_view(self, plane: Plane) -> GraphView:
        """Remember what a plane shows as the graph's :attr:`view` (saved with it).

        Args:
            plane: The plane.

        Returns:
            The view.
        """
        self.view = GraphView.of(plane)
        return self.view

    def apply_view(self, plane: Plane) -> None:
        """Show the graph's :attr:`view` on a plane (nothing if it has none).

        Args:
            plane: The plane.
        """
        if self.view is not None:
            self.view.apply(plane)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw(self, plane: Plane, surface: pygame.Surface) -> None:
        """Draw every visible object onto a plane, in order, draggable points on top.

        Call it every frame, after :meth:`Plane.draw_grid`, or add the graph
        to the plane (``plane.add(graph)``) and let :meth:`Plane.draw` do it.
        Objects with an error are skipped (see :attr:`errors`), with a
        :class:`~pygridplot.PlotWarning` the first time each error appears.

        Args:
            plane: The plane to draw on.
            surface: The surface (usually the display).
        """
        self.update(plane)
        self.render(plane, surface)
        shown = self.handles(plane)
        if shown:
            plane.draw_draggables(surface, *shown)

    def update(self, plane: Plane) -> None:
        """Compile if anything changed, and update every object (half of :meth:`draw`).

        Args:
            plane: The plane about to be drawn on.
        """
        self._compiled()
        for item in self._items:
            item.update(plane)

    def render(self, plane: Plane, surface: pygame.Surface) -> None:
        """Draw every object, in order (the second half of :meth:`draw`).

        Args:
            plane: The plane.
            surface: The surface.
        """
        for item in self._items:
            item.render(plane, surface)

    def handles(self, plane: Plane) -> tuple[DraggablePoint, ...]:
        """Return the handles of the graph's draggable points on a plane.

        Args:
            plane: The plane.

        Returns:
            The handles of the visible draggable points, to draw last.
        """
        return tuple(h for item in self._items for h in item.handles(plane))

    def detach(self, plane: Plane) -> None:
        """Take the graph's draggable points off a plane it will no longer draw on.

        Args:
            plane: The plane.
        """
        for item in self._items:
            item.detach(plane)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return the graph as plain data (dicts, lists, strings, numbers).

        Returns:
            The data, ready for :func:`json.dumps`.

        Raises:
            GraphError: If an item uses a Python function that is not
                registered (see :meth:`register`).
        """
        from pygridplot.graph.serialize import graph_to_dict

        return graph_to_dict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return the graph as JSON text (see :meth:`to_dict`).

        Args:
            indent: Indentation (``None`` for one line).

        Returns:
            The JSON text.
        """
        from pygridplot.graph.serialize import dumps

        return dumps(self.to_dict(), indent)

    def save(self, path: str | PathLike[str]) -> None:
        """Save the graph to a JSON file (UTF-8).

        Args:
            path: The file.
        """
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.to_json())
            file.write("\n")

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        functions: Mapping[str, Callable[..., Any]] | None = None,
    ) -> Graph:
        """Make a graph from data made by :meth:`to_dict`.

        Args:
            data: The data.
            functions: The Python functions the graph was saved with.

        Returns:
            The graph.

        Raises:
            GraphError: If the data is not a valid saved graph, or needs
                functions that are not given.
        """
        from pygridplot.graph.serialize import graph_from_dict

        return graph_from_dict(cls, data, functions or {})

    @classmethod
    def from_json(
        cls, text: str, *, functions: Mapping[str, Callable[..., Any]] | None = None
    ) -> Graph:
        """Make a graph from JSON text made by :meth:`to_json`.

        Args:
            text: The JSON text.
            functions: The Python functions the graph was saved with.

        Returns:
            The graph.

        Raises:
            GraphError: If the text is not a valid saved graph.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            msg = f"not valid JSON: {error}"
            raise GraphError(msg) from None
        if not isinstance(data, Mapping):
            msg = "a saved graph is a JSON object"
            raise GraphError(msg)
        return cls.from_dict(data, functions=functions)

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        *,
        functions: Mapping[str, Callable[..., Any]] | None = None,
    ) -> Graph:
        """Load a graph saved with :meth:`save`.

        Args:
            path: The file.
            functions: The Python functions the graph was saved with.

        Returns:
            The graph.

        Raises:
            GraphError: If the file is not a valid saved graph.
            OSError: If the file cannot be read.
        """
        with open(path, encoding="utf-8") as file:
            return cls.from_json(file.read(), functions=functions)

    def __repr__(self) -> str:
        """Return the number of items and their names."""
        names = ", ".join(self.names)
        return f"Graph({len(self._items)} items{': ' if names else ''}{names})"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _new(self, item: ItemT) -> ItemT:
        """Add a new item at the end."""
        self.append(item)
        return item

    def _check_free(self, name: str) -> None:
        """Raise if a name is taken by an item or a registered function."""
        if name in self.names:
            msg = f"the name {name!r} is already taken"
            raise GraphError(msg)
        if name in self._functions:
            msg = f"the name {name!r} is a registered function"
            raise GraphError(msg)

    def _rename(self, item: PlotObject, name: str | None) -> None:
        """Check a new name for an object (called by the object)."""
        if name is not None and name != item.name:
            self._check_free(name)

    def _invalidate(self) -> None:
        """Forget the compiled namespace (names or expressions changed)."""
        self._namespace = None

    def _compiled(self) -> Namespace:
        """Return the compiled namespace, compiling first if anything changed."""
        if self._namespace is None:
            self._namespace = compile_items(self._items, self._functions)
        return self._namespace

    def _resolve(self, obj: PlotObject) -> tuple[Any, str | None]:
        """Return an object's compiled runtime and error (for the object)."""
        namespace = self._compiled()
        return namespace.runtime.get(obj), namespace.errors.get(obj)

    def _curve_named(self, point: PlotObject) -> CurveObject:
        """Return the curve a draggable point follows by name (for the point)."""
        return self._compiled().curves[point]  # type: ignore[index]
