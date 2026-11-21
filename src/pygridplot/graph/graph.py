"""The :class:`Graph`: a retained list of items, drawn onto a plane each frame.

A graph is to the plane's immediate-mode API what a document is to a pen:
it holds the plots, curves, points, parameters and definitions of a
picture, by name and in order, drawn with one call each frame and saved to
a file. Under the hood it is exactly the immediate-mode API: drawing a
graph makes the same plane calls a program would make by hand, so
everything (reuse of unchanged plots, log axes, themes) works the same.

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
import math
import warnings
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from pygridplot.core.expr import (
    FunctionBinding,
    compile_expression,
)
from pygridplot.core.handles.point import DraggablePoint
from pygridplot.core.plot.curves import CurveSampling
from pygridplot.core.plot.implicit import ImplicitSampling
from pygridplot.core.plot.sampling import SamplingConfig
from pygridplot.core.plot.style import LineStyle, PointStyle, PolygonStyle
from pygridplot.graph.errors import GraphError, GraphWarning
from pygridplot.graph.items import (
    FULL_TURN,
    Coordinate,
    CurveTarget,
    DefinitionItem,
    DraggableItem,
    Expression,
    ImplicitItem,
    Item,
    ParameterItem,
    ParametricItem,
    PlotItem,
    PointItem,
    PointsItem,
    PolarItem,
    PolygonItem,
    PolylineItem,
    StyledItem,
    check_name,
    parse_signature,
)
from pygridplot.graph.lines import interpret
from pygridplot.graph.namespace import Namespace, compile_items
from pygridplot.graph.view import GraphView

if TYPE_CHECKING:
    import pygame

    from pygridplot.plane import Plane


ItemT = TypeVar("ItemT", bound=Item)


@dataclass(slots=True, eq=False)
class _Bound:
    """A draggable item's point on one plane.

    Attributes:
        point: The plane's point.
        settings: The item's constraint when the point was made.
        synced: The item's ``(x, y, t)`` after the last synchronisation.
    """

    point: DraggablePoint
    settings: tuple[object, ...]
    synced: tuple[float, float, float | None]


class Graph:
    """An ordered list of named items: plots, curves, points, parameters, ...

    Add items with :meth:`add` (one line of maths, ``"y = x^2"``) or with
    the method for each kind (:meth:`plot`, :meth:`parametric`,
    :meth:`polar`, :meth:`implicit`, :meth:`point`, :meth:`points`,
    :meth:`polyline`, :meth:`polygon`, :meth:`draggable`,
    :meth:`parameter`, :meth:`define`), which take the same arguments as
    the plane's methods. Each returns its item, which can be changed
    afterwards; ``graph["name"]`` finds a named one.

    Expressions are text in the expression language (``"a sin(x - P.x)"``,
    see :mod:`pygridplot.core.expr`) and may use the names of other items,
    in any order, and Python functions registered with :meth:`register`
    (or ``functions=``). Python functions can also be given directly in
    place of an expression's text; such a graph can be drawn but only
    saved if the functions are registered.

    :meth:`draw` draws the graph onto a plane; :meth:`save` and
    :meth:`load` (or :meth:`to_json` and :meth:`from_json`) keep it in a
    file.

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
        self._items: list[Item] = []
        self._functions: dict[str, Callable[..., Any]] = {}
        self._namespace: Namespace | None = None
        self._draw_errors: dict[Item, str] = {}
        self._reported: dict[Item, str] = {}
        self._next_palette = 0
        self._bound: weakref.WeakKeyDictionary[Plane, dict[DraggableItem, _Bound]] = (
            weakref.WeakKeyDictionary()
        )
        self.view = view
        for name, func in (functions or {}).items():
            self.register(name, func)

    # ------------------------------------------------------------------
    # The items
    # ------------------------------------------------------------------
    @property
    def items(self) -> tuple[Item, ...]:
        """The items, in drawing order."""
        return tuple(self._items)

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self._items)

    def __iter__(self) -> Iterator[Item]:
        """Iterate over the items in drawing order."""
        return iter(tuple(self._items))

    def __getitem__(self, key: str | int) -> Item:
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

    def get(self, name: str, default: Item | None = None) -> Item | None:
        """Return the item with a name, or ``default``.

        Args:
            name: The name.
            default: What to return if no item has it.

        Returns:
            The item, or ``default``.
        """
        return next((item for item in self._items if item.name == name), default)

    def append(self, item: Item, *, index: int | None = None) -> Item:
        """Add an item made separately (or taken from another graph).

        Args:
            item: The item.
            index: Its position (default: last).

        Returns:
            The item.

        Raises:
            GraphError: If it already belongs to a graph, or its name is
                taken.
        """
        if item._graph is not None:
            msg = f"{item!r} already belongs to a graph (remove it first)"
            raise GraphError(msg)
        if item.name is not None:
            self._check_free(item.name)
        if isinstance(item, StyledItem):
            if item.palette is None:
                item.palette = self._next_palette
            self._next_palette = max(self._next_palette, item.palette + 1)
        position = len(self._items) if index is None else index
        self._items.insert(position, item)
        item._graph = self
        self._invalidate()
        return item

    def remove(self, item: Item | str) -> Item:
        """Remove an item (by itself or by name).

        Its points are taken off every plane at once.

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
        target._graph = None
        if isinstance(target, DraggableItem):
            for bound in self._bound.values():
                entry = bound.pop(target, None)
                if entry is not None:
                    entry.point.remove()
        self._reported.pop(target, None)
        self._draw_errors.pop(target, None)
        target._error = None
        self._invalidate()
        return target

    def clear(self) -> None:
        """Remove every item (registered functions and the view are kept)."""
        for item in list(self._items):
            self.remove(item)
        self._next_palette = 0

    def move(self, item: Item | str, index: int) -> None:
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
    def add(self, text: str, **options: Any) -> Item:
        """Add an item from one line of maths, as a graphing calculator reads it.

        ==============================  ======================================
        Line                            Item
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

        Args:
            text: The line.
            **options: Further arguments for the item, as for the method that
                creates it: styles (``color="red"``), ``t_range``,
                ``domain``, ``name``, ... A point with ``draggable=True``
                becomes a draggable point (its coordinates must then be
                numbers), which also accepts ``lock``, ``curve``, and so on.

        Returns:
            The new item.

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
            return self.draggable((x, y), **arguments)
        if method == "define":
            return self.define(arguments.pop("signature"), **arguments)
        creator: Callable[..., Item] = getattr(self, method)
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
    ) -> PlotItem:
        """Add the graph of a function, ``y = f(x)`` (see :class:`PlotItem`).

        Returns:
            The new item.
        """
        return self._new(
            PlotItem(
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
    ) -> ParametricItem:
        """Add a parametric curve ``(x(t), y(t))`` (see :class:`ParametricItem`).

        Returns:
            The new item.
        """
        return self._new(
            ParametricItem(
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
    ) -> PolarItem:
        """Add a polar curve ``r = r(θ)`` (see :class:`PolarItem`).

        Returns:
            The new item.
        """
        return self._new(
            PolarItem(
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
    ) -> ImplicitItem:
        """Add an implicit curve ``F(x, y) = level`` (see :class:`ImplicitItem`).

        Returns:
            The new item.
        """
        return self._new(
            ImplicitItem(
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
    ) -> PointItem:
        """Add a point whose coordinates may be expressions (see :class:`PointItem`).

        Returns:
            The new item.
        """
        return self._new(
            PointItem(
                position,
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
    ) -> PointsItem:
        """Add several points (see :class:`PointsItem`).

        Returns:
            The new item.
        """
        return self._new(
            PointsItem(
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
    ) -> PolylineItem:
        """Add vertices joined by straight lines (see :class:`PolylineItem`).

        Returns:
            The new item.
        """
        return self._new(
            PolylineItem(
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
    ) -> PolygonItem:
        """Add a filled polygon (see :class:`PolygonItem`).

        Returns:
            The new item.
        """
        return self._new(
            PolygonItem(
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
        self,
        position: tuple[float, float] | None = None,
        *,
        name: str | None = None,
        lock: Any = None,
        curve: CurveTarget | None = None,
        t: float | None = None,
        x_range: Any = None,
        y_range: Any = None,
        color: Any = None,
        style: PointStyle | None = None,
        coordinates: Any = "hover",
        enabled: bool = True,
        visible: bool = True,
    ) -> DraggableItem:
        """Add a point the user can drag (see :class:`DraggableItem`).

        Returns:
            The new item.
        """
        return self._new(
            DraggableItem(
                position,
                name=name,
                lock=lock,
                curve=curve,
                t=t,
                x_range=x_range,
                y_range=y_range,
                color=color,
                style=style,
                coordinates=coordinates,
                enabled=enabled,
                visible=visible,
            )
        )

    def parameter(self, name: str, value: Coordinate) -> ParameterItem:
        """Add a named value (see :class:`ParameterItem`).

        Returns:
            The new item.
        """
        return self._new(ParameterItem(name, value))

    def define(self, signature: str, expression: str) -> DefinitionItem:
        """Add a function for other items, not drawn (see :class:`DefinitionItem`).

        Args:
            signature: Its name and parameters, as in ``"g(u, v)"``.
            expression: Its value, in the parameters.

        Returns:
            The new item.
        """
        name, parameters = parse_signature(signature)
        return self._new(DefinitionItem(name, parameters, expression))

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
        if item is not None and item in namespace.errors:
            msg = f"{name!r} has an error: {namespace.errors[item]}"
            raise GraphError(msg)
        msg = f"{name!r} is not a function of this graph"
        raise GraphError(msg)

    @property
    def errors(self) -> dict[Item, str]:
        """The items that cannot be drawn, in order, with the reason for each.

        Errors in expressions (an unknown name, a circular definition, ...)
        are found without drawing; the rare errors found only when drawing
        (an invalid setting for a plane) are those of the last :meth:`draw`.
        """
        found = {**self._compiled().errors, **self._draw_errors}
        return {item: found[item] for item in self._items if item in found}

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
        """Draw every visible item onto a plane, in order, draggable points on top.

        Call it every frame, after :meth:`Plane.draw_grid`. Items with an
        error are skipped (see :attr:`errors`), with a :class:`GraphWarning`
        the first time each error appears.

        Args:
            plane: The plane to draw on.
            surface: The surface (usually the display).
        """
        namespace = self._compiled()
        self._draw_errors = {}
        handles = self._sync(plane, namespace)
        palette = plane.theme.palette
        for item in self._items:
            if not item.visible or item in namespace.errors:
                continue
            if isinstance(item, (DraggableItem, ParameterItem, DefinitionItem)):
                continue
            assert isinstance(item, StyledItem)
            try:
                _draw_item(
                    plane, surface, item, namespace.runtime[item], _color(item, palette)
                )
            except (TypeError, ValueError) as error:
                self._draw_errors[item] = str(error)
        if handles:
            plane.draw_draggables(surface, *handles)
        self._warn({**namespace.errors, **self._draw_errors})

    def detach(self, plane: Plane) -> None:
        """Take the graph's draggable points off a plane it will no longer draw on.

        Args:
            plane: The plane.
        """
        for entry in self._bound.pop(plane, {}).values():
            entry.point.remove()

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

    def _rename(self, item: Item, name: str | None) -> None:
        """Check a new name for an item (called by the item)."""
        if name is not None and name != item.name:
            self._check_free(name)

    def _invalidate(self) -> None:
        """Forget the compiled namespace (names or expressions changed)."""
        self._namespace = None

    def _compiled(self) -> Namespace:
        """Return the compiled namespace, compiling first if anything changed."""
        if self._namespace is None:
            namespace = compile_items(self._items, self._functions)
            self._namespace = namespace
            for item in self._items:
                item._error = namespace.errors.get(item, self._draw_errors.get(item))
        return self._namespace

    def _warn(self, errors: Mapping[Item, str]) -> None:
        """Record the errors of a drawing, warning about those that are new.

        An item's error is warned about when it appears or changes, not again
        while it stays the same; :attr:`Item.error` always holds it.
        """
        for index, item in enumerate(self._items):
            message = errors.get(item)
            item._error = message
            if message is None:
                self._reported.pop(item, None)
                continue
            if self._reported.get(item) == message:
                continue
            self._reported[item] = message
            label = repr(item.name) if item.name else f"item {index} ({item.kind})"
            warnings.warn(f"{label}: {message}", GraphWarning, stacklevel=3)

    def _sync(self, plane: Plane, namespace: Namespace) -> list[DraggablePoint]:
        """Keep each draggable item's point on a plane in step with the item.

        Returns:
            The visible points, to draw on top.
        """
        bound = self._bound.setdefault(plane, {})
        draggables = [i for i in self._items if isinstance(i, DraggableItem)]
        present = set(draggables)
        for item in list(bound):
            if item not in present or item in namespace.errors:
                bound.pop(item).point.remove()
        palette = plane.theme.palette
        handles: list[DraggablePoint] = []
        for item in draggables:
            if item in namespace.errors:
                continue
            source = namespace.curves.get(item)
            entry = bound.get(item)
            try:
                if entry is None or entry.settings != item._settings():
                    if entry is not None:
                        entry.point.remove()
                    entry = _create_point(plane, item, source)
                    bound[item] = entry
                else:
                    _update_point(plane, item, entry, source)
            except (TypeError, ValueError) as error:
                stale = bound.pop(item, None)
                if stale is not None:  # do not leave an undrawn point draggable
                    stale.point.remove()
                self._draw_errors[item] = str(error)
                continue
            point = entry.point
            point.color = _color(item, palette)
            point.style = item.style
            point.coordinates = item.coordinates
            point.enabled = item.enabled and item.visible
            if item.visible:
                handles.append(point)
        return handles


def _color(item: StyledItem, palette: Sequence[Any]) -> Any:
    """Return an item's colour: its own, or its palette slot's."""
    if item.color is not None:
        return item.color
    slot = item.palette or 0
    return palette[slot % len(palette)]


def _draw_item(
    plane: Plane, surface: pygame.Surface, item: StyledItem, runtime: Any, color: Any
) -> None:
    """Draw one item with the plane's method for its kind."""
    style = item.style
    if isinstance(item, PlotItem):
        plane.plot(
            surface,
            runtime,
            style=style,
            color=color,
            domain=item.domain,
            sampling=item.sampling,
        )
    elif isinstance(item, ParametricItem):
        x, y = runtime
        plane.parametric(
            surface,
            x,
            y,
            item.t_range,
            style=style,
            color=color,
            sampling=item.sampling,
        )
    elif isinstance(item, PolarItem):
        plane.polar(
            surface,
            runtime,
            item.theta_range,
            style=style,
            color=color,
            sampling=item.sampling,
        )
    elif isinstance(item, ImplicitItem):
        plane.implicit(
            surface,
            runtime,
            item.level,
            style=style,
            color=color,
            sampling=item.sampling,
        )
    elif isinstance(item, PointItem):
        plane.points(surface, [runtime()], style=style, color=color)
    elif isinstance(item, PointsItem):
        plane.points(surface, runtime(), style=style, color=color)
    elif isinstance(item, PolylineItem):
        plane.polyline(surface, runtime(), style=style, color=color, closed=item.closed)
    elif isinstance(item, PolygonItem):
        plane.polygon(surface, runtime(), style=style, color=color)


def _create_point(plane: Plane, item: DraggableItem, source: Any) -> _Bound:
    """Make the plane's point for a draggable item."""
    curve = None if source is None else source.spec(plane.view, item.t)
    position = item.position
    t = item.t
    if curve is not None and t is None and position is None:
        t = source.start(curve)
    point = plane.draggable(
        None if curve is not None and t is not None else position,
        lock=item.lock,
        curve=curve,
        t=t if curve is not None else None,
        x_range=item.x_range,
        y_range=item.y_range,
        style=item.style,
        coordinates=item.coordinates,
        enabled=item.enabled,
    )
    item._points.add(point)
    item._set_state(point.position, point.t)
    return _Bound(point, item._settings(), _state(item))


def _update_point(
    plane: Plane, item: DraggableItem, entry: _Bound, source: Any
) -> None:
    """Bring a plane's point and its item into step.

    The point moved by dragging (or because its curve changed) updates the
    item; the item changed from code (or by another plane) moves the point.
    A drag wins over a change from code.
    """
    point = entry.point
    curve = None if source is None else source.spec(plane.view, item.t)
    changed_by_code = not _same(_state(item), entry.synced)
    if curve is not None:
        point.curve = curve  # follows a changed curve, keeping its parameter
    if changed_by_code and not point.dragging:
        before = entry.synced
        if (
            curve is not None
            and item.t is not None
            and not _same_number(item.t, before[2])
        ):
            point.t = item.t
        elif item.position is not None:
            point.position = item.position
    item._set_state(point.position, point.t)
    entry.synced = _state(item)


def _state(item: DraggableItem) -> tuple[float, float, float | None]:
    return (item.x, item.y, item.t)


def _same_number(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return a == b or (math.isnan(a) and math.isnan(b))


def _same(
    a: tuple[float, float, float | None], b: tuple[float, float, float | None]
) -> bool:
    return all(_same_number(p, q) for p, q in zip(a, b, strict=True))
