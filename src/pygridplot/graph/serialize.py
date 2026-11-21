"""Saving graphs as JSON, and reading them back.

The format is plain JSON: an object with a format marker, a version, the
names of the Python functions the graph needs, an optional view, and the
items in order. Each item is an object with its ``kind`` and the arguments
that made it, leaving out those at their defaults (a style lists only the
fields that differ from the default style)::

    {
      "format": "pygridplot-graph",
      "version": 1,
      "view": {"x_range": [-10, 10], "y_range": [-6, 6]},
      "items": [
        {"kind": "parameter", "name": "a", "value": 2},
        {"kind": "draggable", "name": "P", "position": [1, 1], "palette": 0},
        {"kind": "plot", "expression": "a sin(x - P.x) + P.y", "palette": 1,
         "style": {"width": 3}}
      ]
    }

Expressions are saved as their text; numbers as numbers; a Python
function as ``{"function": "name"}``, the name it was registered under.
Colours are saved as given (a name, a hex string or an RGB(A) list).
Infinite ends of a draggable point's bounds are saved as ``null``.

Reading is strict: an unknown key, a value of the wrong type or an invalid
setting is a :class:`~pygridplot.graph.GraphError` that says where it is
(``items[2].style.width: expected a number``), so hand-written files are
easy to fix. Files written by a newer version of the format are refused.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any, Final

from pygridplot.core.expr import ExpressionError
from pygridplot.core.plot.curves import CurveSampling
from pygridplot.core.plot.implicit import ImplicitSampling
from pygridplot.core.plot.sampling import SamplingConfig
from pygridplot.graph.errors import GraphError
from pygridplot.graph.items import (
    FULL_TURN,
    DefinitionItem,
    DraggableItem,
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
)
from pygridplot.graph.view import GraphView

if TYPE_CHECKING:
    from pygridplot.graph.graph import Graph

FORMAT: Final = "pygridplot-graph"
"""The value of the ``format`` key of every saved graph."""

VERSION: Final = 1
"""The version of the format written (and the newest read)."""

_MISSING: Final = object()


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def graph_to_dict(graph: Graph) -> dict[str, Any]:
    """Return a graph as plain data.

    Args:
        graph: The graph.

    Returns:
        The data.

    Raises:
        GraphError: If an item uses a Python function that is not registered.
    """
    data: dict[str, Any] = {"format": FORMAT, "version": VERSION}
    if graph.functions:
        data["functions"] = sorted(graph.functions)
    if graph.view is not None:
        data["view"] = _view_to_dict(graph.view)
    writer = _Writer(graph)
    data["items"] = [
        writer.item(item, f"items[{i}]") for i, item in enumerate(graph.items)
    ]
    return data


class _Writer:
    """Turns items into plain data, knowing the graph's registered functions."""

    def __init__(self, graph: Graph) -> None:
        self.names = {id(func): name for name, func in graph.functions.items()}

    def formula(self, source: Any, path: str) -> Any:
        """Return an expression as saved: its text, a number, a function's name."""
        if isinstance(source, (str, float)):
            return source
        name = self.names.get(id(source))
        if name is None:
            label = getattr(source, "__name__", repr(source))
            msg = (
                f"{path}: uses the Python function {label}, which is not registered: "
                "register it with graph.register(name, function) to save the graph, "
                "or write the expression as text"
            )
            raise GraphError(msg)
        return {"function": name}

    def item(self, item: Item, path: str) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": item.kind}
        if item.name is not None:
            data["name"] = item.name
        if not item.visible:
            data["visible"] = False
        data.update(self.fields(item, path))
        if isinstance(item, StyledItem):
            if item.palette is not None:
                data["palette"] = item.palette
            style = _changed_fields(item.style, type(item).default_style)
            if style:
                data["style"] = style
        sampling = getattr(item, "sampling", None)
        if sampling is not None:
            data["sampling"] = _changed_fields(sampling, type(sampling)())
        return data

    def fields(self, item: Item, path: str) -> dict[str, Any]:
        """Return the fields particular to an item's kind."""
        if isinstance(item, PlotItem):
            result = {"expression": self.formula(item.expression, f"{path}.expression")}
            if item.domain is not None:
                result["domain"] = list(item.domain)
            return result
        if isinstance(item, ParametricItem):
            result = {
                "x": self.formula(item.x, f"{path}.x"),
                "y": self.formula(item.y, f"{path}.y"),
            }
            if item.t_range != FULL_TURN:
                result["t_range"] = list(item.t_range)
            return result
        if isinstance(item, PolarItem):
            result = {"r": self.formula(item.r, f"{path}.r")}
            if item.theta_range != FULL_TURN:
                result["theta_range"] = list(item.theta_range)
            return result
        if isinstance(item, ImplicitItem):
            level = item.level
            result = {"expression": self.formula(item.expression, f"{path}.expression")}
            if level != 0.0:
                result["level"] = list(level) if isinstance(level, tuple) else level
            return result
        if isinstance(item, PointItem):
            return {"position": list(item.position)}
        if isinstance(item, (PointsItem, PolylineItem, PolygonItem)):
            result = {"points": [[_gap(v) for v in point] for point in item.points]}
            if isinstance(item, PolylineItem) and item.closed:
                result["closed"] = True
            return result
        if isinstance(item, DraggableItem):
            return self.draggable(item, path)
        if isinstance(item, ParameterItem):
            return {"value": item.value}
        assert isinstance(item, DefinitionItem)
        return {"parameters": list(item.parameters), "expression": item.expression}

    def draggable(self, item: DraggableItem, path: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if item.position is not None and all(math.isfinite(v) for v in item.position):
            result["position"] = list(item.position)
        if item.t is not None:
            result["t"] = item.t
        if item.lock is not None:
            result["lock"] = item.lock
        curve = item.curve
        if isinstance(curve, str):
            result["curve"] = curve
        elif curve is not None:
            x, y, t_range = curve
            result["curve"] = {
                "x": self.formula(x, f"{path}.curve.x"),
                "y": self.formula(y, f"{path}.curve.y"),
                "t_range": list(t_range),
            }
        for key in ("x_range", "y_range"):
            bounds = getattr(item, key)
            if bounds is not None:
                result[key] = [None if math.isinf(v) else v for v in bounds]
        if item.coordinates != "hover":
            result["coordinates"] = item.coordinates
        if not item.enabled:
            result["enabled"] = False
        return result


def _gap(value: Any) -> Any:
    """Write a NaN coordinate of a data point (a gap) as ``null``."""
    return None if isinstance(value, float) and math.isnan(value) else value


def _view_to_dict(view: GraphView) -> dict[str, Any]:
    data: dict[str, Any] = {
        "x_range": list(view.x_range),
        "y_range": list(view.y_range),
    }
    if view.log_x:
        data["log_x"] = True
    if view.log_y:
        data["log_y"] = True
    if view.equal_aspect is not None:
        data["equal_aspect"] = view.equal_aspect
    return data


def _changed_fields(value: Any, default: Any) -> dict[str, Any]:
    """Return a dataclass' fields that differ from a default instance's."""
    return {
        f.name: _plain(getattr(value, f.name))
        for f in fields(value)
        if getattr(value, f.name) != getattr(default, f.name)
    }


def _plain(value: Any) -> Any:
    """Turn a setting into JSON data (tuples, such as colours, into lists)."""
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def dumps(data: Any, indent: int | None = 2) -> str:
    """Write saved data as JSON that is easy to read and to edit by hand.

    Objects are spread over lines, but lists of plain values (a range, a
    colour, a point) stay on one line, and whole numbers are written without
    a decimal point.

    Args:
        data: The data (from :func:`graph_to_dict`).
        indent: Spaces per level, or ``None`` for a single line.

    Returns:
        The JSON text.

    Raises:
        ValueError: For numbers that JSON cannot represent (NaN, infinity).
    """
    if indent is None:
        return json.dumps(_tidy(data), ensure_ascii=False, allow_nan=False)
    return _dump(_tidy(data), indent, 0)


def _tidy(value: Any) -> Any:
    """Write whole floats as integers (``2`` rather than ``2.0``)."""
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return int(value)
    if isinstance(value, dict):
        return {key: _tidy(v) for key, v in value.items()}
    if isinstance(value, list):
        return [_tidy(v) for v in value]
    return value


def _dump(value: Any, indent: int, level: int) -> str:
    """Write one value at a nesting level."""
    pad, inner = " " * (indent * level), " " * (indent * (level + 1))
    if isinstance(value, dict) and value:
        entries = [
            f"{inner}{json.dumps(key, ensure_ascii=False)}: "
            f"{_dump(item, indent, level + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(entries) + f"\n{pad}}}"
    if (
        isinstance(value, list)
        and value
        and any(isinstance(item, (dict, list)) for item in value)
    ):
        entries = [f"{inner}{_dump(item, indent, level + 1)}" for item in value]
        return "[\n" + ",\n".join(entries) + f"\n{pad}]"
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(", ", ": ")
    )


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
def graph_from_dict(
    cls: type[Graph],
    data: Mapping[str, Any],
    functions: Mapping[str, Callable[..., Any]],
) -> Graph:
    """Make a graph from plain data.

    Args:
        cls: The graph class.
        data: The data.
        functions: The Python functions the graph needs, by name.

    Returns:
        The graph.

    Raises:
        GraphError: If the data is not a valid saved graph.
    """
    reader = _Reader(data, "")
    if reader.take("format", str, None) != FORMAT:
        msg = f"not a saved graph (its format is not {FORMAT!r})"
        raise GraphError(msg)
    version = reader.take("version", int)
    if version > VERSION or version < 1:
        msg = (
            f"the graph was saved in version {version} of the format; "
            f"this version of PyGridPlot reads versions 1 to {VERSION}"
        )
        raise GraphError(msg)
    needed = reader.take("functions", list, [])
    for i, name in enumerate(needed):
        if not isinstance(name, str):
            reader.fail(f"functions[{i}]", "expected a function's name")
    missing = sorted(set(needed) - set(functions))
    if missing:
        msg = (
            f"the graph needs the Python functions {missing}: "
            "pass them as functions={...}"
        )
        raise GraphError(msg)
    view_data = reader.take("view", dict, None)
    items = reader.take("items", list)
    reader.done()
    try:
        graph = cls(functions=functions)
    except (GraphError, TypeError) as error:
        msg = f"functions: {error}"
        raise GraphError(msg) from None
    if view_data is not None:
        graph.view = _view_from_dict(view_data)
    for i, item_data in enumerate(items):
        path = f"items[{i}]"
        if not isinstance(item_data, Mapping):
            msg = f"{path}: expected an object"
            raise GraphError(msg)
        item = _item_from_dict(item_data, path, graph)
        try:
            graph.append(item)
        except GraphError as error:
            msg = f"{path}: {error}"
            raise GraphError(msg) from None
    return graph


class _Reader:
    """Reads the keys of one object of saved data, tracking where it is."""

    def __init__(self, data: Mapping[str, Any], path: str) -> None:
        self.data = data
        self.path = path
        self.used: set[str] = set()

    def where(self, key: str) -> str:
        if not self.path:
            return key
        return f"{self.path}.{key}" if not key.startswith("[") else self.path + key

    def fail(self, key: str, message: str) -> None:
        msg = f"{self.where(key)}: {message}"
        raise GraphError(msg)

    def take(self, key: str, kind: type, default: Any = _MISSING) -> Any:
        """Return a key's value, checking its type (``default`` if it is absent)."""
        self.used.add(key)
        if key not in self.data:
            if default is _MISSING:
                self.fail(key, "missing")
            return default
        value = self.data[key]
        if kind is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                self.fail(key, "expected a number")
            if not math.isfinite(value):
                self.fail(key, "expected a finite number")
            return float(value)
        if kind is int and (isinstance(value, bool) or not isinstance(value, int)):
            self.fail(key, "expected a whole number")
        if kind is dict and not isinstance(value, Mapping):
            self.fail(key, "expected an object")
        if kind is list and not isinstance(value, list):
            self.fail(key, "expected a list")
        if kind is str and not isinstance(value, str):
            self.fail(key, "expected text")
        if kind is bool and not isinstance(value, bool):
            self.fail(key, "expected true or false")
        return value

    def done(self) -> None:
        """Fail if the object has keys that were not read."""
        unknown = sorted(set(self.data) - self.used)
        if unknown:
            self.fail(unknown[0], "unknown key")


def _view_from_dict(data: Mapping[str, Any]) -> GraphView:
    reader = _Reader(data, "view")
    x_range = _pair(reader, "x_range")
    y_range = _pair(reader, "y_range")
    log_x = reader.take("log_x", bool, False)
    log_y = reader.take("log_y", bool, False)
    equal_aspect = reader.take("equal_aspect", bool, None)
    reader.done()
    try:
        return GraphView(x_range, y_range, log_x, log_y, equal_aspect)
    except ValueError as error:
        msg = f"view: {error}"
        raise GraphError(msg) from None


def _pair(reader: _Reader, key: str, *, open_ends: bool = False) -> Any:
    """Read a two-number list (``null`` ends allowed for bounds)."""
    value = reader.take(key, list)
    if len(value) != 2:
        reader.fail(key, "expected two numbers")
    result = []
    for end, number in zip((-math.inf, math.inf), value, strict=True):
        if number is None and open_ends:
            result.append(end)
        elif isinstance(number, bool) or not isinstance(number, (int, float)):
            reader.fail(key, "expected two numbers")
        else:
            result.append(float(number))
    return tuple(result)


def _formula_value(value: Any, path: str, graph: Graph, *, numbers: bool) -> Any:
    """Read a saved formula: text, a number, or a registered function."""
    if isinstance(value, str):
        return value
    if numbers and not isinstance(value, bool) and isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping) and set(value) == {"function"}:
        name = value["function"]
        func = graph.functions.get(name) if isinstance(name, str) else None
        if func is None:
            msg = f"{path}: unknown function {name!r} (pass it as functions={{...}})"
            raise GraphError(msg)
        return func
    expected = "an expression, a number or a function" if numbers else "an expression"
    msg = f"{path}: expected {expected}"
    raise GraphError(msg)


def _coordinate(value: Any, path: str) -> Any:
    if isinstance(value, str):
        return value
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        return float(value)
    msg = f"{path}: expected a number or an expression"
    raise GraphError(msg)


def _points(reader: _Reader, key: str) -> list[tuple[Any, Any]]:
    """Read data points (``null`` is a NaN coordinate: a gap)."""
    value = reader.take(key, list)
    points = []
    for i, point in enumerate(value):
        where = f"{reader.where(key)}[{i}]"
        if not isinstance(point, list) or len(point) != 2:
            msg = f"{where}: expected an [x, y] pair"
            raise GraphError(msg)
        points.append(
            tuple(math.nan if v is None else _coordinate(v, where) for v in point)
        )
    return points  # type: ignore[return-value]


def _style(reader: _Reader, item_type: type[StyledItem]) -> Any:
    data = reader.take("style", dict, None)
    default = item_type.default_style
    if data is None:
        return default
    return _dataclass_from(data, default, f"{reader.where('style')}")


def _sampling(reader: _Reader, kind: type) -> Any:
    data = reader.take("sampling", dict, None)
    if data is None:
        return None
    return _dataclass_from(data, kind(), reader.where("sampling"))


def _dataclass_from(data: Mapping[str, Any], default: Any, path: str) -> Any:
    """Make a settings object from the fields saved for it.

    Each value must have the kind of its field's default (a number, a whole
    number, true or false, text); fields without a fixed kind (colours,
    which may be names or lists) are checked by the settings class itself.
    """
    names = {f.name for f in fields(default)}
    values = {}
    for key, value in data.items():
        if key not in names:
            msg = f"{path}.{key}: unknown key"
            raise GraphError(msg)
        values[key] = _field_value(value, getattr(default, key), f"{path}.{key}")
    try:
        return replace(default, **values)
    except (TypeError, ValueError) as error:
        msg = f"{path}: {error}"
        raise GraphError(msg) from None


def _field_value(value: Any, default: Any, path: str) -> Any:
    """Check a saved setting against its default's kind and convert it."""
    number = not isinstance(value, bool) and isinstance(value, (int, float))
    if isinstance(default, bool):
        if not isinstance(value, bool):
            msg = f"{path}: expected true or false"
            raise GraphError(msg)
        return value
    if isinstance(default, int):
        if not (number and float(value).is_integer()):
            msg = f"{path}: expected a whole number"
            raise GraphError(msg)
        return int(value)
    if isinstance(default, float):
        if not number:
            msg = f"{path}: expected a number"
            raise GraphError(msg)
        return float(value)
    if isinstance(default, str) and not isinstance(value, str):
        msg = f"{path}: expected text"
        raise GraphError(msg)
    return _settings_value(value)


def _settings_value(value: Any) -> Any:
    """Turn JSON data back into a setting (lists, such as colours, into tuples)."""
    if isinstance(value, list):
        return tuple(_settings_value(v) for v in value)
    return value


def _item_from_dict(data: Mapping[str, Any], path: str, graph: Graph) -> Item:
    """Make one item from its saved data."""
    reader = _Reader(data, path)
    kind = reader.take("kind", str)
    name = reader.take("name", str, None)
    visible = reader.take("visible", bool, True)
    try:
        item = _make_item(kind, reader, graph, name)
    except GraphError:
        raise
    except (ExpressionError, TypeError, ValueError) as error:
        detail = error.message if isinstance(error, ExpressionError) else str(error)
        msg = f"{path}: {detail}"
        raise GraphError(msg) from None
    reader.done()
    item.visible = visible
    return item


def _make_item(kind: str, reader: _Reader, graph: Graph, name: str | None) -> Item:
    """Make an item of a kind from the reader's keys."""
    where = reader.where

    def palette() -> int | None:
        value = reader.take("palette", int, None)
        if value is not None and value < 0:
            reader.fail("palette", "expected a non-negative whole number")
        return value  # type: ignore[no-any-return]

    if kind == "plot":
        domain = _pair(reader, "domain") if "domain" in reader.data else None
        return PlotItem(
            _formula_value(
                reader.take("expression", object),
                where("expression"),
                graph,
                numbers=False,
            ),
            name=name,
            domain=domain,
            style=_style(reader, PlotItem),
            sampling=_sampling(reader, SamplingConfig),
            palette=palette(),
        )
    if kind == "parametric":
        return ParametricItem(
            _formula_value(reader.take("x", object), where("x"), graph, numbers=True),
            _formula_value(reader.take("y", object), where("y"), graph, numbers=True),
            _pair(reader, "t_range") if "t_range" in reader.data else FULL_TURN,
            name=name,
            style=_style(reader, ParametricItem),
            sampling=_sampling(reader, CurveSampling),
            palette=palette(),
        )
    if kind == "polar":
        return PolarItem(
            _formula_value(reader.take("r", object), where("r"), graph, numbers=True),
            _pair(reader, "theta_range") if "theta_range" in reader.data else FULL_TURN,
            name=name,
            style=_style(reader, PolarItem),
            sampling=_sampling(reader, CurveSampling),
            palette=palette(),
        )
    if kind == "implicit":
        level = reader.take("level", object, 0.0)
        if isinstance(level, list):
            level = tuple(level)
        return ImplicitItem(
            _formula_value(
                reader.take("expression", object),
                where("expression"),
                graph,
                numbers=False,
            ),
            level,
            name=name,
            style=_style(reader, ImplicitItem),
            sampling=_sampling(reader, ImplicitSampling),
            palette=palette(),
        )
    if kind == "point":
        position = reader.take("position", list)
        if len(position) != 2:
            reader.fail("position", "expected an [x, y] pair")
        return PointItem(
            (
                _coordinate(position[0], where("position")),
                _coordinate(position[1], where("position")),
            ),
            name=name,
            style=_style(reader, PointItem),
            palette=palette(),
        )
    if kind == "points":
        return PointsItem(
            _points(reader, "points"),
            name=name,
            style=_style(reader, PointsItem),
            palette=palette(),
        )
    if kind == "polyline":
        return PolylineItem(
            _points(reader, "points"),
            closed=reader.take("closed", bool, False),
            name=name,
            style=_style(reader, PolylineItem),
            palette=palette(),
        )
    if kind == "polygon":
        return PolygonItem(
            _points(reader, "points"),
            name=name,
            style=_style(reader, PolygonItem),
            palette=palette(),
        )
    if kind == "draggable":
        return _draggable(reader, graph, name, palette())
    if kind == "parameter":
        if name is None:
            reader.fail("name", "missing")
        value = reader.take("value", object)
        return ParameterItem(name, _coordinate(value, where("value")))  # type: ignore[arg-type]
    if kind == "definition":
        if name is None:
            reader.fail("name", "missing")
        parameters = reader.take("parameters", list)
        return DefinitionItem(
            name,  # type: ignore[arg-type]
            parameters,
            reader.take("expression", str),
        )
    reader.fail("kind", f"unknown kind of item {kind!r}")
    raise AssertionError  # unreachable: fail always raises


def _draggable(
    reader: _Reader, graph: Graph, name: str | None, palette: int | None
) -> DraggableItem:
    position = _pair(reader, "position") if "position" in reader.data else None
    t = reader.take("t", float, None)
    lock = reader.take("lock", str, None)
    curve_data = reader.take("curve", object, None)
    curve: Any = None
    if isinstance(curve_data, str):
        curve = curve_data
    elif isinstance(curve_data, Mapping):
        sub = _Reader(curve_data, reader.where("curve"))
        curve = (
            _formula_value(sub.take("x", object), sub.where("x"), graph, numbers=True),
            _formula_value(sub.take("y", object), sub.where("y"), graph, numbers=True),
            _pair(sub, "t_range"),
        )
        sub.done()
    elif curve_data is not None:
        reader.fail("curve", "expected a curve's name or an object")
    x_range = (
        _pair(reader, "x_range", open_ends=True) if "x_range" in reader.data else None
    )
    y_range = (
        _pair(reader, "y_range", open_ends=True) if "y_range" in reader.data else None
    )
    return DraggableItem(
        position,
        name=name,
        lock=lock,
        curve=curve,
        t=t,
        x_range=x_range,
        y_range=y_range,
        style=_style(reader, DraggableItem),
        coordinates=reader.take("coordinates", str, "hover"),
        enabled=reader.take("enabled", bool, True),
        palette=palette,
    )
