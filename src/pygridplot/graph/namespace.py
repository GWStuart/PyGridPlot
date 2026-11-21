"""Compiling a graph's items: names, dependencies and the functions to draw.

Items refer to each other by name, in any order: a plot may use a parameter
added after it. Compiling therefore works out an order first:

1. Every named item's dependencies are the names its expressions use (and,
   for a draggable point, the curve it follows).
2. Items are compiled after everything they depend on (a topological sort).
   Items that cannot be ordered are part of a circular definition, or
   depend on one; each gets an error saying which.
3. Each item's expressions are compiled with the names compiled so far
   bound to what they stand for (:mod:`pygridplot.core.expr.compiler`): a
   parameter to its value (read live, so changing it needs no
   recompiling), a named plot or definition to its function, a point to
   its coordinates. Registered Python functions are bound from the start.

An item that fails gets an error message, and the items depending on it get
one saying so; everything else compiles and draws normally.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from pygridplot.core.expr import (
    Binding,
    ExpressionError,
    FunctionBinding,
    compile_expression,
)
from pygridplot.core.view import View
from pygridplot.graph.items import (
    DraggableItem,
    Formula,
    Item,
    ParametricItem,
    PlotItem,
    PolarItem,
)

CURVE_KINDS: Final = (PlotItem, ParametricItem, PolarItem)
"""The items a draggable point can follow by name."""

NOUNS: Final = {
    "plot": "a plot",
    "parametric": "a parametric curve",
    "polar": "a polar curve",
    "implicit": "an implicit curve",
    "point": "a point",
    "points": "a set of points",
    "polyline": "a polyline",
    "polygon": "a polygon",
    "draggable": "a draggable point",
    "parameter": "a parameter",
    "definition": "a definition",
}
"""How error messages name each kind of item."""


def noun(item: Item) -> str:
    """Return how error messages name an item's kind (``"a parametric curve"``)."""
    return NOUNS.get(item.kind, f"a {item.kind}")


CurveSpec = tuple[Any, Any, tuple[float, float]]
"""A curve for a draggable point: ``(x(t), y(t), t_range)``."""


@dataclass(frozen=True, slots=True)
class CurveSource:
    """How to build the curve a draggable point follows.

    Attributes:
        spec: Returns the curve for a view and the point's current parameter
            (a function's graph spans the visible x range, which changes).
        from_zero: Whether a point without a parameter starts at ``t = 0``
            (on a function's graph, at ``x = 0``) rather than at the start of
            the range.
    """

    spec: Callable[[View, float | None], CurveSpec]
    from_zero: bool = False

    def start(self, curve: CurveSpec) -> float:
        """Return where a point without a parameter starts on ``curve``."""
        low, high = curve[2]
        return 0.0 if self.from_zero and low <= 0.0 <= high else low


@dataclass(slots=True)
class Namespace:
    """The result of compiling a graph.

    Attributes:
        runtime: What each compiled item needs to be drawn.
        errors: Why each failed item failed.
        bindings: What each name stands for in expressions.
        curves: For draggable points on a curve: how to build the curve.
    """

    runtime: dict[Item, Any] = field(default_factory=dict)
    errors: dict[Item, str] = field(default_factory=dict)
    bindings: dict[str, Binding] = field(default_factory=dict)
    curves: dict[DraggableItem, CurveSource] = field(default_factory=dict)


def compile_items(
    items: Sequence[Item], functions: Mapping[str, Callable[..., Any]]
) -> Namespace:
    """Compile a graph's items.

    Args:
        items: The items, in the graph's order.
        functions: Registered Python functions, by name.

    Returns:
        The compiled namespace.
    """
    result = Namespace()
    result.bindings.update(
        {name: FunctionBinding.of(func) for name, func in functions.items()}
    )
    named = {item.name: item for item in items if item.name is not None}
    depends = {
        item: [name for name in sorted(item.references()) if name in named]
        for item in items
    }
    order, stuck = _order(items, depends, named)
    for item in stuck:
        result.errors[item] = _cycle_message(item, depends, named, stuck)
    for item in order:
        message = _compile_item(item, depends[item], named, result)
        if message is not None:
            result.errors[item] = message
    return result


def _compile_item(
    item: Item, depends: list[str], named: dict[str, Item], result: Namespace
) -> str | None:
    """Compile one item into ``result``; return its error message, if any."""
    for name in depends:
        if named[name] in result.errors:
            return f"depends on {name!r}, which has an error"
    used = set()
    for formula, variables in item._formulas():
        used |= formula.names(variables)
    for name in sorted(used):
        target = named.get(name)
        if target is not None and name not in result.bindings:
            return f"{name!r} is {noun(target)} and cannot be used in an expression"

    def make(formula: Formula, variables: tuple[str, ...]) -> Any:
        if formula.node is None:
            return formula.source  # a Python function or a number
        return compile_expression(
            formula.node, variables, result.bindings, text=str(formula.source)
        )

    try:
        runtime = item._runtime(make)
    except ExpressionError as error:
        return error.message
    if isinstance(item, DraggableItem):
        curve = _curve(item, runtime, named, result)
        if isinstance(curve, str):
            return curve
        if curve is not None:
            result.curves[item] = curve
    result.runtime[item] = runtime
    if item.name is not None:
        binding = item._binding(runtime)
        if binding is not None:
            result.bindings[item.name] = binding
    return None


def _curve(
    item: DraggableItem,
    runtime: Any,
    named: dict[str, Item],
    result: Namespace,
) -> CurveSource | str | None:
    """Return how to build a draggable point's curve, or an error message."""
    if runtime is not None:  # its own (x(t), y(t), t_range)
        x, y, t_range = runtime
        return CurveSource(lambda view, t: (x, y, t_range))
    name = item._curve_name
    if name is None:
        return None
    target = named.get(name)
    if target is None:
        return f"unknown curve {name!r}"
    if not isinstance(target, CURVE_KINDS):
        return (
            f"{name!r} is {noun(target)}: a draggable point can follow a plot, "
            "a parametric curve or a polar curve"
        )
    compiled = result.runtime[target]
    if isinstance(target, PlotItem):
        return CurveSource(_plot_curve(compiled, target.domain), from_zero=True)
    if isinstance(target, ParametricItem):
        fx, fy = compiled
        t_range = target.t_range
        return CurveSource(lambda view, t: (fx, fy, t_range))
    return CurveSource(_polar_curve(compiled, target.theta_range))


def _plot_curve(
    func: Any, domain: tuple[float, float] | None
) -> Callable[[View, float | None], CurveSpec]:
    """A function's graph as a curve ``(t, f(t))``.

    Without a domain, the curve spans the visible x range and as much again
    on both sides (and always the point's own parameter), so the point can
    be dragged anywhere on screen.
    """

    def identity(t: Any) -> Any:
        return t

    def spec(view: View, t: float | None) -> CurveSpec:
        if domain is not None:
            return (identity, func, domain)
        bounds = view.bounds
        if view.log_x:
            low, high = bounds.x_min / 10.0, bounds.x_max * 10.0
        else:
            width = bounds.x_max - bounds.x_min
            low, high = bounds.x_min - width, bounds.x_max + width
        if t is not None and math.isfinite(t):
            low, high = min(low, t - 1.0), max(high, t + 1.0)
        return (identity, func, (low, high))

    return spec


def _polar_curve(
    radius: Any, theta_range: tuple[float, float]
) -> Callable[[View, float | None], CurveSpec]:
    """A polar curve as ``(r(θ) cos θ, r(θ) sin θ)``."""

    def r(theta: Any) -> Any:
        return radius(theta) if callable(radius) else np.full(np.shape(theta), radius)

    def x(theta: Any) -> Any:
        return np.multiply(r(theta), np.cos(theta))

    def y(theta: Any) -> Any:
        return np.multiply(r(theta), np.sin(theta))

    return lambda view, t: (x, y, theta_range)


def _order(
    items: Sequence[Item], depends: dict[Item, list[str]], named: dict[str, Item]
) -> tuple[list[Item], list[Item]]:
    """Order items after their dependencies (Kahn's algorithm).

    Returns:
        ``(order, stuck)``: the items that can be ordered, in a valid order
        (keeping the graph's order where there is a choice), and those that
        cannot (in or after a circular definition).
    """
    waiting = {item: {named[name] for name in depends[item]} for item in items}
    needed_by: dict[Item, list[Item]] = {item: [] for item in items}
    for item, needs in waiting.items():
        for dependency in needs:
            needed_by[dependency].append(item)
    position = {item: i for i, item in enumerate(items)}
    ready = sorted(
        (item for item in items if not waiting[item]), key=position.__getitem__
    )
    order: list[Item] = []
    while ready:
        item = ready.pop(0)
        order.append(item)
        freed = []
        for dependent in needed_by[item]:
            waiting[dependent].discard(item)
            if not waiting[dependent]:
                freed.append(dependent)
        ready = sorted([*ready, *freed], key=position.__getitem__)
    done = set(order)
    return order, [item for item in items if item not in done]


def _cycle_message(
    item: Item,
    depends: dict[Item, list[str]],
    named: dict[str, Item],
    stuck: list[Item],
) -> str:
    """Explain why an item could not be ordered: a cycle it is in, or one it needs."""
    blocked = set(stuck)
    path = [item]
    current = item
    while True:
        following = [named[n] for n in depends[current] if named[n] in blocked]
        if not following:  # cannot happen for a stuck item, but stay safe
            return "circular definition"
        current = following[0]
        if current in path:
            cycle = [*path[path.index(current) :], current]
            if item in cycle:
                return "circular definition: " + " → ".join(
                    str(step.name) for step in cycle
                )
            first = next(n for n in depends[item] if named[n] in blocked)
            return f"depends on {first!r}, which has an error"
        path.append(current)
