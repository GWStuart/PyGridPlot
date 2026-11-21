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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from pygridplot.core.expr import (
    Binding,
    ExpressionError,
    FunctionBinding,
    compile_expression,
)
from pygridplot.objects import (
    CurveObject,
    Formula,
    PlotObject,
    Point,
)

NOUNS: Final = {
    "plot": "a plot",
    "parametric": "a parametric curve",
    "polar": "a polar curve",
    "implicit": "an implicit curve",
    "point": "a point",
    "points": "a set of points",
    "polyline": "a polyline",
    "polygon": "a polygon",
    "parameter": "a parameter",
    "definition": "a definition",
}
"""How error messages name each kind of object."""


def noun(item: PlotObject) -> str:
    """Return how error messages name an object's kind (``"a parametric curve"``)."""
    if isinstance(item, Point) and item.draggable:
        return "a draggable point"
    return NOUNS.get(item.kind, f"a {item.kind}")


@dataclass(slots=True)
class Namespace:
    """The result of compiling a graph.

    Attributes:
        runtime: What each compiled item needs to be drawn.
        errors: Why each failed item failed.
        bindings: What each name stands for in expressions.
        curves: For draggable points following a curve by name: the curve.
    """

    runtime: dict[PlotObject, Any] = field(default_factory=dict)
    errors: dict[PlotObject, str] = field(default_factory=dict)
    bindings: dict[str, Binding] = field(default_factory=dict)
    curves: dict[Point, CurveObject] = field(default_factory=dict)


def compile_items(
    items: Sequence[PlotObject], functions: Mapping[str, Callable[..., Any]]
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
    item: PlotObject,
    depends: list[str],
    named: dict[str, PlotObject],
    result: Namespace,
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
    if isinstance(item, Point) and item._curve_name is not None:
        target = named.get(item._curve_name)
        if target is None:
            return f"unknown curve {item._curve_name!r}"
        if not isinstance(target, CurveObject):
            return (
                f"{item._curve_name!r} is {noun(target)}: a draggable point can "
                "follow a plot, a parametric curve or a polar curve"
            )
        result.curves[item] = target
    result.runtime[item] = runtime
    if item.name is not None:
        binding = item._binding(runtime)
        if binding is not None:
            result.bindings[item.name] = binding
    return None


def _order(
    items: Sequence[PlotObject],
    depends: dict[PlotObject, list[str]],
    named: dict[str, PlotObject],
) -> tuple[list[PlotObject], list[PlotObject]]:
    """Order items after their dependencies (Kahn's algorithm).

    Returns:
        ``(order, stuck)``: the items that can be ordered, in a valid order
        (keeping the graph's order where there is a choice), and those that
        cannot (in or after a circular definition).
    """
    waiting = {item: {named[name] for name in depends[item]} for item in items}
    needed_by: dict[PlotObject, list[PlotObject]] = {item: [] for item in items}
    for item, needs in waiting.items():
        for dependency in needs:
            needed_by[dependency].append(item)
    position = {item: i for i, item in enumerate(items)}
    ready = sorted(
        (item for item in items if not waiting[item]), key=position.__getitem__
    )
    order: list[PlotObject] = []
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
    item: PlotObject,
    depends: dict[PlotObject, list[str]],
    named: dict[str, PlotObject],
    stuck: list[PlotObject],
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
