"""Reading one line of maths into an item, as a graphing calculator does.

:meth:`Graph.add <pygridplot.graph.Graph.add>` accepts lines such as
``"y = x^2"`` and decides which kind of item they are:

===============================  ==============================================
Line                             Item
===============================  ==============================================
``y = x^2`` (or ``x^2``)         a plot of ``x^2``
``f(x) = x^2``                   a plot named ``f`` (also a function, ``f(2)``)
``r = 1 + cos(θ)``               a polar curve
``(cos(t), sin(t))``             a parametric curve (it uses ``t``)
``C = (cos(t), sin(t))``         a parametric curve named ``C``
``(1, 2)`` or ``P = (a, 2a)``    a point (named ``P``)
``a = 2`` or ``b = 2a``          a parameter
``g(u, v) = u v``                a definition (a function, not drawn)
``x^2 + y^2 = 9``, ``x = 3``     an implicit curve: any other equation in x, y
===============================  ==============================================

Inequalities (``y < x^2``) are not supported yet.
"""

from __future__ import annotations

from typing import Any

from pygridplot.core.expr import ExpressionError, parse_equation, references
from pygridplot.core.expr.parser import Call, Compare, Name, Node, Number, Tuple, Unary
from pygridplot.graph.items import VARIABLE_NAMES

INEQUALITY = "inequalities (shaded regions) are not supported yet"


def interpret(text: str) -> tuple[str, dict[str, Any]]:
    """Decide what kind of item a line is, and its arguments.

    Args:
        text: The line.

    Returns:
        ``(method, arguments)``: the name of the
        :class:`~pygridplot.graph.Graph` method that creates the item, and
        the arguments to call it with.

    Raises:
        ExpressionError: If the line is not valid, or not something a graph
            can show.
    """
    equation = parse_equation(text)
    lhs, rhs = equation.lhs, equation.rhs

    def source(node: Node) -> str:
        return text[node.start : node.end]

    if rhs is None:
        return _expression_line(text, lhs, source)
    assert equation.equals is not None
    right_names = references(rhs)
    left_names = references(lhs)
    if isinstance(lhs, Name) and lhs.name == "y" and "y" not in right_names:
        return "plot", {"expression": source(rhs)}
    if isinstance(rhs, Name) and rhs.name == "y" and "y" not in left_names:
        return "plot", {"expression": source(lhs)}
    if isinstance(lhs, Name) and lhs.name == "r":
        return "polar", {"r": source(rhs)}
    if isinstance(lhs, Call):
        return _definition(text, lhs, rhs, source)
    if isinstance(lhs, Name) and lhs.name not in VARIABLE_NAMES:
        if isinstance(rhs, Tuple):
            return _pair(text, rhs, source, name=lhs.name)
        if not right_names & VARIABLE_NAMES:
            return "parameter", {"name": lhs.name, "value": _number(rhs, source)}
    if isinstance(lhs, Tuple) or isinstance(rhs, Tuple):
        msg = "a point or curve cannot be one side of an equation"
        raise ExpressionError(msg, text, equation.equals)
    level = _number(rhs, source)
    if isinstance(level, float):
        return "implicit", {"expression": source(lhs), "level": level}
    return "implicit", {"expression": f"{source(lhs)} - ({source(rhs)})"}


def _expression_line(text: str, node: Node, source: Any) -> tuple[str, dict[str, Any]]:
    """Interpret a line without ``=``: a curve, a point or a plot."""
    if isinstance(node, Tuple):
        return _pair(text, node, source)
    if isinstance(node, Compare):
        raise ExpressionError(INEQUALITY, text, node.start)
    if "y" in references(node):
        msg = "an expression in x and y needs an equation, such as ... = 0"
        raise ExpressionError(msg, text, node.start)
    return "plot", {"expression": source(node)}


def _pair(
    text: str, node: Tuple, source: Any, name: str | None = None
) -> tuple[str, dict[str, Any]]:
    """Interpret ``(a, b)``: a parametric curve if it uses ``t``, else a point."""
    if len(node.items) != 2:
        msg = f"a point or curve has two coordinates, not {len(node.items)}"
        raise ExpressionError(msg, text, node.start)
    first, second = node.items
    arguments: dict[str, Any] = {} if name is None else {"name": name}
    if "t" in references(node):
        return "parametric", {**arguments, "x": source(first), "y": source(second)}
    position = (_number(first, source), _number(second, source))
    return "point", {**arguments, "position": position}


def _definition(
    text: str, lhs: Call, rhs: Node, source: Any
) -> tuple[str, dict[str, Any]]:
    """Interpret ``f(x) = ...`` (a named plot) or ``g(u, v) = ...`` (a definition)."""
    if lhs.primes or not all(isinstance(arg, Name) for arg in lhs.args):
        msg = "the left side of a definition is a name with parameters, as in f(x)"
        raise ExpressionError(msg, text, lhs.start)
    parameters = [arg.name for arg in lhs.args]  # type: ignore[union-attr]
    if parameters == ["x"]:
        return "plot", {"expression": source(rhs), "name": lhs.name}
    return "define", {"signature": source(lhs), "expression": source(rhs)}


def _number(node: Node, source: Any) -> float | str:
    """Return a node's value if it is a plain number (``2``, ``-1.5``), or its text."""
    if isinstance(node, Number):
        return node.value
    if isinstance(node, Unary) and isinstance(node.operand, Number):
        return node.operand.value if node.op == "+" else -node.operand.value
    return str(source(node))
