"""Turning an expression's text into a function, for use without a graph.

:func:`expression` is the shortest way from text to a function that a
plane can plot, or that NumPy code can call::

    f = pgp.expression("x^2 sin(x) / 4")
    plane.plot(screen, f)
    g = pgp.expression("a x + b", a=2, b=-1)
    F = pgp.expression("x^2 + y^2")           # a function of x and y
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Final

from pygridplot.core.expr.compiler import (
    Binding,
    CompiledExpression,
    ConstantBinding,
    FunctionBinding,
    PointBinding,
    compile_expression,
)
from pygridplot.core.expr.parser import parse, references

STANDARD_VARIABLES: Final = ("x", "y", "t", "θ", "theta")
"""The variables found automatically, in calling order."""


def expression(text: str, *variables: str, **names: Any) -> CompiledExpression:
    """Compile an expression into a NumPy function of its variables.

    The result can be plotted (``plane.plot(screen, expression("x^2"))``),
    called on numbers or arrays, and differentiated exactly with
    :func:`pygridplot.derivative`. See :mod:`pygridplot.core.expr` for the
    language: ``2x``, ``a sin(x)``, ``x^2``, ``sqrt``, ``ln``, ``if``,
    ``pi``, and so on.

    Args:
        text: The expression, such as ``"a sin(x - 1)"``.
        *variables: The function's arguments, in order. By default, those of
            ``x``, ``y``, ``t``, ``θ`` and ``theta`` that the expression uses
            (``x`` alone if it uses none).
        **names: Values for the other names: numbers (``a=2``), functions
            (``f=np.sinc``, called as ``f(x)``) or points (``P=(1, 2)``,
            used as ``P.x`` and ``P.y``).

    Returns:
        The function.

    Raises:
        ExpressionError: If the text is not a valid expression, or uses a
            name that is not given.
        TypeError: If a value in ``names`` is not a number, function or point.
    """
    node = parse(text)
    if not variables:
        used = references(node) - set(names)
        found = tuple(v for v in STANDARD_VARIABLES if v in used)
        variables = found or ("x",)
    bindings: dict[str, Binding] = {}
    for name, value in names.items():
        bindings[name] = _binding(name, value)
    return compile_expression(node, variables, bindings, text=text)


def _binding(name: str, value: Any) -> Binding:
    """Return what a value given for a name stands for."""
    if callable(value):
        func: Callable[..., Any] = value
        return FunctionBinding.of(func)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ConstantBinding(float(value))
    if isinstance(value, tuple) and len(value) == 2:
        x, y = float(value[0]), float(value[1])
        if math.isfinite(x) and math.isfinite(y):
            return PointBinding(lambda: (x, y))
    msg = f"{name} must be a number, a function or an (x, y) point, got {value!r}"
    raise TypeError(msg)
