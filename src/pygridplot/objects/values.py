"""The values objects are made of, and the checks their arguments share.

An object's function, coordinates or parameter is a :class:`Formula`: what
the program gave (a Python function, a number, or text in the expression
language, parsed at once so that syntax errors show where they are made)
plus the parsed tree of a text. Text is compiled when the object is drawn:
with the built-in names only for an object on its own, with the names of
its graph inside a :class:`~pygridplot.Graph`.

The functions below validate the arguments several kinds of object take
(names, ranges, pairs, finite numbers).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, Final, TypeAlias, cast

import numpy as np

from pygridplot.core.expr import (
    CONSTANTS,
    FUNCTIONS,
    RESERVED,
    ExpressionError,
    compile_expression,
    parse,
    references,
)
from pygridplot.core.expr.parser import Node, Tuple
from pygridplot.core.plot.builders import check_range

Expression: TypeAlias = "str | Callable[..., Any]"
"""An expression: a Python function, or text in the expression language."""

Coordinate: TypeAlias = "float | str | Callable[[], float]"
"""A coordinate: a number, a function of nothing, or text without variables."""

VARIABLE_NAMES: Final = frozenset({"x", "y", "t", "θ", "theta"})
"""Names reserved for the variables of expressions."""

FULL_TURN: Final = (0.0, 2 * math.pi)
"""The default range of parametric and polar curves."""

Maker: TypeAlias = "Callable[[Formula, tuple[str, ...]], Any]"
"""Compiles a formula with the given variables (supplied by the owner)."""


def check_name(name: object, what: str = "a name") -> str:
    """Validate the name of an object or of a definition's parameter.

    A name starts with a letter or ``_`` and goes on with letters, ASCII
    digits and ``_`` (``a``, ``f2``, ``speed_max``, ``θ1``). It cannot be a
    built-in constant or function, nor a variable name (x, y, t, θ, theta).

    Args:
        name: The name.
        what: What is being named, for the error message.

    Returns:
        The name.

    Raises:
        ValueError: If the name is not valid.
    """
    if not isinstance(name, str) or not name:
        msg = f"{what} must be a non-empty string, got {name!r}"
        raise ValueError(msg)
    first, rest = name[0], name[1:]
    valid = (first.isalpha() or first == "_") and all(
        c.isalpha() or c in "0123456789" or c == "_" for c in rest
    )
    if not valid:
        msg = f"{name!r} is not a valid name (letters, digits and _, not first a digit)"
        raise ValueError(msg)
    if name in RESERVED:
        msg = f"{name!r} is a built-in name and cannot be redefined"
        raise ValueError(msg)
    if name in VARIABLE_NAMES:
        msg = f"{name!r} is a variable name and cannot name an object"
        raise ValueError(msg)
    return name


class Formula:
    """One value of an object: a Python function, a number or text.

    Text is parsed when the formula is made, so syntax errors show at once.

    Attributes:
        source: What was given: the function, the number, or the text.
        node: The syntax tree of a text (``None`` otherwise).
    """

    __slots__ = ("node", "source")

    def __init__(
        self, source: str | Callable[..., Any] | float, node: Node | None
    ) -> None:
        self.source = source
        self.node = node

    @classmethod
    def make(
        cls, value: object, what: str, *, numbers: bool = False, functions: bool = False
    ) -> Formula:
        """Make a formula from what a program gave.

        Args:
            value: Text, a function or a number.
            what: What the value is, for error messages.
            numbers: Whether numbers are allowed.
            functions: Whether Python functions are allowed.

        Returns:
            The formula.

        Raises:
            ExpressionError: For text with a syntax error.
            TypeError: For a value of a kind that is not allowed.
            ValueError: For a number that is not finite.
        """
        if isinstance(value, str):
            node = parse(value)
            if isinstance(node, Tuple):
                msg = f"{what} must be a single value, not a point"
                raise ExpressionError(msg, value, node.start)
            return cls(value, node)
        if functions and callable(value):
            return cls(value, None)
        if numbers and is_number(value):
            number = float(value)  # type: ignore[arg-type]
            if not math.isfinite(number):
                msg = f"{what} must be finite, got {value!r}"
                raise ValueError(msg)
            return cls(number, None)
        kinds = (
            ["text"]
            + (["a number"] if numbers else [])
            + (["a function"] if functions else [])
        )
        msg = f"{what} must be {' or '.join(kinds)}, got {value!r}"
        raise TypeError(msg)

    @property
    def is_text(self) -> bool:
        """Whether the formula is an expression's text."""
        return self.node is not None

    def names(self, variables: Iterable[str]) -> set[str]:
        """Return the names the formula refers to, besides variables and built-ins.

        Args:
            variables: The formula's variables.

        Returns:
            The names (empty for functions and numbers).
        """
        if self.node is None:
            return set()
        own = set(variables)
        return {
            n
            for n in references(self.node)
            if n not in own and n not in CONSTANTS and n not in FUNCTIONS
        }

    def __repr__(self) -> str:
        """Return the source."""
        return f"Formula({self.source!r})"


def standalone(formula: Formula, variables: tuple[str, ...]) -> Any:
    """Compile a formula of an object on its own: built-in names only.

    This is the :data:`Maker` of objects that do not belong to a graph.

    Args:
        formula: The formula.
        variables: Its variables.

    Returns:
        The compiled expression, or the function or number as given.

    Raises:
        ExpressionError: If the text uses a name that is not built in.
    """
    if formula.node is None:
        return formula.source
    return compile_expression(formula.node, variables, {}, text=str(formula.source))


def is_number(value: object) -> bool:
    """Return whether a value is a real number (not a bool)."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def two(value: object, what: str) -> tuple[Any, Any]:
    """Return the two elements of a pair (a tuple, a list, an array row).

    Raises:
        ValueError: If ``value`` does not have exactly two elements.
    """
    try:
        items: tuple[Any, ...] = tuple(cast("Iterable[Any]", value))
    except TypeError:
        items = ()
    if len(items) != 2 or isinstance(value, str):
        msg = f"{what} must be a pair of two values, got {value!r}"
        raise ValueError(msg)
    return items[0], items[1]


def finite_range(value: object, name: str) -> tuple[float, float]:
    """Validate an increasing finite range (a ``t_range``, a domain)."""
    return check_range(name, two(value, name))


def optional_setting(value: object, kind: type, name: str) -> Any:
    """Validate an optional settings object (``None`` or an instance of ``kind``)."""
    if value is not None and not isinstance(value, kind):
        msg = f"{name} must be a {kind.__name__} or None, got {value!r}"
        raise TypeError(msg)
    return value


def finite(value: object, what: str) -> float:
    """Validate a finite number."""
    if not is_number(value):
        msg = f"{what} must be a number, got {value!r}"
        raise TypeError(msg)
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        msg = f"{what} must be finite, got {value!r}"
        raise ValueError(msg)
    return number


def as_float(value: Any) -> float:
    """Convert an expression's value to a float (NaN if it is not a number)."""
    try:
        number = float(np.asarray(value, dtype=np.float64))
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def getter(compiled: Any) -> Callable[[], float]:
    """Return a function giving a compiled coordinate's current value as a float."""
    if not callable(compiled):
        value = float(compiled)
        return lambda: value
    return lambda: as_float(call_quietly(compiled))


def call_quietly(func: Callable[[], Any]) -> Any:
    """Call a function of nothing; an exception gives NaN, as for plots."""
    try:
        return func()
    except Exception:
        return math.nan


def show(source: Any) -> str:
    """Describe a value briefly: its text, its number, or its function's name."""
    if isinstance(source, Formula):
        source = source.source
    if callable(source):
        return getattr(source, "__name__", "<function>")
    return str(source) if isinstance(source, str) else f"{source:g}"


def component(value: object, what: str) -> Formula:
    """Make the formula of a curve's component: text, a function or a number."""
    return Formula.make(value, what, numbers=True, functions=True)


def coordinate(value: object, what: str) -> Formula:
    """Make the formula of a coordinate: a number, text, or a function of nothing."""
    return Formula.make(value, what, numbers=True, functions=True)
