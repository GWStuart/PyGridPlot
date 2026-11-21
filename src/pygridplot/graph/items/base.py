"""What every item of a graph shares: names, formulas and styles.

* :class:`Formula`: one value of an item: an expression's text (parsed at
  once), a Python function, or a number.
* :class:`Item`: the base of every item: a name, visibility and the error
  that stopped it being drawn.
* :class:`StyledItem` and its kinds (:class:`LineItem`,
  :class:`PointLikeItem`): items drawn with a style, with a palette slot
  for their default colour.

The helpers below the classes validate the arguments common to several
kinds of item.

**The items' interface to the graph.** Besides their public attributes,
items have a few methods and attributes starting with ``_`` that the rest
of :mod:`pygridplot.graph` (``graph.py``, ``namespace.py``) uses and
programs do not: ``_formulas``, ``_runtime``, ``_binding`` and
``_describe`` (what an item compiles and stands for, see :class:`Item`),
``_changed`` (it tells the graph to compile again), and ``_graph`` and
``_error``, which the graph keeps up to date. They are private to the
package, not to this module: a new kind of item implements them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, Final, TypeAlias, cast

import numpy as np

from pygridplot.core.expr import (
    CONSTANTS,
    FUNCTIONS,
    RESERVED,
    Binding,
    ExpressionError,
    parse,
    references,
)
from pygridplot.core.expr.parser import Node, Tuple
from pygridplot.core.plot.builders import check_range
from pygridplot.core.plot.style import (
    LineStyle,
    PointStyle,
    PolygonStyle,
    with_overrides,
)
from pygridplot.graph.errors import GraphError

if TYPE_CHECKING:
    from pygridplot.graph.graph import Graph


Expression: TypeAlias = "str | Callable[..., Any]"
"""An expression: text in the expression language, or a Python function."""


Coordinate: TypeAlias = "float | str"
"""A coordinate: a number, or an expression without variables (``"a + 1"``)."""


VARIABLE_NAMES: Final = frozenset({"x", "y", "t", "θ", "theta"})
"""Names reserved for the variables of expressions."""


FULL_TURN: Final = (0.0, 2 * math.pi)


Maker: TypeAlias = Callable[["Formula", tuple[str, ...]], Any]
"""Compiles a formula with the given variables (supplied by the graph)."""


def check_name(name: object, what: str = "a name") -> str:
    """Validate a name for an item or a parameter of a definition.

    A name starts with a letter or ``_`` and goes on with letters, ASCII
    digits and ``_`` (``a``, ``f2``, ``speed_max``, ``θ1``). It cannot be a
    built-in constant or function, nor a variable name (x, y, t, θ, theta).

    Args:
        name: The name.
        what: What is being named, for the error message.

    Returns:
        The name.

    Raises:
        GraphError: If the name is not valid.
    """
    if not isinstance(name, str) or not name:
        msg = f"{what} must be a non-empty string, got {name!r}"
        raise GraphError(msg)
    first, rest = name[0], name[1:]
    valid = (first.isalpha() or first == "_") and all(
        c.isalpha() or c in "0123456789" or c == "_" for c in rest
    )
    if not valid:
        msg = f"{name!r} is not a valid name (letters, digits and _, not first a digit)"
        raise GraphError(msg)
    if name in RESERVED:
        msg = f"{name!r} is a built-in name and cannot be redefined"
        raise GraphError(msg)
    if name in VARIABLE_NAMES:
        msg = f"{name!r} is a variable name and cannot name an item"
        raise GraphError(msg)
    return name


class Formula:
    """One value of an item: an expression's text, a Python function or a number.

    Text is parsed when the formula is made, so syntax errors show at once.

    Attributes:
        source: What was given: the text, the function, or the number.
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
        """Make a formula from what a user gave.

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


def getter(compiled: Any) -> Callable[[], float]:
    """Return a function giving a coordinate's current value as a float."""
    if not callable(compiled):
        value = float(compiled)
        return lambda: value
    return lambda: as_float(compiled())


def as_float(value: Any) -> float:
    """Convert an expression's value to a float (NaN if it is not a number)."""
    try:
        number = float(np.asarray(value, dtype=np.float64))
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def show(formula: Formula) -> str:
    """Describe a formula briefly."""
    source = formula.source
    if callable(source):
        return getattr(source, "__name__", "<function>")
    return str(source) if isinstance(source, str) else f"{source:g}"


def component(value: object, what: str) -> Formula:
    """Make the formula of a curve's component: text, a function or a number."""
    return Formula.make(value, what, numbers=True, functions=True)


def coordinates(value: object, what: str) -> tuple[Formula, Formula]:
    """Validate a pair of coordinates (numbers or expressions)."""
    x, y = two(value, what)
    return (
        Formula.make(x, f"the x of {what}", numbers=True),
        Formula.make(y, f"the y of {what}", numbers=True),
    )


class Item:
    """A row of a graph: something drawn, or a name defined for other rows.

    Subclasses describe themselves to the graph through four hooks:
    ``_formulas()`` (their formulas, with the variables of each),
    ``_runtime(make)`` (compile them with the graph's ``make`` into what
    drawing needs), ``_binding(runtime)`` (what their name stands for in
    expressions, if anything) and ``_describe()`` (for ``repr``). Setters
    that change names or expressions call ``_changed()``.

    Attributes:
        kind: The kind of item, as written in saved graphs (``"plot"``, ...).
    """

    kind: ClassVar[str] = "item"
    colored: ClassVar[bool] = False
    """Whether the item is drawn in a colour (and so takes a palette colour)."""

    def __init__(self, *, name: str | None = None, visible: bool = True) -> None:
        self._graph: Graph | None = None
        self._name = None if name is None else check_name(name)
        self._visible = bool(visible)
        self._error: str | None = None

    @property
    def name(self) -> str | None:
        """The item's name, by which other items refer to it (``None`` if unnamed).

        Renaming updates the graph; references in other items are not
        rewritten (they then refer to the old name, and show an error).
        """
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        new = None if name is None else check_name(name)
        if self._graph is not None:
            self._graph._rename(self, new)
        self._name = new
        self._changed()

    @property
    def visible(self) -> bool:
        """Whether the item is drawn (hidden items still define their name)."""
        return self._visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        self._visible = bool(visible)

    @property
    def error(self) -> str | None:
        """Why the item cannot be drawn, or ``None`` if it can.

        For example ``"unknown name 'b'"`` or ``"circular definition: f → g
        → f"``. Errors in expressions are always up to date (reading this
        checks the graph's names if anything changed); the rare errors found
        only when drawing are those of the last :meth:`Graph.draw
        <pygridplot.Graph.draw>`.
        """
        if self._graph is not None:
            self._graph._compiled()
        return self._error

    @property
    def graph(self) -> Graph | None:
        """The graph the item belongs to (``None`` once removed)."""
        return self._graph

    def references(self) -> set[str]:
        """Return the names of other items this item refers to.

        Returns:
            The names used in its expressions (and a followed curve), besides
            variables and built-ins.
        """
        names: set[str] = set()
        for formula, variables in self._formulas():
            names |= formula.names(variables)
        return names

    # --- For the graph ----------------------------------------------------
    def _changed(self) -> None:
        """Tell the graph that names or expressions changed: recompile."""
        if self._graph is not None:
            self._graph._invalidate()

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        """Return the item's formulas and the variables of each."""
        return []

    def _runtime(self, make: Maker) -> Any:
        """Compile the item's formulas into what drawing it needs."""
        return None

    def _binding(self, runtime: Any) -> Binding | None:
        """Return what the item's name stands for in expressions, if anything."""
        return None

    def _describe(self) -> str:
        """Return a short description of the item, for errors and ``repr``."""
        return ""

    def __repr__(self) -> str:
        """Return the kind, the name and a description."""
        name = f" {self._name!r}" if self._name else ""
        detail = self._describe()
        return f"<{type(self).__name__}{name}{': ' if detail else ''}{detail}>"


StyleT: TypeAlias = "LineStyle | PointStyle | PolygonStyle"


def style_property(field: str, annotation: str, doc: str) -> property:
    """Make a property for one field of a styled item's style.

    Reading it reads the style's field; setting it replaces the style with
    one where the field is changed (styles are immutable).

    Args:
        field: The style's field.
        annotation: Its type, for the documentation.
        doc: The property's documentation.

    Returns:
        The property.
    """

    def get(self: StyledItem) -> Any:
        return getattr(self._style, field)

    def put(self: StyledItem, value: Any) -> None:
        self._set_style(**{field: value})

    get.__annotations__["return"] = annotation
    return property(get, put, doc=doc)


class StyledItem(Item):
    """An item drawn with a style: a line, a point or a polygon style.

    An item without a colour takes the colour of its *palette slot*,
    chosen when it is added (the next slot of the graph), from the plane's
    theme. So colours stay with their items when others are hidden or
    removed, and follow the theme.
    """

    colored: ClassVar[bool] = True
    style_type: ClassVar[type] = LineStyle
    default_style: ClassVar[Any] = LineStyle()

    def __init__(
        self,
        *,
        style: Any = None,
        overrides: dict[str, Any] | None = None,
        palette: int | None = None,
        name: str | None = None,
        visible: bool = True,
    ) -> None:
        super().__init__(name=name, visible=visible)
        base = self.default_style if style is None else self._check_style(style)
        self._style = with_overrides(base, **(overrides or {}))
        self._palette = self._check_palette(palette)

    @property
    def style(self) -> Any:
        """The item's style (a :class:`~pygridplot.LineStyle`, ... as for the plane)."""
        return self._style

    @style.setter
    def style(self, style: Any) -> None:
        self._style = self._check_style(style)

    @property
    def palette(self) -> int | None:
        """The palette slot giving the item its colour when ``color`` is ``None``."""
        return self._palette

    @palette.setter
    def palette(self, slot: int | None) -> None:
        self._palette = self._check_palette(slot)

    color = style_property(
        "color",
        "ColorLike | None",
        "The colour (``None`` for the colour of the item's palette slot).",
    )
    opacity = style_property("opacity", "float", "The opacity, in ``[0, 1]``.")

    def _set_style(self, **changes: Any) -> None:
        self._style = replace(self._style, **changes)

    def _check_style(self, style: Any) -> Any:
        if not isinstance(style, self.style_type):
            msg = f"style must be a {self.style_type.__name__}, got {style!r}"
            raise TypeError(msg)
        return style

    @staticmethod
    def _check_palette(slot: object) -> int | None:
        if slot is None:
            return None
        if (
            isinstance(slot, bool)
            or not isinstance(slot, (int, np.integer))
            or slot < 0
        ):
            msg = f"palette must be a non-negative whole number or None, got {slot!r}"
            raise ValueError(msg)
        return int(slot)


class LineItem(StyledItem):
    """An item drawn as lines, with a :class:`~pygridplot.LineStyle`."""

    style_type: ClassVar[type] = LineStyle
    default_style: ClassVar[Any] = LineStyle()

    width = style_property("width", "float", "The line width, in pixels.")


class PointLikeItem(StyledItem):
    """An item drawn as markers, with a :class:`~pygridplot.PointStyle`."""

    style_type: ClassVar[type] = PointStyle
    default_style: ClassVar[Any] = PointStyle()

    radius = style_property("radius", "float", "The marker size, in pixels.")
    marker = style_property(
        "marker", "MarkerShape", 'The marker shape (``"circle"``, ``"square"``, ...).'
    )
    edge_color = style_property(
        "edge_color",
        "ColorLike | None",
        "The colour of the marker's outline (``None`` for none).",
    )
    edge_width = style_property(
        "edge_width", "float", "The outline's thickness, in pixels."
    )
