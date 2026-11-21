"""Items that only define names: parameters and function definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from pygridplot.core.expr import (
    RESERVED,
    Binding,
    FunctionBinding,
    ValueBinding,
    parse,
)
from pygridplot.core.expr.parser import Call, Name
from pygridplot.graph.errors import GraphError
from pygridplot.graph.items.base import (
    Coordinate,
    Formula,
    Item,
    Maker,
    check_name,
    getter,
    show,
)


class ParameterItem(Item):
    """A named value that other items use: ``a = 2``.

    The value is a number, or an expression of other names (``"2 b"``).
    Change it at any time (``graph["a"].value = 3``): everything using it
    follows. A number can be changed every frame without recompiling.

    Args:
        name: The parameter's name.
        value: A number, or an expression without variables.
    """

    kind: ClassVar[str] = "parameter"

    def __init__(self, name: str, value: Coordinate) -> None:
        super().__init__(name=check_name(name))
        self._value = Formula.make(value, f"the value of {name}", numbers=True)

    @property
    def name(self) -> str:
        """The parameter's name."""
        assert self._name is not None
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        Item.name.fset(self, check_name(name))  # type: ignore[attr-defined]

    @property
    def value(self) -> Coordinate:
        """The value as given: a number or an expression."""
        return self._value.source  # type: ignore[return-value]

    @value.setter
    def value(self, value: Coordinate) -> None:
        formula = Formula.make(value, f"the value of {self.name}", numbers=True)
        was_text, self._value = self._value.is_text, formula
        if was_text or formula.is_text:
            self._changed()  # a number is read live; an expression is compiled

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        return [(self._value, ())]

    def _runtime(self, make: Maker) -> Any:
        if not self._value.is_text:
            return lambda: float(self._value.source)  # type: ignore[arg-type]
        return getter(make(self._value, ()))

    def _binding(self, runtime: Any) -> Binding | None:
        return ValueBinding(runtime)

    def _describe(self) -> str:
        return f"{self._name} = {show(self._value)}"


class DefinitionItem(Item):
    """A function defined for other items, not drawn: ``g(u, v) = u v + 1``.

    Args:
        name: The function's name.
        parameters: The names of its arguments.
        expression: Its value, as text in the parameters.
    """

    kind: ClassVar[str] = "definition"

    def __init__(self, name: str, parameters: Sequence[str], expression: str) -> None:
        super().__init__(name=check_name(name))
        self._parameters = _parameters(parameters)
        self._expression = Formula.make(expression, f"the definition of {name}")

    @property
    def name(self) -> str:
        """The function's name."""
        assert self._name is not None
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        Item.name.fset(self, check_name(name))  # type: ignore[attr-defined]

    @property
    def parameters(self) -> tuple[str, ...]:
        """The names of the function's arguments."""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: Sequence[str]) -> None:
        self._parameters = _parameters(parameters)
        self._changed()

    @property
    def expression(self) -> str:
        """The function's value, as text in its parameters."""
        return self._expression.source  # type: ignore[return-value]

    @expression.setter
    def expression(self, expression: str) -> None:
        self._expression = Formula.make(expression, f"the definition of {self.name}")
        self._changed()

    def _formulas(self) -> list[tuple[Formula, tuple[str, ...]]]:
        return [(self._expression, self._parameters)]

    def _runtime(self, make: Maker) -> Any:
        return make(self._expression, self._parameters)

    def _binding(self, runtime: Any) -> Binding | None:
        return FunctionBinding(runtime, len(self._parameters))

    def _describe(self) -> str:
        return (
            f"{self._name}({', '.join(self._parameters)}) = {self._expression.source}"
        )


def _parameters(parameters: object) -> tuple[str, ...]:
    """Validate the parameters of a definition."""
    if isinstance(parameters, str) or not isinstance(parameters, Sequence):
        msg = f"parameters must be a sequence of names, got {parameters!r}"
        raise TypeError(msg)
    names = tuple(parameters)
    if not names:
        msg = "a definition needs at least one parameter"
        raise GraphError(msg)
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or name in RESERVED
            or not (
                (name[0].isalpha() or name[0] == "_")
                and all(c.isalpha() or c in "0123456789" or c == "_" for c in name[1:])
            )
        ):
            msg = f"{name!r} is not a valid parameter name"
            raise GraphError(msg)
    if len(set(names)) != len(names):
        msg = f"the parameters {names!r} repeat a name"
        raise GraphError(msg)
    return names


def parse_signature(signature: str) -> tuple[str, tuple[str, ...]]:
    """Read a definition's signature, ``"g(u, v)"``.

    Args:
        signature: The name and parameters.

    Returns:
        ``(name, parameters)``.

    Raises:
        GraphError: If it is not a name with parameter names in parentheses.
    """
    node = parse(signature)
    if (
        not isinstance(node, Call)
        or node.primes
        or not all(isinstance(arg, Name) for arg in node.args)
    ):
        msg = f"a signature is a name with parameters, as in g(u, v), not {signature!r}"
        raise GraphError(msg)
    return node.name, tuple(arg.name for arg in node.args)  # type: ignore[union-attr]
