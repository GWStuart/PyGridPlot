"""Compiling syntax trees into NumPy functions.

A compiled expression is a tree of small closures, one per operation, each
calling a NumPy ufunc. So it works on arrays as fast as hand-written NumPy
code, never raises for values outside a function's domain (they become
NaN), and passes Taylor arithmetic through untouched: compiled expressions
are differentiated exactly by :func:`pygridplot.derivative`.

Names are looked up when compiling, in this order:

1. the expression's variables (``x``, or ``x`` and ``y``, ...);
2. the built-in constants and functions (:mod:`.builtins`);
3. the caller's *bindings*: :class:`ValueBinding` (a number that may change
   between calls, read each time the expression is evaluated, such as a
   parameter or a draggable point's coordinate), :class:`FunctionBinding`
   and :class:`PointBinding` (for ``P.x`` and ``P.y``).

Unknown names are errors, with a suggestion when one is likely (``sinx``
suggests ``sin(x)``, ``xy`` suggests ``x*y``). Sub-expressions that are
constant are evaluated once, when compiling. ``f'(x)`` compiles to the
exact derivative of ``f`` (any number of primes), and ``a(x + 1)``, where
``a`` is a number rather than a function, to a multiplication.
"""

from __future__ import annotations

import difflib
import inspect
import operator
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final, TypeAlias

import numpy as np

from pygridplot.core.expr.builtins import CONSTANTS, FUNCTIONS, Builtin, truth
from pygridplot.core.expr.errors import ExpressionError
from pygridplot.core.expr.parser import (
    Attribute,
    Binary,
    Call,
    Compare,
    Name,
    Node,
    Number,
    Tuple,
    Unary,
    parse,
)
from pygridplot.core.plot.derivatives import derivative
from pygridplot.core.plot.jets import Jet

MAX_PRIMES: Final = 10
"""The most primes a derivative may have (``f''''''''''(x)``)."""


@dataclass(frozen=True, slots=True)
class ValueBinding:
    """A name for a number that may change: read every time it is used.

    Attributes:
        get: Returns the current value.
    """

    get: Callable[[], Any]


@dataclass(frozen=True, slots=True)
class ConstantBinding:
    """A name for a fixed number (folded into the expression when compiling).

    Attributes:
        value: The number.
    """

    value: float


@dataclass(frozen=True, slots=True)
class FunctionBinding:
    """A name for a function.

    Attributes:
        func: The function, taking and returning arrays or numbers.
        arity: How many arguments it takes, checked when compiling, or
            ``None`` if it varies (the function is then called with whatever
            it is given). :meth:`of` finds it for a Python function.
    """

    func: Callable[..., Any]
    arity: int | None

    @classmethod
    def of(cls, func: Callable[..., Any]) -> FunctionBinding:
        """Bind a Python function, finding how many arguments it takes.

        The count comes from a ufunc's ``nin`` or the function's signature;
        it is unknown (``None``) for functions taking a varying number of
        arguments, or whose signature cannot be read.

        Args:
            func: The function.

        Returns:
            The binding.
        """
        return cls(func, _arity(func))


def _arity(func: Callable[..., Any]) -> int | None:
    """Return the number of positional arguments a function takes, if fixed."""
    if isinstance(func, np.ufunc):
        return int(func.nin)
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return None
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    required = optional = 0
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return None
        if parameter.kind in positional:
            if parameter.default is inspect.Parameter.empty:
                required += 1
            else:
                optional += 1
        elif parameter.default is inspect.Parameter.empty:
            return None  # a required keyword-only argument: not callable from text
    return required if optional == 0 else None


@dataclass(frozen=True, slots=True)
class PointBinding:
    """A name for a point, whose coordinates are ``P.x`` and ``P.y``.

    Attributes:
        get: Returns the current ``(x, y)``.
    """

    get: Callable[[], tuple[Any, Any]]


Binding: TypeAlias = ValueBinding | ConstantBinding | FunctionBinding | PointBinding
"""What a name can stand for."""

_Run: TypeAlias = Callable[[tuple[Any, ...]], Any]

_ARITHMETIC: Final[dict[str, Callable[[Any, Any], Any]]] = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.true_divide,
    "^": np.power,
}


def _comparison(test: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    """Make a comparison giving 1 (true), 0 (false) or NaN (undefined)."""

    def compare(a: Any, b: Any) -> Any:
        return truth(test(a, b), np.logical_or(np.isnan(a), np.isnan(b)))

    return compare


_COMPARISON: Final[dict[str, Callable[[Any, Any], Any]]] = {
    "<": _comparison(np.less),
    "<=": _comparison(np.less_equal),
    ">": _comparison(np.greater),
    ">=": _comparison(np.greater_equal),
    "==": _comparison(np.equal),
    "!=": _comparison(np.not_equal),
}

_GROUPS: Final = ({"+", "-"}, {"*", "/"})
"""Operators of equal precedence, whose chains are evaluated in a loop."""


class CompiledExpression:
    """An expression compiled into a function of its variables.

    Call it with one value (a number or an array) per variable; the result
    has the shape of the arguments (a constant expression is broadcast to
    it), with NaN wherever the expression is undefined. NumPy's warnings
    are silenced, as undefined values are expected.

    Attributes:
        text: The expression's text.
        variables: The names of its variables, in calling order.
    """

    __slots__ = ("_constant", "_run", "text", "variables")

    def __init__(
        self, text: str, variables: tuple[str, ...], run: _Run, constant: bool
    ) -> None:
        self.text = text
        self.variables = variables
        self._run = run
        self._constant = constant

    @property
    def constant(self) -> bool:
        """Whether the value never changes (no variables, parameters or points)."""
        return self._constant

    def __call__(self, *args: Any) -> Any:
        """Evaluate the expression.

        Args:
            *args: One value per variable.

        Returns:
            The value(s).

        Raises:
            TypeError: If the number of arguments is wrong.
        """
        if len(args) != len(self.variables):
            names = ", ".join(self.variables) or "no variables"
            msg = f"{self.text!r} takes {len(self.variables)} argument(s) ({names})"
            raise TypeError(msg)
        with np.errstate(all="ignore"):
            result = self._run(args)
        return _broadcast(result, args)

    def __repr__(self) -> str:
        """Return the text and variables."""
        return f"CompiledExpression({self.text!r}, variables={self.variables!r})"


def compile_expression(
    expression: str | Node,
    variables: Sequence[str] = (),
    bindings: Mapping[str, Binding] | None = None,
    *,
    text: str | None = None,
) -> CompiledExpression:
    """Compile an expression into a function of its variables.

    Args:
        expression: The text, or an already parsed syntax tree.
        variables: The names of the variables, in calling order.
        bindings: What other names stand for (see the module documentation).
        text: The text a syntax tree was parsed from, for error messages
            (taken from ``expression`` when that is the text).

    Returns:
        The compiled expression.

    Raises:
        ExpressionError: For a syntax error, an unknown name, a wrong number
            of arguments, or a name used as the wrong kind of thing.
    """
    if isinstance(expression, str):
        source = expression if text is None else text
        node = parse(expression)
    else:
        source = text or ""
        node = expression
    compiler = _Compiler(source, tuple(variables), bindings or {})
    run, value = compiler.compile(node)
    if value is not _VARYING:
        constant = value
        return CompiledExpression(source, tuple(variables), lambda a: constant, True)
    return CompiledExpression(source, tuple(variables), run, False)


class _Varying:
    """Marks a piece of an expression whose value is not known when compiling."""

    def __repr__(self) -> str:
        return "<varying>"


_VARYING: Final = _Varying()


class _Compiler:
    """Compiles the nodes of one expression."""

    def __init__(
        self, text: str, variables: tuple[str, ...], bindings: Mapping[str, Binding]
    ) -> None:
        self.text = text
        self.variables = variables
        self.bindings = bindings
        for name in variables:
            if name in CONSTANTS or name in FUNCTIONS:
                msg = f"{name!r} is a built-in name and cannot be a variable"
                raise ExpressionError(msg)

    def fail(self, message: str, node: Node) -> ExpressionError:
        return ExpressionError(message, self.text, node.start)

    def compile(self, node: Node) -> tuple[_Run, Any]:
        """Return a node's evaluator and, if it is constant, its value."""
        if isinstance(node, Number):
            return _constant(np.float64(node.value))
        if isinstance(node, Name):
            return self.name(node)
        if isinstance(node, Attribute):
            return self.attribute(node)
        if isinstance(node, Call):
            return self.call(node)
        if isinstance(node, Unary):
            run, value = self.compile(node.operand)
            if node.op == "+":
                return run, value
            return self.apply(np.negative, [(run, value)])
        if isinstance(node, Binary):
            if node.op == "^":
                return self.apply(
                    np.power, [self.compile(node.left), self.compile(node.right)]
                )
            return self.chain(node)
        if isinstance(node, Compare):
            return self.apply(
                _COMPARISON[node.op],
                [self.compile(node.left), self.compile(node.right)],
            )
        assert isinstance(node, Tuple)
        raise self.fail(
            "a point (a, b) cannot be used inside an expression "
            "(use its coordinates instead)",
            node,
        )

    # --- Names ------------------------------------------------------------
    def name(self, node: Name) -> tuple[_Run, Any]:
        name = node.name
        if name in self.variables:
            return operator.itemgetter(self.variables.index(name)), _VARYING
        if name in CONSTANTS:
            return _constant(np.float64(CONSTANTS[name]))
        binding = self.bindings.get(name)
        if isinstance(binding, ConstantBinding):
            return _constant(np.float64(binding.value))
        if isinstance(binding, ValueBinding):
            get = binding.get
            return (lambda a: get()), _VARYING
        if isinstance(binding, FunctionBinding) or name in FUNCTIONS:
            raise self.fail(f"{name!r} is a function: call it, as in {name}(x)", node)
        if isinstance(binding, PointBinding):
            raise self.fail(
                f"{name!r} is a point: use its coordinates, {name}.x and {name}.y", node
            )
        raise self.fail(self.unknown(name), node)

    def attribute(self, node: Attribute) -> tuple[_Run, Any]:
        if node.attribute not in ("x", "y"):
            raise self.fail(
                f"points have coordinates .x and .y, not .{node.attribute}", node
            )
        binding = self.bindings.get(node.target)
        if not isinstance(binding, PointBinding):
            if (
                binding is None
                and node.target not in self.variables
                and (node.target not in CONSTANTS and node.target not in FUNCTIONS)
            ):
                raise self.fail(self.unknown(node.target), node)
            raise self.fail(f"{node.target!r} is not a point", node)
        get = binding.get
        index = 0 if node.attribute == "x" else 1
        return (lambda a: get()[index]), _VARYING

    def call(self, node: Call) -> tuple[_Run, Any]:
        name = node.name
        binding = self.bindings.get(name)
        if (
            name in self.variables
            or name in CONSTANTS
            or isinstance(binding, (ValueBinding, ConstantBinding))
        ):
            # a(x + 1) with a number a: a multiplication.
            if node.primes:
                raise self.fail(
                    f"{name!r} is not a function: it has no derivative", node
                )
            if len(node.args) != 1:
                raise self.fail(f"{name!r} is not a function", node)
            factor = self.name(Name(name, node.start, node.start + len(name)))
            return self.apply(np.multiply, [factor, self.compile(node.args[0])])
        if isinstance(binding, PointBinding):
            raise self.fail(f"{name!r} is a point, not a function", node)
        if isinstance(binding, FunctionBinding):
            func, low, high = binding.func, binding.arity, binding.arity
        elif name in FUNCTIONS:
            builtin: Builtin = FUNCTIONS[name]
            func, low, high = builtin.func, builtin.min_args, builtin.max_args
        else:
            raise self.fail(self.unknown(name), node)
        count = len(node.args)
        if (low is not None and count < low) or (high is not None and count > high):
            raise self.fail(f"{name} takes {_arguments(low, high)}, not {count}", node)
        if node.primes:
            if count != 1 or (low is not None and low > 1):
                raise self.fail(
                    "only functions of one variable can be differentiated with "
                    f"primes ({name}{chr(39) * node.primes})",
                    node,
                )
            if node.primes > MAX_PRIMES:
                raise self.fail(f"at most {MAX_PRIMES} primes are supported", node)
            func = _derivative(func, node.primes)
        return self.apply(func, [self.compile(arg) for arg in node.args])

    # --- Operations -------------------------------------------------------
    def chain(self, node: Binary) -> tuple[_Run, Any]:
        """Compile ``a op b op c ...`` of one precedence, without recursing."""
        group = next(g for g in _GROUPS if node.op in g)
        steps: list[tuple[str, Node]] = []
        current: Node = node
        while isinstance(current, Binary) and current.op in group:
            steps.append((current.op, current.right))
            current = current.left
        first = self.compile(current)
        rest = [(op, self.compile(right)) for op, right in reversed(steps)]
        if first[1] is not _VARYING and all(v is not _VARYING for _, (_, v) in rest):
            value = first[1]
            with np.errstate(all="ignore"):
                for op, (_, operand) in rest:
                    value = _ARITHMETIC[op](value, operand)
            return _constant(value)
        start = first[0]
        operations = [(_ARITHMETIC[op], run) for op, (run, _) in rest]

        def run(a: tuple[Any, ...]) -> Any:
            value = start(a)
            for function, operand in operations:
                value = function(value, operand(a))
            return value

        return run, _VARYING

    def apply(
        self, func: Callable[..., Any], pieces: list[tuple[_Run, Any]]
    ) -> tuple[_Run, Any]:
        """Compile a call of ``func`` on pieces, folding it if they are constant."""
        if all(value is not _VARYING for _, value in pieces):
            with np.errstate(all="ignore"):
                try:
                    return _constant(func(*(value for _, value in pieces)))
                except Exception:
                    return _constant(np.float64(np.nan))
        runs = [run for run, _ in pieces]
        if len(runs) == 1:
            (only,) = runs
            return (lambda a: func(only(a))), _VARYING
        if len(runs) == 2:
            left, right = runs
            return (lambda a: func(left(a), right(a))), _VARYING
        return (lambda a: func(*(run(a) for run in runs))), _VARYING

    # --- Errors -----------------------------------------------------------
    def unknown(self, name: str) -> str:
        """Describe an unknown name, with a suggestion if one is likely."""
        values = set(self.variables) | set(CONSTANTS)
        functions = set(FUNCTIONS)
        for key, binding in self.bindings.items():
            if isinstance(binding, FunctionBinding):
                functions.add(key)
            else:
                values.add(key)
        hint = _split(name, frozenset(values), frozenset(functions))
        if hint is None:
            close = difflib.get_close_matches(name, sorted(values | functions), 1, 0.75)
            hint = close[0] if close else None
        return f"unknown name {name!r}" + (f" (did you mean {hint}?)" if hint else "")


def _constant(value: Any) -> tuple[_Run, Any]:
    """A piece whose value is known when compiling."""
    return (lambda a: value), value


def _arguments(low: int | None, high: int | None) -> str:
    """Say how many arguments a function takes."""
    if low == high:
        return f"{low} argument" + ("" if low == 1 else "s")
    if high is None:
        return f"at least {low} arguments"
    return f"{low} to {high} arguments"


@lru_cache(maxsize=256)
def _derivative_of_builtin(func: Callable[..., Any], order: int) -> Callable[..., Any]:
    return derivative(func, order)


def _derivative(func: Callable[..., Any], order: int) -> Callable[..., Any]:
    """Return the derivative of a function (cached for built-ins)."""
    if any(func is builtin.func for builtin in FUNCTIONS.values()):
        return _derivative_of_builtin(func, order)
    return derivative(func, order)


def _broadcast(result: Any, args: tuple[Any, ...]) -> Any:
    """Give a constant result the shape of the arguments.

    The samplers expect one value per point; a constant expression (or one
    whose value does not depend on the variables) gives a single number,
    which is spread over the arguments' shape. Jets are left alone: the
    derivative code handles constants itself.
    """
    if isinstance(result, Jet) or any(isinstance(arg, Jet) for arg in args):
        return result
    if np.ndim(result) != 0:
        return result
    shapes = [np.shape(arg) for arg in args if np.ndim(arg) > 0]
    if not shapes:
        return result
    return np.full(np.broadcast_shapes(*shapes), result, dtype=np.float64)


def _split(name: str, values: frozenset[str], functions: frozenset[str]) -> str | None:
    """Suggest how a run-together name may have been meant.

    ``xy`` may be ``x*y``, ``2pi`` cannot occur (numbers end names), and
    ``sinx`` may be ``sin(x)``.

    Args:
        name: The unknown name.
        values: Known names of values.
        functions: Known names of functions.

    Returns:
        The suggestion, or ``None``.
    """
    parts = _partition(name, values)
    if parts is not None and len(parts) > 1:
        return "*".join(parts)
    for function in sorted(functions, key=len, reverse=True):
        if name.startswith(function) and len(name) > len(function):
            rest = _partition(name[len(function) :], values)
            if rest is not None:
                return f"{function}({'*'.join(rest)})"
    return None


def _partition(name: str, known: Iterable[str]) -> list[str] | None:
    """Split a name into known names, preferring longer pieces.

    Args:
        name: The name to split.
        known: The names allowed as pieces.

    Returns:
        The pieces, or ``None`` if the name cannot be split.
    """
    pieces = sorted(set(known), key=len, reverse=True)
    best: dict[int, list[str] | None] = {len(name): []}
    for i in range(len(name) - 1, -1, -1):
        best[i] = None
        for piece in pieces:
            tail = best.get(i + len(piece))
            if name.startswith(piece, i) and tail is not None:
                best[i] = [piece, *tail]
                break
    return best[0]
