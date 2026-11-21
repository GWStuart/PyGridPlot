"""The constants and functions every expression can use.

Every function is built from NumPy ufuncs, so compiled expressions work on
arrays, never raise for values outside a function's domain (they give NaN
instead), and can be differentiated exactly by
:func:`pygridplot.derivative` (the Taylor arithmetic sees every operation).
The exceptions are :func:`gamma` and :func:`factorial`, which use
:func:`math.gamma` and are differentiated numerically.

Conventions follow calculators and graphing tools rather than Python:
``log`` is the base-10 logarithm and ``ln`` the natural one; ``round``
rounds halves up; ``mod`` takes the sign of the divisor.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import numpy as np


@dataclass(frozen=True, slots=True)
class Builtin:
    """A built-in function.

    Attributes:
        func: The function, taking and returning arrays (or numbers).
        min_args: The fewest arguments it takes.
        max_args: The most (``None`` for any number).
        usage: How to call it, for documentation and error messages.
    """

    func: Callable[..., Any]
    min_args: int
    max_args: int | None
    usage: str


def _sec(x: Any) -> Any:
    return np.true_divide(1.0, np.cos(x))


def _csc(x: Any) -> Any:
    return np.true_divide(1.0, np.sin(x))


def _cot(x: Any) -> Any:
    return np.true_divide(np.cos(x), np.sin(x))


def _arctan(y: Any, x: Any = None) -> Any:
    return np.arctan(y) if x is None else np.arctan2(y, x)


def _log(x: Any, base: Any = None) -> Any:
    if base is None:
        return np.log10(x)
    return np.true_divide(np.log(x), np.log(base))


def _round(x: Any) -> Any:
    return np.floor(np.add(x, 0.5))


def _minimum(*args: Any) -> Any:
    result = args[0]
    for value in args[1:]:
        result = np.minimum(result, value)
    return result


def _maximum(*args: Any) -> Any:
    result = args[0]
    for value in args[1:]:
        result = np.maximum(result, value)
    return result


def _clamp(x: Any, low: Any, high: Any) -> Any:
    return np.minimum(np.maximum(x, low), high)


# --- Logic ------------------------------------------------------------------
# Truth values are numbers: 1 (true) and 0 (false); any other number counts
# as true. An undefined (NaN) input makes the result undefined, so a
# condition on an undefined value never silently counts as false.


def truth(values: Any, undefined: Any) -> Any:
    """Return 1.0 where ``values`` hold, 0.0 where not, NaN where ``undefined``.

    Args:
        values: Booleans (the outcome of a test).
        undefined: Booleans: where an input was undefined.

    Returns:
        The truth values, as floats.
    """
    return np.where(undefined, np.nan, np.where(values, 1.0, 0.0))


def _undefined(*args: Any) -> Any:
    """Return where any of ``args`` is NaN."""
    result = np.isnan(args[0])
    for value in args[1:]:
        result = np.logical_or(result, np.isnan(value))
    return result


def _if(condition: Any, then: Any, otherwise: Any = math.nan) -> Any:
    chosen = np.where(np.not_equal(condition, 0.0), then, otherwise)
    return np.where(np.isnan(condition), np.nan, chosen)


def _and(*args: Any) -> Any:
    result = np.not_equal(args[0], 0.0)
    for value in args[1:]:
        result = np.logical_and(result, np.not_equal(value, 0.0))
    return truth(result, _undefined(*args))


def _or(*args: Any) -> Any:
    result = np.not_equal(args[0], 0.0)
    for value in args[1:]:
        result = np.logical_or(result, np.not_equal(value, 0.0))
    return truth(result, _undefined(*args))


def _not(a: Any) -> Any:
    return truth(np.equal(a, 0.0), np.isnan(a))


def _gamma_of(x: float) -> float:
    """Return Γ(x): NaN at the poles (0, -1, -2, ...), infinity past 171.6."""
    try:
        return math.gamma(x)
    except ValueError:  # a pole, or -inf
        return math.nan
    except OverflowError:
        return math.inf


_gamma_scalar = np.vectorize(_gamma_of, otypes=[np.float64])


def _gamma(x: Any) -> Any:
    return _gamma_scalar(x)


def _factorial(x: Any) -> Any:
    return _gamma_scalar(np.add(x, 1.0))


CONSTANTS: Final[dict[str, float]] = {
    "pi": math.pi,
    "π": math.pi,
    "tau": math.tau,
    "τ": math.tau,
    "e": math.e,
}
"""Built-in constants."""


def _unary(func: Callable[..., Any], usage: str) -> Builtin:
    return Builtin(func, 1, 1, usage)


FUNCTIONS: Final[dict[str, Builtin]] = {
    # Trigonometry
    "sin": _unary(np.sin, "sin(x)"),
    "cos": _unary(np.cos, "cos(x)"),
    "tan": _unary(np.tan, "tan(x)"),
    "sec": _unary(_sec, "sec(x)"),
    "csc": _unary(_csc, "csc(x)"),
    "cot": _unary(_cot, "cot(x)"),
    "arcsin": _unary(np.arcsin, "arcsin(x)"),
    "asin": _unary(np.arcsin, "asin(x)"),
    "arccos": _unary(np.arccos, "arccos(x)"),
    "acos": _unary(np.arccos, "acos(x)"),
    "arctan": Builtin(_arctan, 1, 2, "arctan(x) or arctan(y, x)"),
    "atan": Builtin(_arctan, 1, 2, "atan(x) or atan(y, x)"),
    "sinh": _unary(np.sinh, "sinh(x)"),
    "cosh": _unary(np.cosh, "cosh(x)"),
    "tanh": _unary(np.tanh, "tanh(x)"),
    "arcsinh": _unary(np.arcsinh, "arcsinh(x)"),
    "asinh": _unary(np.arcsinh, "asinh(x)"),
    "arccosh": _unary(np.arccosh, "arccosh(x)"),
    "acosh": _unary(np.arccosh, "acosh(x)"),
    "arctanh": _unary(np.arctanh, "arctanh(x)"),
    "atanh": _unary(np.arctanh, "atanh(x)"),
    # Powers and logarithms
    "exp": _unary(np.exp, "exp(x)"),
    "ln": _unary(np.log, "ln(x)"),
    "log": Builtin(_log, 1, 2, "log(x) (base 10) or log(x, base)"),
    "log2": _unary(np.log2, "log2(x)"),
    "log10": _unary(np.log10, "log10(x)"),
    "sqrt": _unary(np.sqrt, "sqrt(x)"),
    "cbrt": _unary(np.cbrt, "cbrt(x)"),
    "hypot": Builtin(np.hypot, 2, 2, "hypot(x, y)"),
    # Rounding and pieces
    "abs": _unary(np.abs, "abs(x)"),
    "sign": _unary(np.sign, "sign(x)"),
    "floor": _unary(np.floor, "floor(x)"),
    "ceil": _unary(np.ceil, "ceil(x)"),
    "round": _unary(_round, "round(x)"),
    "mod": Builtin(np.mod, 2, 2, "mod(x, m)"),
    "min": Builtin(_minimum, 1, None, "min(a, b, ...)"),
    "max": Builtin(_maximum, 1, None, "max(a, b, ...)"),
    "clamp": Builtin(_clamp, 3, 3, "clamp(x, low, high)"),
    # Conditions
    "if": Builtin(_if, 2, 3, "if(condition, value) or if(condition, value, otherwise)"),
    "and": Builtin(_and, 2, None, "and(a, b, ...)"),
    "or": Builtin(_or, 2, None, "or(a, b, ...)"),
    "not": _unary(_not, "not(a)"),
    # Special functions
    "gamma": _unary(_gamma, "gamma(x)"),
    "factorial": _unary(_factorial, "factorial(x)"),
}
"""Built-in functions, by name."""

RESERVED: Final = frozenset(CONSTANTS) | frozenset(FUNCTIONS)
"""Names that cannot be given to anything else."""
