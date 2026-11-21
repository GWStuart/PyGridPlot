"""The values objects are made of, and the checks their arguments share.

Objects take Python values: functions (NumPy or ``math``) where the plane's
methods take functions, numbers where they take numbers, and, for
coordinates, either a number or a *function of nothing* that is called
whenever the object is drawn (``lambda: p.x``), so that a coordinate can
follow other objects. The functions below validate these arguments and
read them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, Final, TypeAlias, cast

import numpy as np

from pygridplot.core.plot.builders import check_range

Coordinate: TypeAlias = "float | Callable[[], float]"
"""A coordinate: a number, or a function of nothing giving it when drawn."""

Component: TypeAlias = "Callable[[Any], Any] | float"
"""A component of a curve: a function of the parameter, or a constant."""

FULL_TURN: Final = (0.0, 2 * math.pi)
"""The default range of parametric and polar curves."""


def is_number(value: object) -> bool:
    """Return whether a value is a real number (not a bool)."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def finite(value: object, what: str) -> float:
    """Validate a finite number.

    Raises:
        TypeError: If ``value`` is not a number.
        ValueError: If it is not finite.
    """
    if not is_number(value):
        msg = f"{what} must be a number, got {value!r}"
        raise TypeError(msg)
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        msg = f"{what} must be finite, got {value!r}"
        raise ValueError(msg)
    return number


def function(value: object, what: str) -> Callable[..., Any]:
    """Validate a function (anything callable).

    Raises:
        TypeError: If ``value`` is not callable.
    """
    if not callable(value):
        msg = f"{what} must be a function, got {value!r}"
        raise TypeError(msg)
    return value


def component(value: object, what: str) -> Component:
    """Validate a curve's component: a function, or a finite number."""
    if callable(value):
        return value
    return finite(value, what)


def coordinate(value: object, what: str) -> Coordinate:
    """Validate a coordinate: a finite number, or a function of nothing."""
    if callable(value):
        return value
    return finite(value, what)


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


def as_float(value: Any) -> float:
    """Convert a value to a float (NaN if it is not a finite number)."""
    try:
        number = float(np.asarray(value, dtype=np.float64))
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def call_quietly(func: Callable[[], Any]) -> Any:
    """Call a function of nothing; an exception gives NaN, as for plots."""
    try:
        return func()
    except Exception:  # a program's function must not break a frame
        return math.nan


def current(value: Coordinate) -> float:
    """Return a coordinate's value now: the number, or what its function gives."""
    if callable(value):
        return as_float(call_quietly(value))
    return float(value)


def show(value: Any) -> str:
    """Describe a value briefly: its number, or its function's name."""
    if callable(value):
        return getattr(value, "__name__", "<function>")
    return f"{value:g}"
