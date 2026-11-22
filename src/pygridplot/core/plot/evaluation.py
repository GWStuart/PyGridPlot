"""Evaluating user functions robustly, without ever raising.

Plotted functions come from users and can do anything: work only on single
numbers (:mod:`math`), raise errors for parts of their domain, return NaN,
infinities or complex numbers, or return values of the wrong type or shape.
:func:`evaluate` (and :func:`evaluate_xy` for functions of two variables)
turns all of that into a float array with NaN wherever the function has no
real value, so the code that samples and draws curves never has to worry
about it. :func:`evaluate_vectors` does the same for vector fields,
functions returning a pair ``(u, v)``.

The function is first called once on the whole NumPy array (fast for
vectorised functions). If that raises, or returns something of the wrong
shape, it is called point by point instead. Complex results keep their real
part where the imaginary part is negligible and become NaN elsewhere.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

Function = Callable[[Any], Any]
"""A plottable function: it maps an x value (or an array of them) to y."""

PlaneFunction = Callable[[Any, Any], Any]
"""A function of two variables, ``F(x, y)``, as used for implicit curves."""

VectorFunction = Callable[[Any, Any], Any]
"""A vector field: maps ``(x, y)`` to a vector ``(u, v)``, as a pair."""

_COMPLEX_TOLERANCE: Final = 1e-12
"""Largest relative imaginary part that still counts as a real result."""


def evaluate(func: Function, xs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate a user function at many points without ever raising.

    Args:
        func: The function. It may be vectorised (works on NumPy arrays) or
            scalar-only (for example written with :mod:`math`).
        xs: The x values.

    Returns:
        ``f(x)`` for every x as floats, with NaN wherever the function raised,
        returned a non-number, or returned a genuinely complex value.
    """
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate_quietly(func, xs)


def evaluate_quietly(func: Function, xs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate like :func:`evaluate`, without silencing warnings itself.

    The samplers (:mod:`pygridplot.core.plot.sampling` and
    :mod:`pygridplot.core.plot.curves`) evaluate a function many times per
    frame. They silence NumPy errors and Python warnings once, around all
    of their evaluations, which is much cheaper than doing it for every
    batch. Call this only inside such a block (``np.errstate(all="ignore")``
    and ``warnings.catch_warnings()`` with ``simplefilter("ignore")``);
    otherwise use :func:`evaluate`.

    Args:
        func: The function.
        xs: The x values.

    Returns:
        ``f(x)`` for every x, with NaN where undefined.
    """
    return _evaluate_arrays(func, (xs,))


def evaluate_xy(
    func: PlaneFunction, xs: NDArray[np.float64], ys: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate a function of two variables at many points without ever raising.

    The two-variable version of :func:`evaluate`, used for implicit curves
    ``F(x, y) = 0``: the function is called once with arrays of x and y
    values, or point by point if that fails.

    Args:
        func: The function ``F(x, y)``, vectorised or scalar-only.
        xs: The x coordinates of the points.
        ys: The y coordinates of the points (the same length as ``xs``).

    Returns:
        ``F(x, y)`` for every point as floats, with NaN wherever the function
        raised, returned a non-number, or returned a genuinely complex value.
    """
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate_xy_quietly(func, xs, ys)


def evaluate_xy_quietly(
    func: PlaneFunction, xs: NDArray[np.float64], ys: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate like :func:`evaluate_xy`, without silencing warnings itself.

    See :func:`evaluate_quietly` for when to use the quiet versions.

    Args:
        func: The function ``F(x, y)``.
        xs: The x coordinates.
        ys: The y coordinates.

    Returns:
        ``F(x, y)`` for every point, with NaN where undefined.
    """
    return _evaluate_arrays(func, (xs, ys))


def _evaluate_arrays(
    func: Callable[..., Any], args: tuple[NDArray[np.float64], ...]
) -> NDArray[np.float64]:
    """Evaluate ``func(*args)``, vectorised if possible, else point by point.

    Args:
        func: The function.
        args: One array of values per argument, all the same length.

    Returns:
        The value at every point, with NaN where undefined.
    """
    n = args[0].shape[0]
    if n == 0:
        return np.zeros(0)
    try:
        values = _as_values(func(*(arg.copy() for arg in args)), n)
    except Exception:
        values = None
    if values is None:
        values = _evaluate_each(func, args)
    return values


def evaluate_vectors(
    func: VectorFunction, xs: NDArray[np.float64], ys: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate a vector field at many points without ever raising.

    The field returns a pair ``(u, v)``: a tuple or list of two arrays (or
    numbers, for a constant component, as in ``lambda x, y: (1, y)``), or an
    array of shape ``(2, N)``. It is called once with arrays of x and y
    values, or point by point if that fails or returns something unusable.

    Args:
        func: The field ``F(x, y) -> (u, v)``, vectorised or scalar-only.
        xs: The x coordinates of the points.
        ys: The y coordinates of the points (the same length as ``xs``).

    Returns:
        The vectors, shape ``(N, 2)``: ``u`` in the first column and ``v`` in
        the second, NaN wherever a component is undefined (an error, NaN, a
        complex value, or not a pair at all).
    """
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate_vectors_quietly(func, xs, ys)


def evaluate_vectors_quietly(
    func: VectorFunction, xs: NDArray[np.float64], ys: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate like :func:`evaluate_vectors`, without silencing warnings itself.

    See :func:`evaluate_quietly` for when to use the quiet versions.

    Args:
        func: The field.
        xs: The x coordinates.
        ys: The y coordinates.

    Returns:
        The vectors, shape ``(N, 2)``, NaN where undefined.
    """
    n = xs.shape[0]
    if n == 0:
        return np.zeros((0, 2))
    try:
        values = _as_vectors(func(xs.copy(), ys.copy()), n)
    except Exception:
        values = None
    if values is None:
        values = _evaluate_vectors_each(func, xs, ys)
    return values


def _as_vectors(result: object, n: int) -> NDArray[np.float64] | None:
    """Convert a vectorised field's result to an ``(n, 2)`` array, or ``None``.

    Args:
        result: What the field returned for arrays of ``n`` points.
        n: Number of points.

    Returns:
        The vectors, or ``None`` if the result is not a pair of usable
        components (the caller then evaluates point by point). Both
        components being single numbers also gives ``None``, as for
        :func:`evaluate`: point by point is then right whatever the
        function does.
    """
    if isinstance(result, np.ndarray):
        if result.ndim == 0 or result.shape[0] != 2:
            return None
        parts: tuple[object, object] = (result[0], result[1])
    elif isinstance(result, (tuple, list)) and len(result) == 2:
        parts = (result[0], result[1])
    else:
        return None
    columns = []
    constants = 0
    for part in parts:
        try:
            array = np.asarray(part)
        except Exception:
            return None
        if array.shape == ():
            constants += 1
            array = np.broadcast_to(array, (n,))
        values = _as_values(array, n)
        if values is None:
            return None
        columns.append(values)
    if constants == 2:
        return None
    return np.column_stack(columns)


def _evaluate_vectors_each(
    func: VectorFunction, xs: NDArray[np.float64], ys: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate a scalar-only vector field one point at a time.

    Args:
        func: The field.
        xs: The x coordinates.
        ys: The y coordinates.

    Returns:
        The vectors, shape ``(N, 2)``, NaN wherever the field raised or did
        not return a pair of real numbers.
    """
    out = np.full((xs.shape[0], 2), math.nan)
    for i, (x, y) in enumerate(zip(xs.tolist(), ys.tolist(), strict=True)):
        try:
            u, v = func(x, y)
            out[i] = (_scalar_value(u), _scalar_value(v))
        except Exception:
            continue
    return out


def _as_values(result: object, n: int) -> NDArray[np.float64] | None:
    """Convert a vectorised result to floats, or return ``None`` if unusable.

    Args:
        result: What the function returned for an array of ``n`` values.
        n: Number of x values.

    Returns:
        A float array of shape ``(n,)``, or ``None`` if the result is a scalar,
        has the wrong shape or cannot be converted (the caller then falls back
        to evaluating point by point).
    """
    if isinstance(result, np.ma.MaskedArray):
        dtype = np.complex128 if result.dtype.kind == "c" else np.float64
        result = np.ma.filled(result.astype(dtype), np.nan)
    try:
        array = np.asarray(result)
    except Exception:
        return None
    if array.shape != (n,):
        return None
    kind = array.dtype.kind
    if kind == "c":
        return _real_part(array)
    if kind in "biuf":
        return array.astype(np.float64)
    if kind == "O":
        try:
            return _real_part(array.astype(np.complex128))
        except (TypeError, ValueError):
            return None
    return None


def _real_part(values: NDArray[Any]) -> NDArray[np.float64]:
    """Keep the real part of nearly-real complex values; NaN for the rest.

    Args:
        values: Complex (or real) values.

    Returns:
        The real parts, with NaN where the imaginary part is not negligible.
    """
    real = np.real(values).astype(np.float64)
    imag = np.abs(np.imag(values))
    tolerance = _COMPLEX_TOLERANCE * np.maximum(1.0, np.abs(real))
    return np.where(imag <= tolerance, real, np.nan)


def _evaluate_each(
    func: Callable[..., Any], args: tuple[NDArray[np.float64], ...]
) -> NDArray[np.float64]:
    """Evaluate a scalar-only function one point at a time.

    Args:
        func: The function.
        args: One array of values per argument.

    Returns:
        The values, with NaN wherever the function raised or returned
        something that is not a real number.
    """
    out = np.empty(args[0].shape[0])
    for i, point in enumerate(zip(*(arg.tolist() for arg in args), strict=True)):
        try:
            out[i] = _scalar_value(func(*point))
        except Exception:
            out[i] = math.nan
    return out


def _scalar_value(value: object) -> float:
    """Convert one returned value to a float (NaN if it is not real).

    Args:
        value: What the function returned for a single x.

    Returns:
        The value as a float.

    Raises:
        TypeError: If the value is not a number.
        ValueError: If the value cannot be converted.
    """
    if isinstance(value, np.ndarray):
        if value.size != 1:
            msg = "expected a single number"
            raise TypeError(msg)
        value = value.item()
    if isinstance(value, (complex, np.complexfloating)):
        z = complex(value)
        limit = _COMPLEX_TOLERANCE * max(1.0, abs(z.real))
        return z.real if abs(z.imag) <= limit else math.nan
    return float(value)  # type: ignore[arg-type]
