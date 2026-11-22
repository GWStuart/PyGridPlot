"""Derivatives of any order of functions of one variable.

:func:`derivative` turns a function ``f`` into ``f⁽ⁿ⁾``, a function that can
be plotted, evaluated at a point (the slope of a tangent line) or passed
anywhere a function is expected. :func:`derivatives` gives all the
derivatives up to order ``n`` at once.

Derivatives are computed in the best way the function allows, tried in turn:

1. **Exactly, for all points at once**, with Taylor arithmetic
   (:mod:`pygridplot.core.plot.jets`): the function is called once with a
   :class:`~pygridplot.core.plot.jets.Jet` in place of ``x``. This works for
   functions written with NumPy operations and ufuncs, ``np.where``,
   ``np.clip`` and ``np.piecewise``, and is exact to rounding at any order,
   including right next to jumps, kinks and poles.
2. **Exactly, point by point**, if the function only works on single
   values but still uses NumPy (``if x > 0: ...``): it is called with a jet
   of one point at a time.
3. **Numerically**, for functions that cannot take a jet at all (those
   written with :mod:`math`, or using functions without a Taylor rule, such
   as SciPy's): central differences with Richardson extrapolation, using a
   step suited to the order and to ``x``. These are accurate to about
   ``1e-10`` for a first derivative, ``1e-6`` by the fourth, and worse
   beyond; next to a jump or a kink they see a steep slope that is not
   there, and at a domain edge they are undefined.

The derivative of an antiderivative (:func:`~pygridplot.antiderivative`)
is the function integrated, by the fundamental theorem of calculus, so it
is taken exactly that way: ``derivative(antiderivative(f), n)`` is
``derivative(f, n - 1)``.

Like every function the library evaluates, a derivative never raises: it
is NaN wherever it is undefined.
"""

from __future__ import annotations

import warnings
from math import comb
from typing import Any, Final, cast, overload

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.plot.evaluation import Function, evaluate_quietly
from pygridplot.core.plot.integrals import Antiderivative
from pygridplot.core.plot.jets import MAX_ORDER, Jet

_EPS: Final = float(np.finfo(np.float64).eps)

_PROBES: Final = 3
"""Points tried with a single-point jet before differentiating numerically."""


class Derivative:
    """The ``n``-th derivative of a function, itself a function of ``x``.

    Create derivatives with :func:`derivative`. Calling one with a number
    returns a number, and with an array an array of the same shape; it never
    raises, returning NaN where the derivative is undefined.

    Attributes:
        func: The function being differentiated.
        order: Which derivative this is.
    """

    __slots__ = ("func", "order")

    def __init__(self, func: Function, order: int) -> None:
        self.func = func
        self.order = order

    @overload
    def __call__(self, x: float) -> float: ...
    @overload
    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def __call__(self, x: Any) -> Any:
        """Evaluate the derivative.

        Args:
            x: A number or an array of numbers.

        Returns:
            The derivative at ``x``, a float for a number and an array of the
            same shape for an array.

        Raises:
            TypeError: If ``x`` is a jet (a derivative being differentiated
                again): the caller then differentiates numerically.
        """
        if isinstance(x, Jet):
            msg = "derivatives are differentiated by raising the order instead"
            raise TypeError(msg)
        values = np.asarray(x, dtype=np.float64)
        result = taylor_derivatives(self.func, values.ravel(), self.order)[self.order]
        if values.ndim == 0:
            return float(result[0])
        return result.reshape(values.shape)

    def __repr__(self) -> str:
        """Return a short description."""
        name = getattr(self.func, "__name__", repr(self.func))
        return f"derivative({name}, {self.order})"


def derivative(func: Function, n: int = 1) -> Derivative:
    """Return the ``n``-th derivative of a function of one variable.

    The result is a function of ``x`` like any other: plot it, call it at a
    point, or differentiate it again::

        df = derivative(np.sin)        # cos
        d3f = derivative(f, 3)
        plane.plot(screen, df)
        slope = df(1.5)

    The derivative is exact for functions written with NumPy, and computed
    numerically for others (see :mod:`pygridplot.core.plot.derivatives`).

    Args:
        func: The function, taking and returning numbers or NumPy arrays.
        n: The order of the derivative (0 returns the function itself,
            evaluated robustly).

    Returns:
        The derivative.

    Raises:
        TypeError: If ``func`` is not callable.
        ValueError: If ``n`` is not a whole number in ``[0, 40]``.
    """
    if not callable(cast(object, func)):  # checked for callers without types
        msg = f"expected a function, got {func!r}"
        raise TypeError(msg)
    order = check_order(n)
    if isinstance(func, Derivative):  # a derivative of a derivative
        order = check_order(func.order + order)
        func = func.func
    if isinstance(func, Antiderivative) and order >= 1:
        return derivative(func.func, order - 1)  # F' = f
    return Derivative(func, order)


@overload
def derivatives(func: Function, x: float, n: int) -> tuple[float, ...]: ...
@overload
def derivatives(
    func: Function, x: NDArray[np.float64], n: int
) -> NDArray[np.float64]: ...
def derivatives(func: Function, x: Any, n: int) -> Any:
    """Return a function's value and its derivatives up to order ``n``.

    Computing them together costs little more than the highest one alone::

        value, slope, curvature = derivatives(f, 1.5, 2)

    Args:
        func: The function.
        x: A number, or an array of numbers.
        n: The highest order.

    Returns:
        For a number, the tuple ``(f(x), f'(x), ..., f⁽ⁿ⁾(x))``; for an
        array, an array of shape ``(n + 1, *x.shape)``. NaN where undefined.

    Raises:
        TypeError: If ``func`` is not callable.
        ValueError: If ``n`` is not a whole number in ``[0, 40]``.
    """
    if not callable(cast(object, func)):  # checked for callers without types
        msg = f"expected a function, got {func!r}"
        raise TypeError(msg)
    order = check_order(n)
    offset = 0
    if isinstance(func, Derivative):
        offset = func.order
        order = check_order(offset + order)
        func = func.func
    values = np.asarray(x, dtype=np.float64)
    result = taylor_derivatives(func, values.ravel(), order)[offset:]
    if values.ndim == 0:
        return tuple(float(v) for v in result[:, 0])
    return result.reshape((result.shape[0], *values.shape))


def taylor_derivatives(
    func: Function, xs: NDArray[np.float64], n: int
) -> NDArray[np.float64]:
    """Return ``f⁽ᵏ⁾(x)`` for ``k = 0 ... n`` at every point, by the best method.

    Args:
        func: The function.
        xs: The points, a 1-D array.
        n: The highest order.

    Returns:
        Shape ``(n + 1, len(xs))``, NaN where undefined (never infinite).
    """
    if isinstance(func, Antiderivative) and n >= 1:
        # The fundamental theorem of calculus: F' = f, exactly.
        values = func(xs)[np.newaxis]
        return np.concatenate([values, taylor_derivatives(func.func, xs, n - 1)])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if xs.size == 0:
            return np.zeros((n + 1, 0))
        result = _vectorised(func, xs, n)
        if result is None:
            result = _point_by_point(func, xs, n)
        if result is None:
            result = _numerical(func, xs, n)
    return np.where(np.isfinite(result), result, np.nan)


def check_order(n: object) -> int:
    """Validate a derivative order (or a Taylor polynomial's degree).

    Args:
        n: The order.

    Returns:
        It, as an int.

    Raises:
        ValueError: If it is not a whole number in ``[0, MAX_ORDER]``.
    """
    if (
        isinstance(n, bool)
        or not isinstance(n, (int, np.integer))
        or not 0 <= n <= MAX_ORDER
    ):
        msg = f"the order must be a whole number from 0 to {MAX_ORDER}, got {n!r}"
        raise ValueError(msg)
    return int(n)


def _call(func: Function, jet: Jet) -> Jet | None:
    """Call a function with a jet; return its result as a jet, if it is one.

    Args:
        func: The function.
        jet: The variable.

    Returns:
        The result (a constant result becomes a jet with zero derivatives),
        or ``None`` if the call failed or returned something that is not a
        number or a jet of the right shape.
    """
    try:
        result = func(jet)
    except Exception:
        return None
    if isinstance(result, Jet):
        if result.order != jet.order:
            return None
        try:
            return Jet(np.broadcast_to(result.c, jet.c.shape))
        except ValueError:
            return None
    if isinstance(result, bool | int | float | np.number) or (
        isinstance(result, np.ndarray)
        and np.issubdtype(result.dtype, np.number)
        and not np.iscomplexobj(result)
    ):
        try:
            values = np.broadcast_to(np.asarray(result, dtype=np.float64), jet.shape)
        except ValueError:
            return None
        return Jet.constant(values, jet.order)
    return None


def _vectorised(
    func: Function, xs: NDArray[np.float64], n: int
) -> NDArray[np.float64] | None:
    """Differentiate exactly at all points with one call, if the function allows."""
    result = _call(func, Jet.variable(xs.copy(), n))
    return None if result is None else result.derivatives()


def _point_by_point(
    func: Function, xs: NDArray[np.float64], n: int
) -> NDArray[np.float64] | None:
    """Differentiate exactly one point at a time, if the function allows.

    A few points are tried first; if none works, the function cannot take a
    jet (it uses :mod:`math`, say), and ``None`` is returned. Points where
    the function fails on its own are undefined (NaN).
    """
    probes = np.unique(np.linspace(0, xs.size - 1, min(_PROBES, xs.size)).astype(int))
    if all(_call(func, Jet.variable(xs[i], n)) is None for i in probes):
        return None
    result = np.full((n + 1, xs.size), np.nan)
    for i, x in enumerate(xs):
        jet = _call(func, Jet.variable(x, n))
        if jet is not None:
            result[:, i] = jet.derivatives()
    return result


def _numerical(func: Function, xs: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Differentiate by central differences with Richardson extrapolation.

    For order ``k`` the step is ``eps^(1 / (k + 4))`` times ``max(1, |x|)``,
    which balances rounding against truncation for smooth functions.
    """
    result = np.empty((n + 1, xs.size))
    result[0] = evaluate_quietly(func, xs)
    scale = np.maximum(1.0, np.abs(xs))
    for k in range(1, n + 1):
        h = _EPS ** (1.0 / (k + 4)) * scale
        coarse = _central(func, xs, k, h)
        fine = _central(func, xs, k, h / 2)
        result[k] = (4.0 * fine - coarse) / 3.0
    return result


def _central(
    func: Function, xs: NDArray[np.float64], k: int, h: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the ``k``-th central difference quotient with steps ``h``.

    All ``k + 1`` shifted points are evaluated in one call.
    """
    offsets = np.arange(k + 1) - k / 2
    weights = np.array([(-1.0) ** (k - i) * comb(k, i) for i in range(k + 1)])
    points = xs[np.newaxis, :] + offsets[:, np.newaxis] * h[np.newaxis, :]
    values = evaluate_quietly(func, points.ravel()).reshape(points.shape)
    return np.asarray(weights @ values / h**k, dtype=np.float64)
