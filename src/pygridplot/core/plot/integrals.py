"""Definite integrals and antiderivatives of functions of one variable.

:func:`integral` returns ``∫ₐᵇ f(x) dx`` as a number, for showing the area
under a curve between two draggable points, say. :func:`antiderivative`
returns ``F(x) = ∫ₐˣ f(t) dt`` as a function, to plot like any other or to
evaluate.

Both use the adaptive Gauss-Kronrod quadrature of
:mod:`pygridplot.core.plot.quadrature`, which is accurate to about 10
significant digits (relative to the integral of ``|f|``) for smooth
functions, functions with jumps and kinks, and integrable singularities
such as ``1/sqrt(x)`` at 0. Like every function the library evaluates,
they never raise because of the function: the result is NaN when the
integral diverges (``1/x`` across 0) or the function is undefined on part
of the interval (``sqrt(x)`` left of 0). A point where the function alone
is undefined (``sin(x) / x`` at 0) does not matter.

**Infinite limits** are mapped onto a finite interval: ``x = c + 1/t - 1``
takes ``t`` in ``(0, 1]`` to ``x`` in ``[c, ∞)``, with ``dx = dt / t²``.
Functions that decay like ``1/x²`` or faster give a smooth integrand in
``t``; the far tail lands next to ``t = 0``, where floating point is
densest, so it is resolved however slowly it decays, and an integral that
diverges (``1/x``) is found to.

**Antiderivatives** are computed by cumulative quadrature on the points
they are evaluated at: the points are sorted with the lower limit ``a``,
the integral over each gap between neighbours is found (all gaps together,
in one adaptive run), and ``F`` is their running sum outwards from ``a``.
Plotting an antiderivative therefore costs about as much as integrating
over the plotted range once per call of the function, not once per point.
A gap that diverges makes ``F`` undefined beyond it: the antiderivative of
``1/x`` from 1 is ``log x`` for ``x > 0`` and undefined for ``x < 0``.

Because ``F``'s value at a point comes from the gaps between the points it
is evaluated with, values at the same ``x`` computed in different calls can
differ in the last few digits. The plot cache allows for this: it checks an
antiderivative's *integrand* to tell whether a plot has changed (see
:mod:`pygridplot.core.plot.memo`).
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Final, overload

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.plot.evaluation import Function, evaluate_quietly
from pygridplot.core.plot.jets import Jet
from pygridplot.core.plot.quadrature import Integrand, integrate

INITIAL_PARTS: Final = 8
"""Equal parts a definite integral starts from (in the mapped variable).

Starting from several parts rather than one makes it less likely that a
narrow feature falls between the first nodes unseen. An antiderivative's
gaps start from up to this many parts each, in proportion to their share of
the whole range."""


class Antiderivative:
    """``F(x) = ∫ₐˣ f(t) dt``, a function of ``x``.

    Create antiderivatives with :func:`antiderivative`. Calling one with a
    number returns a number, and with an array an array of the same shape;
    it never raises, returning NaN where the integral is undefined.

    Attributes:
        func: The function being integrated.
        lower: The lower limit ``a``, where ``F`` is 0 (may be infinite).
    """

    __slots__ = ("func", "lower")

    def __init__(self, func: Function, lower: float) -> None:
        self.func = func
        self.lower = lower

    @overload
    def __call__(self, x: float) -> float: ...
    @overload
    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def __call__(self, x: Any) -> Any:
        """Evaluate the antiderivative.

        Args:
            x: A number or an array of numbers (``±inf`` allowed).

        Returns:
            ``F(x)``, a float for a number and an array of the same shape for
            an array.

        Raises:
            TypeError: If ``x`` is a jet (an antiderivative inside a function
                being differentiated): the caller then differentiates
                numerically. (:func:`~pygridplot.derivative` of an
                antiderivative itself is exact: it is the integrand.)
        """
        if isinstance(x, Jet):
            msg = "antiderivatives are differentiated through derivative() instead"
            raise TypeError(msg)
        values = np.asarray(x, dtype=np.float64)
        result = antiderivative_values(self.func, self.lower, values.ravel())
        if values.ndim == 0:
            return float(result[0])
        return result.reshape(values.shape)

    def __repr__(self) -> str:
        """Return a short description."""
        name = getattr(self.func, "__name__", repr(self.func))
        return f"antiderivative({name}, {self.lower!r})"


def integral(func: Function, a: float, b: float) -> float:
    """Return the definite integral of a function from ``a`` to ``b``.

    For example::

        area = integral(np.sin, 0, np.pi)                  # 2.0
        integral(lambda x: np.exp(-x**2), -np.inf, np.inf)  # √π
        integral(f, p.x, q.x)          # between two draggable points

    The integral is computed adaptively, to about 10 significant digits
    for smooth functions, jumps, kinks and integrable singularities such as
    ``1/sqrt(x)`` at 0. Either limit may be infinite, and ``b < a`` gives
    the negative of the integral from ``b`` to ``a``.

    Args:
        func: The function, written with NumPy or :mod:`math` (or anything
            :meth:`~pygridplot.Grid.plot` accepts).
        a: The lower limit.
        b: The upper limit.

    Returns:
        The integral, or NaN if it diverges (``1/x`` from -1 to 1), if the
        function is undefined on part of the interval (``sqrt(x)`` from -1 to
        1), if a limit is NaN, or if it cannot be computed accurately (an
        integrand oscillating many thousands of times). A point where the
        function is undefined on its own (``sin(x) / x`` at 0) does not
        matter.

    Raises:
        TypeError: If ``func`` is not callable or a limit is not a number.
    """
    _check_function(func)
    lo, hi = _limit(a, "a"), _limit(b, "b")
    if math.isnan(lo) or math.isnan(hi):
        return math.nan
    if lo == hi:
        return 0.0
    if lo > hi:
        return -definite_integral(_robust(func), hi, lo)
    return definite_integral(_robust(func), lo, hi)


def antiderivative(func: Function, a: float = 0.0) -> Antiderivative:
    """Return the antiderivative ``F(x) = ∫ₐˣ f(t) dt`` of a function.

    The result is a function of ``x`` like any other: plot it, call it at a
    point, or differentiate it (exactly: the derivative of an
    antiderivative is the function itself)::

        F = antiderivative(np.cos)                         # sin
        grid.plot(screen, F)
        cdf = antiderivative(lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), -np.inf)
        cdf(1.96)                                          # 0.975

    Args:
        func: The function, written with NumPy or :mod:`math`.
        a: The lower limit, where ``F`` is 0 (``-np.inf`` for a cumulative
            distribution). Choose it where the function is defined: the
            antiderivative of ``1/x`` from 0 is undefined everywhere.

    Returns:
        The antiderivative. It is NaN where the integral from ``a`` is
        undefined, as for :func:`integral`: beyond a point where it diverges,
        or across a stretch where the function is undefined.

    Raises:
        TypeError: If ``func`` is not callable or ``a`` is not a number.
        ValueError: If ``a`` is NaN.
    """
    _check_function(func)
    lower = _limit(a, "a")
    if math.isnan(lower):
        msg = "the lower limit a must not be NaN"
        raise ValueError(msg)
    return Antiderivative(func, lower)


def definite_integral(func: Integrand, lo: float, hi: float) -> float:
    """Integrate a robust function from ``lo`` to ``hi``.

    Args:
        func: The integrand, vectorised and robust (it never raises).
        lo: The lower limit (``-inf`` allowed), below ``hi``.
        hi: The upper limit (``inf`` allowed).

    Returns:
        The integral, NaN if it cannot be computed.
    """
    integrand, start, stop = _mapped(func, lo, hi)
    edges = np.linspace(start, stop, INITIAL_PARTS + 1)
    pieces = np.zeros(INITIAL_PARTS, dtype=np.intp)
    return float(integrate(integrand, edges[:-1], edges[1:], pieces, 1).values[0])


def antiderivative_values(
    func: Function, lower: float, xs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return ``∫_lower^x f(t) dt`` at every point, by cumulative quadrature.

    Args:
        func: The user's function (evaluated robustly here).
        lower: The lower limit (finite or infinite, not NaN).
        xs: The points, a 1-D array (NaN gives NaN, ``±inf`` the improper
            integral).

    Returns:
        The values, NaN where undefined.
    """
    robust = _robust(func)
    out = np.full(xs.shape, np.nan)
    finite = np.isfinite(xs)
    known = [xs[finite]]
    if math.isfinite(lower):
        known.append(np.array([lower]))
    points = np.unique(np.concatenate(known))
    if points.size == 0:  # only infinite queries and limit: start from 0
        points = np.zeros(1)
    values = _cumulative(robust, points, lower)
    out[finite] = values[np.searchsorted(points, xs[finite])]
    if np.any(xs == np.inf):
        top = 0.0 if lower == np.inf else values[-1] + _tail(robust, points[-1], 1)
        out[xs == np.inf] = top
    if np.any(xs == -np.inf):
        bottom = 0.0 if lower == -np.inf else values[0] - _tail(robust, points[0], -1)
        out[xs == -np.inf] = bottom
    return out


def _cumulative(
    func: Integrand, points: NDArray[np.float64], lower: float
) -> NDArray[np.float64]:
    """Return the integral from ``lower`` to each of the sorted ``points``.

    Args:
        func: The robust integrand.
        points: Sorted, distinct, finite points; they include ``lower`` if
            it is finite.
        lower: The lower limit.

    Returns:
        The integrals, summed outwards from ``lower``.
    """
    gaps = _gaps(func, points)
    if math.isfinite(lower):
        start = int(np.searchsorted(points, lower))
        values = np.empty(points.size)
        values[start] = 0.0
        values[start + 1 :] = np.cumsum(gaps[start:])
        values[:start] = -np.cumsum(gaps[:start][::-1])[::-1]
        return values
    if lower < 0:
        first = _tail(func, points[0], -1)
        return np.asarray(first + np.concatenate([[0.0], np.cumsum(gaps)]))
    last = -_tail(func, points[-1], 1)
    return np.asarray(last - np.concatenate([np.cumsum(gaps[::-1])[::-1], [0.0]]))


def _gaps(func: Integrand, points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the integral over each gap between neighbouring points.

    All the gaps are integrated together, as the pieces of one adaptive
    run. Each starts from parts in proportion to its share of the whole
    range (from one to :data:`INITIAL_PARTS`), so that a long gap (from a
    distant lower limit to the plotted points, say) is not sampled more
    coarsely than a short integral would be.

    Args:
        func: The robust integrand.
        points: Sorted, distinct, finite points.

    Returns:
        ``len(points) - 1`` integrals, NaN where undefined.
    """
    count = points.size - 1
    if count < 1:
        return np.zeros(0)
    with np.errstate(all="ignore"):
        span = points[-1] - points[0]
        share = np.diff(points) / span
    parts = np.clip(np.ceil(INITIAL_PARTS * np.nan_to_num(share)), 1, INITIAL_PARTS)
    parts = parts.astype(np.intp)
    piece = np.repeat(np.arange(count), parts)
    k = np.arange(piece.size) - (np.cumsum(parts) - parts)[piece]
    m = parts[piece]
    left, right = points[piece], points[piece + 1]
    # Both ends from the same formula, so neighbouring parts meet exactly;
    # the first part starts at `left` and the last ends at `right`.
    lo = left * ((m - k) / m) + right * (k / m)
    hi = left * ((m - k - 1) / m) + right * ((k + 1) / m)
    return integrate(func, lo, hi, piece, count).values


def _tail(func: Integrand, end: float, side: int) -> float:
    """Return the integral from ``end`` to infinity (``side`` 1) or from minus infinity.

    Args:
        func: The robust integrand.
        end: The finite end.
        side: 1 for ``[end, ∞)``, -1 for ``(-∞, end]``.

    Returns:
        The integral (positive orientation), NaN if it cannot be computed.
    """
    if side > 0:
        return definite_integral(func, end, math.inf)
    return definite_integral(func, -math.inf, end)


def _mapped(func: Integrand, lo: float, hi: float) -> tuple[Integrand, float, float]:
    """Map an integral with infinite limits onto a finite interval.

    ``x = c + 1/t - 1`` takes ``t`` in ``(0, 1]`` to ``[c, ∞)`` and
    ``x = c + 1/t + 1`` takes ``t`` in ``[-1, 0)`` to ``(-∞, c]``; both
    together (``c = 0``) cover the whole line. Then ``dx = -dt / t²``, so
    the integrand becomes ``f(x(t)) / t²``. ``t = 0`` is an end of the
    intervals, never evaluated.

    Args:
        func: The robust integrand.
        lo: The lower limit, below ``hi``.
        hi: The upper limit.

    Returns:
        The integrand to use and the interval to integrate it over.
    """
    if math.isfinite(lo) and math.isfinite(hi):
        return func, lo, hi
    if math.isfinite(lo):
        return (lambda t: func(lo + (1.0 / t - 1.0)) / t / t), 0.0, 1.0
    if math.isfinite(hi):
        return (lambda t: func(hi + (1.0 / t + 1.0)) / t / t), -1.0, 0.0
    return (lambda t: func(1.0 / t - np.sign(t)) / t / t), -1.0, 1.0


def _robust(func: Function) -> Integrand:
    """Wrap a user function so that it is vectorised and never raises."""
    return lambda x: evaluate_quietly(func, x)


def _check_function(func: object) -> None:
    """Reject things that cannot be called (checked for callers without types).

    Raises:
        TypeError: If ``func`` is not callable.
    """
    if not callable(func):
        msg = f"expected a function, got {func!r}"
        raise TypeError(msg)


def _limit(value: object, name: str) -> float:
    """Return a limit as a float.

    Args:
        value: The limit.
        name: Its name, for the error message.

    Returns:
        It, as a float (possibly infinite or NaN).

    Raises:
        TypeError: If it is not a real number.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        msg = f"the limit {name} must be a number, got {value!r}"
        raise TypeError(msg)
    return float(value)
