"""Taylor polynomials and Fourier partial sums, as functions to plot.

:func:`taylor` returns the Taylor polynomial of a function about a point,
from its exact derivatives (:mod:`pygridplot.core.plot.derivatives`).
:func:`fourier` returns a partial sum of the Fourier series of a function
on an interval, from coefficients computed by adaptive quadrature
(:mod:`pygridplot.core.plot.quadrature`). Both are computed once, when
created, and are then cheap functions of ``x`` that can be plotted,
evaluated and differentiated (exactly: they accept jets) like any other;
their coefficients are attributes, for showing the polynomial or the
spectrum.

**Fourier coefficients** are ``aₖ = (2/T) ∫ f(x) cos(kωx) dx`` and
``bₖ = (2/T) ∫ f(x) sin(kωx) dx`` over the interval, with ``T`` its length
and ``ω = 2π/T``. They are computed together from one adaptive run over
``f`` alone: the run starts from ``2n`` equal parts, so that every part
spans at most half a period of the highest harmonic (on which the
21-point rule integrates ``cos`` to rounding), and is refined wherever
``f`` needs it (at its jumps and kinks, which a square or sawtooth wave
has). The weights ``e^{-ikωx}`` at all the nodes are then built by
repeated multiplication, ``n`` complex products per node rather than
``2n`` trigonometric functions.

**Evaluating** a partial sum reduces ``x`` into the interval first (the sum
has period ``T``), then sums ``Re Σ cₖ e^{ikωx}`` with the powers built the
same way.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Final, cast, overload

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.plot.derivatives import check_order, derivatives
from pygridplot.core.plot.evaluation import Function, evaluate_quietly
from pygridplot.core.plot.jets import Jet
from pygridplot.core.plot.quadrature import KRONROD, integrate, nodes_of

MAX_TERMS: Final = 1000
"""Most harmonics in a Fourier partial sum."""

_CHUNK: Final = 1 << 17
"""Most entries in one block of the ``(points, harmonics)`` table of powers."""


class TaylorPolynomial:
    """A Taylor polynomial ``Σ cₖ (x - a)ᵏ``, a function of ``x``.

    Create them with :func:`taylor`. Calling one with a number returns a
    number, and with an array an array of the same shape.

    Attributes:
        func: The function it approximates.
        center: The point ``a`` it is expanded about.
        coefficients: ``cₖ = f⁽ᵏ⁾(a) / k!`` for ``k = 0 ... n`` (read-only).
    """

    __slots__ = ("center", "coefficients", "func")

    def __init__(
        self, func: Function, center: float, coefficients: NDArray[np.float64]
    ) -> None:
        self.func = func
        self.center = center
        self.coefficients = coefficients

    @property
    def degree(self) -> int:
        """The degree ``n`` (the highest power, whatever its coefficient)."""
        return int(self.coefficients.size) - 1

    @overload
    def __call__(self, x: float) -> float: ...
    @overload
    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def __call__(self, x: Any) -> Any:
        """Evaluate the polynomial, by Horner's rule in ``x - a``.

        Args:
            x: A number, an array of numbers, or a jet (so that the
                polynomial can be differentiated exactly).

        Returns:
            The value, a float for a number and an array of the same shape
            for an array.
        """
        if isinstance(x, Jet):
            return self._horner(x)
        values = np.asarray(x, dtype=np.float64)
        with np.errstate(all="ignore"):
            result = np.asarray(self._horner(values), dtype=np.float64)
        return float(result) if values.ndim == 0 else result

    def _horner(self, x: Any) -> Any:
        u = x - self.center
        c = self.coefficients
        total = u * 0.0 + c[-1]
        for k in range(c.size - 2, -1, -1):
            total = total * u + c[k]
        return total

    def __repr__(self) -> str:
        """Return a short description."""
        name = getattr(self.func, "__name__", repr(self.func))
        return f"taylor({name}, {self.center!r}, {self.degree})"


class FourierSeries:
    """A partial sum of a Fourier series, a function of ``x`` with period ``T``.

    ``S(x) = a₀/2 + Σₖ (aₖ cos(kωx) + bₖ sin(kωx))`` for ``k = 1 ... n``,
    with ``ω = 2π/T``. Create them with :func:`fourier`. Calling one with a
    number returns a number, and with an array an array of the same shape.

    Attributes:
        func: The function whose series it is.
        interval: ``(start, end)``, the interval the coefficients were
            computed over; its length is the period ``T``.
        cosines: ``a₀ ... aₙ`` (read-only). ``a₀/2`` is the mean of the
            function over the interval.
        sines: ``b₀ ... bₙ`` (read-only), with ``b₀ = 0``.
    """

    __slots__ = ("cosines", "func", "interval", "sines")

    def __init__(
        self,
        func: Function,
        interval: tuple[float, float],
        cosines: NDArray[np.float64],
        sines: NDArray[np.float64],
    ) -> None:
        self.func = func
        self.interval = interval
        self.cosines = cosines
        self.sines = sines

    @property
    def terms(self) -> int:
        """The number ``n`` of harmonics."""
        return int(self.cosines.size) - 1

    @property
    def period(self) -> float:
        """The period ``T``, the length of the interval."""
        return self.interval[1] - self.interval[0]

    @property
    def amplitudes(self) -> NDArray[np.float64]:
        """The amplitude of each harmonic, ``k = 0 ... n``: the series' spectrum.

        ``√(aₖ² + bₖ²)``, the size of the harmonic ``aₖ cos kωx + bₖ sin kωx``,
        and ``|a₀| / 2`` (the mean's) for ``k = 0``. A new array each time;
        draw it with :meth:`~pygridplot.Grid.sequence`.
        """
        amplitudes = np.hypot(self.cosines, self.sines)
        amplitudes[0] = abs(self.cosines[0]) / 2
        return amplitudes

    @overload
    def __call__(self, x: float) -> float: ...
    @overload
    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def __call__(self, x: Any) -> Any:
        """Evaluate the partial sum.

        Args:
            x: A number, an array of numbers, or a jet (so that the sum can be
                differentiated exactly).

        Returns:
            The value, a float for a number and an array of the same shape
            for an array. NaN for non-finite ``x``.
        """
        if isinstance(x, Jet):
            return self._sum_jet(x)
        values = np.asarray(x, dtype=np.float64)
        with np.errstate(all="ignore"):
            flat = values.ravel()
            result = np.empty(flat.shape)
            step = max(1, _CHUNK // max(1, self.terms))
            for i in range(0, flat.size, step):
                result[i : i + step] = self._sum(flat[i : i + step])
        return float(result[0]) if values.ndim == 0 else result.reshape(values.shape)

    def _reduced(self, x: Any) -> Any:
        """Move ``x`` into the interval, by whole periods."""
        start = self.interval[0]
        return start + np.remainder(x - start, self.period)

    def _sum(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """``Re Σ cₖ zᵏ`` with ``z = e^{iωx}`` and ``cₖ = aₖ - i bₖ``."""
        total = np.full(x.shape, 0.5 * self.cosines[0])
        if self.terms:
            z = np.exp(1j * (2.0 * math.pi / self.period) * self._reduced(x))
            powers = np.cumprod(np.repeat(z[:, np.newaxis], self.terms, axis=1), axis=1)
            total += powers.real @ self.cosines[1:] + powers.imag @ self.sines[1:]
        return total

    def _sum_jet(self, x: Jet) -> Any:
        """The partial sum of a jet, term by term (jets are not complex)."""
        omega = 2.0 * math.pi / self.period
        angle = omega * self._reduced(x)
        total: Any = angle * 0.0 + 0.5 * self.cosines[0]
        for k in range(1, self.terms + 1):
            total = (
                total
                + self.cosines[k] * np.cos(k * angle)
                + self.sines[k] * np.sin(k * angle)
            )
        return total

    def __repr__(self) -> str:
        """Return a short description."""
        name = getattr(self.func, "__name__", repr(self.func))
        return f"fourier({name}, {self.terms}, {self.interval!r})"


def taylor(func: Function, a: float, n: int) -> TaylorPolynomial:
    """Return the Taylor polynomial of degree ``n`` of a function about ``a``.

    The polynomial ``Σ f⁽ᵏ⁾(a) (x - a)ᵏ / k!`` is a function of ``x`` like any
    other::

        p = taylor(np.sin, 0, 5)           # x - x³/6 + x⁵/120
        grid.plot(screen, p)
        p.coefficients                     # [0, 1, 0, -1/6, 0, 1/120]

    Its coefficients come from the exact derivatives of
    :func:`~pygridplot.derivatives` (to rounding, for functions written with
    NumPy; numerical differences for :mod:`math` functions, which are only
    good to the first few orders).

    Args:
        func: The function.
        a: The point to expand about.
        n: The degree, a whole number in ``[0, 40]``.

    Returns:
        The polynomial, with ``center`` (``a``) and ``coefficients``
        (``f⁽ᵏ⁾(a) / k!`` for ``k = 0 ... n``) as attributes. Its values are
        NaN if the function cannot be differentiated at ``a``.

    Raises:
        TypeError: If ``func`` is not callable or ``a`` is not a number.
        ValueError: If ``a`` is not finite or ``n`` is not a whole number in
            ``[0, 40]``.
    """
    _check_function(func)
    center = _number(a, "a")
    degree = check_order(n)
    values = np.asarray(derivatives(func, center, degree), dtype=np.float64)
    factorials = np.array([float(math.factorial(k)) for k in range(degree + 1)])
    coefficients = values / factorials
    coefficients.flags.writeable = False
    return TaylorPolynomial(func, center, coefficients)


def fourier(
    func: Function, n: int, interval: tuple[float, float] | None = None
) -> FourierSeries:
    """Return the partial sum with ``n`` harmonics of a function's Fourier series.

    The series is that of the function on ``interval``, repeated with the
    interval's length as period: a function given on ``(-π, π)``, such as
    ``np.sign`` (a square wave) or ``lambda x: x`` (a sawtooth), need not be
    periodic itself. The partial sum is a function of ``x`` like any other::

        square = fourier(np.sign, 9)                      # 9 harmonics
        grid.plot(screen, square)
        wave = fourier(lambda x: x * (1 - x), 5, (0, 1))   # period 1
        square.sines[1]                                    # 4/π

    The coefficients are integrals over the interval, computed accurately
    (to about 10 digits) for smooth functions and for functions with jumps
    and kinks, as a square wave has.

    Args:
        func: The function, written with NumPy or :mod:`math`.
        n: The number of harmonics, a whole number in ``[0, 1000]`` (0 gives
            the mean).
        interval: ``(start, end)``, one period. The default (``None``) is
            ``(-π, π)``.

    Returns:
        The partial sum ``a₀/2 + Σ (aₖ cos kωx + bₖ sin kωx)``, ``ω = 2π/T``,
        with ``cosines`` (``a₀ ... aₙ``), ``sines`` (``b₀ ... bₙ``),
        ``amplitudes`` (the spectrum), ``interval`` and ``period`` as
        attributes. Its values are NaN if the
        function's integral over the interval is undefined (it diverges, or
        the function is undefined on part of it).

    Raises:
        TypeError: If ``func`` is not callable or the interval's ends are not
            numbers.
        ValueError: If the interval is not two finite numbers with
            ``start < end``, or ``n`` is not a whole number in ``[0, 1000]``.
    """
    _check_function(func)
    terms = _check_terms(n)
    start, end = _interval((-math.pi, math.pi) if interval is None else interval)
    cosines, sines = _coefficients(func, terms, start, end)
    cosines.flags.writeable = False
    sines.flags.writeable = False
    return FourierSeries(func, (start, end), cosines, sines)


def _coefficients(
    func: Function, terms: int, start: float, end: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the Fourier coefficients ``a₀ ... aₙ`` and ``b₀ ... bₙ``.

    Args:
        func: The user's function.
        terms: ``n``.
        start: The interval's start.
        end: Its end.

    Returns:
        The cosine and sine coefficients, all NaN if the integral of the
        function over the interval is undefined.
    """
    parts = max(8, 2 * terms)
    edges = np.linspace(start, end, parts + 1)
    edges[-1] = end
    result = integrate(
        lambda x: evaluate_quietly(func, x),
        edges[:-1],
        edges[1:],
        np.zeros(parts, dtype=np.intp),
        1,
        samples=True,
    )
    if not math.isfinite(result.values[0]):
        return np.full(terms + 1, np.nan), np.full(terms + 1, np.nan)
    period = end - start
    half = 0.5 * result.hi - 0.5 * result.lo
    weights = (result.samples * KRONROD * half[:, np.newaxis]).ravel()  # f(x) dx
    x = nodes_of(result.lo, result.hi).ravel()
    # Σ w e^{-ikωx} for k = 0 ... n, by repeated multiplication: rounding
    # grows only like k ε. Reduced into the interval first (it already is,
    # but the reduction makes the phase match the evaluation's exactly).
    z = np.exp(
        -1j * (2.0 * math.pi / period) * (start + np.remainder(x - start, period))
    )
    sums = np.empty(terms + 1, dtype=np.complex128)
    sums[0] = weights.sum()
    current = weights.astype(np.complex128)
    for k in range(1, terms + 1):
        current *= z
        sums[k] = current.sum()
    scale = 2.0 / period
    cosines = scale * sums.real
    sines = -scale * sums.imag
    sines[0] = 0.0
    return np.asarray(cosines), np.asarray(sines)


def _check_function(func: object) -> None:
    """Reject things that cannot be called (checked for callers without types).

    Raises:
        TypeError: If ``func`` is not callable.
    """
    if not callable(func):
        msg = f"expected a function, got {func!r}"
        raise TypeError(msg)


def _number(value: object, name: str) -> float:
    """Return a finite number as a float.

    Raises:
        TypeError: If it is not a real number.
        ValueError: If it is not finite.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        msg = f"{name} must be a number, got {value!r}"
        raise TypeError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"{name} must be finite, got {value!r}"
        raise ValueError(msg)
    return number


def _check_terms(n: object) -> int:
    """Validate a number of harmonics.

    Raises:
        ValueError: If it is not a whole number in ``[0, MAX_TERMS]``.
    """
    if (
        isinstance(n, bool)
        or not isinstance(n, (int, np.integer))
        or not 0 <= n <= MAX_TERMS
    ):
        msg = f"n must be a whole number from 0 to {MAX_TERMS}, got {n!r}"
        raise ValueError(msg)
    return int(n)


def _interval(interval: object) -> tuple[float, float]:
    """Validate an interval ``(start, end)``.

    Raises:
        TypeError: If its ends are not numbers.
        ValueError: If it is not two finite numbers in increasing order.
    """
    try:
        start, end = cast(tuple[object, object], interval)
    except (TypeError, ValueError):
        msg = f"the interval must be a pair (start, end), got {interval!r}"
        raise ValueError(msg) from None
    lo, hi = _number(start, "the interval's start"), _number(end, "the interval's end")
    if not lo < hi:
        msg = f"the interval must have start < end, got {interval!r}"
        raise ValueError(msg)
    return lo, hi
