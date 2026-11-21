"""Taylor arithmetic: exact derivatives of ordinary NumPy functions.

A :class:`Jet` stands in for the variable ``x`` and carries, at every point
at once, the Taylor coefficients of whatever has been computed from it so
far: ``c[k] = g⁽ᵏ⁾(x) / k!`` for ``k = 0 ... n``. Passing a jet to a function
written with NumPy, such as ``lambda x: np.sin(x) ** 2 * np.exp(-x)``, runs
the function unchanged: NumPy hands every operation on a jet to
:meth:`Jet.__array_ufunc__`, which applies the operation's rule to the
coefficients. The result's coefficients give the function's derivatives up
to order ``n``, exact to rounding (no step size, no truncation error).

This is *forward-mode automatic differentiation in Taylor mode*. Each
operation costs ``O(n²)`` per point; the rules are the classic recurrences
(Griewank and Walther, *Evaluating Derivatives*, chapter 13):

* products are Cauchy products, quotients their inverse;
* ``exp``, ``log``, powers, ``sin``/``cos`` and their hyperbolic versions,
  ``tan`` and ``tanh`` solve the differential equation each satisfies
  (``y' = y u'`` for ``exp``, ``y' = (1 + y²) u'`` for ``tan``, ...);
* inverse functions (``arcsin``, ``arctan``, ...) integrate their
  derivative, itself computed with jets one order lower;
* piecewise functions (``abs``, ``floor``, ``maximum``, ``np.where``)
  take the derivatives of the piece that applies at each point, so a jump or
  a kink never produces the false spikes of finite differences.

Comparisons and logical operations act on the values, so ``np.where(x > 0,
...)`` works, and a jet of a single point converts to ``bool``, so Python
``if`` statements work point by point. A jet refuses to become a ``float``,
so ``math.sin(x)`` raises :class:`TypeError` cleanly: functions written with
:mod:`math` (or with operations that have no rule here) are differentiated
numerically instead (see :mod:`pygridplot.core.plot.derivatives`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

Coefficients = NDArray[np.float64]
"""Taylor coefficients, shape ``(n + 1, *shape)``."""

MAX_ORDER: Final = 40
"""Highest derivative order supported (the work grows as its square)."""


class Jet:
    """A truncated Taylor series in ``x`` at each of many points.

    Args:
        coefficients: ``c[k] = g⁽ᵏ⁾(x) / k!``, shape ``(n + 1, *shape)``.
    """

    __slots__ = ("c",)
    __array_priority__ = 1000  # NumPy defers binary operators to the jet

    def __init__(self, coefficients: Coefficients) -> None:
        self.c = coefficients

    @classmethod
    def variable(cls, x: ArrayLike, order: int) -> Jet:
        """Return the jet of ``x`` itself: value ``x``, derivative 1, then 0.

        Args:
            x: The points.
            order: The highest derivative to carry.

        Returns:
            The jet.
        """
        values = np.asarray(x, dtype=np.float64)
        c = np.zeros((order + 1, *values.shape))
        c[0] = values
        if order >= 1:
            c[1] = 1.0
        return cls(c)

    @classmethod
    def constant(cls, value: ArrayLike, order: int) -> Jet:
        """Return the jet of a constant: all derivatives zero.

        Where the constant is undefined (NaN or infinite), so are its
        derivatives (NaN), as for every other jet.

        Args:
            value: The constant (any shape).
            order: The highest derivative to carry.

        Returns:
            The jet.
        """
        values = np.asarray(value, dtype=np.float64)
        c = np.zeros((order + 1, *values.shape))
        c[0] = values
        undefined = ~np.isfinite(values)
        if undefined.any():
            c[1:, undefined] = np.nan
        return cls(c)

    # ------------------------------------------------------------------
    # Array-like properties
    # ------------------------------------------------------------------
    @property
    def order(self) -> int:
        """The highest derivative carried."""
        return int(self.c.shape[0]) - 1

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the points, as for an array of the values."""
        return tuple(int(s) for s in self.c.shape[1:])

    @property
    def ndim(self) -> int:
        """The number of dimensions of the points."""
        return self.c.ndim - 1

    @property
    def size(self) -> int:
        """The number of points."""
        return int(np.prod(self.shape))

    @property
    def dtype(self) -> np.dtype[np.float64]:
        """The values' type (always float64)."""
        return np.dtype(np.float64)

    @property
    def value(self) -> NDArray[np.float64]:
        """The values ``g(x)``."""
        return np.asarray(self.c[0], dtype=np.float64)

    def derivatives(self) -> NDArray[np.float64]:
        """Return ``g⁽ᵏ⁾(x)`` for ``k = 0 ... n``, shape ``(n + 1, *shape)``."""
        factorials = np.array([math.factorial(k) for k in range(self.order + 1)])
        scale = factorials.reshape((-1,) + (1,) * self.ndim)
        return np.asarray(self.c * scale, dtype=np.float64)

    def copy(self) -> Jet:
        """Return a copy."""
        return Jet(self.c.copy())

    def astype(self, dtype: object, *args: object, **kwargs: object) -> Jet:
        """Return the jet itself for floating types (it is always float64).

        Args:
            dtype: The requested type.
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            A copy of the jet.

        Raises:
            TypeError: For non-floating types, which would lose the derivatives.
        """
        if not np.issubdtype(np.dtype(dtype), np.floating):  # type: ignore[call-overload]
            msg = f"cannot convert a jet to {dtype}"
            raise TypeError(msg)
        return self.copy()

    def __len__(self) -> int:
        """Return the length of the first axis of the points."""
        if self.ndim == 0:
            msg = "len() of a jet of a single point"
            raise TypeError(msg)
        return self.shape[0]

    def __getitem__(self, index: object) -> Jet:
        """Index the points, as for an array of the values."""
        if isinstance(index, Jet):
            index = index.value
        key = index if isinstance(index, tuple) else (index,)
        everything = (slice(None), *key)  # every coefficient of the chosen points
        return Jet(self.c[everything])

    def __bool__(self) -> bool:
        """Return the truth of a single point's value (for ``if`` statements)."""
        if self.size != 1:
            msg = "the truth value of a jet of several points is ambiguous"
            raise ValueError(msg)
        return bool(self.c[0].reshape(()))

    def __float__(self) -> float:
        """Refuse: converting to a number would lose the derivatives.

        Raises:
            TypeError: Always, so that :mod:`math` functions fail cleanly.
        """
        msg = "a variable being differentiated cannot become a plain float"
        raise TypeError(msg)

    __int__ = __float__
    __index__ = __float__
    __complex__ = __float__

    def __repr__(self) -> str:
        """Return the values and the order."""
        return f"Jet(value={self.c[0]!r}, order={self.order})"

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------
    def __add__(self, other: object) -> Any:
        return np.add(self, other)

    def __radd__(self, other: object) -> Any:
        return np.add(other, self)

    def __sub__(self, other: object) -> Any:
        return np.subtract(self, other)

    def __rsub__(self, other: object) -> Any:
        return np.subtract(other, self)

    def __mul__(self, other: object) -> Any:
        return np.multiply(self, other)

    def __rmul__(self, other: object) -> Any:
        return np.multiply(other, self)

    def __truediv__(self, other: object) -> Any:
        return np.true_divide(self, other)

    def __rtruediv__(self, other: object) -> Any:
        return np.true_divide(other, self)

    def __floordiv__(self, other: object) -> Any:
        return np.floor_divide(self, other)

    def __rfloordiv__(self, other: object) -> Any:
        return np.floor_divide(other, self)

    def __mod__(self, other: object) -> Any:
        return np.remainder(self, other)

    def __rmod__(self, other: object) -> Any:
        return np.remainder(other, self)

    def __pow__(self, other: object) -> Any:
        return np.power(self, other)

    def __rpow__(self, other: object) -> Any:
        return np.power(other, self)

    def __neg__(self) -> Any:
        return np.negative(self)

    def __pos__(self) -> Any:
        return self

    def __abs__(self) -> Any:
        return np.absolute(self)

    def __lt__(self, other: object) -> Any:
        return np.less(self, other)

    def __le__(self, other: object) -> Any:
        return np.less_equal(self, other)

    def __gt__(self, other: object) -> Any:
        return np.greater(self, other)

    def __ge__(self, other: object) -> Any:
        return np.greater_equal(self, other)

    def __eq__(self, other: object) -> Any:
        return np.equal(self, other)

    def __ne__(self, other: object) -> Any:
        return np.not_equal(self, other)

    __hash__ = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # The NumPy protocols
    # ------------------------------------------------------------------
    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object
    ) -> Any:
        """Apply a NumPy ufunc to jets, by its Taylor rule.

        Returns:
            A jet, a boolean array for comparisons, or ``NotImplemented`` for
            operations without a rule (NumPy then raises :class:`TypeError`).
        """
        if method != "__call__" or kwargs:
            return NotImplemented
        if ufunc in _VALUE_UFUNCS:
            return ufunc(*(_values(i) for i in inputs))
        rule = _RULES.get(ufunc)
        if rule is None:
            return NotImplemented
        jets = _aligned(inputs)
        if jets is None:
            return NotImplemented
        return rule(*jets)

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Sequence[type],
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        """Support the NumPy functions that piecewise code commonly uses.

        Returns:
            The result, or ``NotImplemented`` (NumPy then raises
            :class:`TypeError`).
        """
        handler = _FUNCTIONS.get(func)
        if handler is None:
            return NotImplemented
        return handler(*args, **kwargs)


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------
def _values(item: object) -> Any:
    """Return the values of a jet, or the item itself."""
    return item.c[0] if isinstance(item, Jet) else item


def _aligned(inputs: Sequence[object]) -> list[Coefficients] | None:
    """Turn ufunc inputs into coefficient arrays of one order and shape.

    Args:
        inputs: Jets, numbers and arrays.

    Returns:
        The coefficient arrays, broadcast together (read-only views), or
        ``None`` if an input is not numeric.
    """
    fast = _same_shape(inputs)
    if fast is not None:
        return fast
    order = max(item.order for item in inputs if isinstance(item, Jet))
    arrays = []
    for item in inputs:
        if isinstance(item, Jet):
            arrays.append(item.c)
            continue
        value = np.asarray(item)
        if not (np.issubdtype(value.dtype, np.number) or value.dtype == np.bool_):
            return None
        if np.iscomplexobj(value):
            return None
        arrays.append(Jet.constant(value, order).c)
    shape = np.broadcast_shapes(*(a.shape[1:] for a in arrays))
    result = []
    for a in arrays:
        expanded = a.reshape(
            (a.shape[0],) + (1,) * (len(shape) - (a.ndim - 1)) + a.shape[1:]
        )
        result.append(np.broadcast_to(expanded, (order + 1, *shape)))
    return result


def _same_shape(inputs: Sequence[object]) -> list[Coefficients] | None:
    """The common case of :func:`_aligned`: jets of one shape, and plain numbers.

    Args:
        inputs: The ufunc's inputs.

    Returns:
        The coefficient arrays, or ``None`` if the general case is needed.
    """
    shape = None
    for item in inputs:
        if isinstance(item, Jet):
            if shape is None:
                shape = item.c.shape
            elif item.c.shape != shape:
                return None
        elif not isinstance(item, int | float | np.floating | np.integer):
            return None
    if shape is None:
        return None
    result = []
    for item in inputs:
        if isinstance(item, Jet):
            result.append(item.c)
        else:
            c = np.zeros(shape)
            c[0] = item
            result.append(c)
    return result


def _series(c0: NDArray[np.float64], like: Coefficients) -> Coefficients:
    """Return an array for a result, with its value set.

    Args:
        c0: The result's values.
        like: An input of the right order and shape.

    Returns:
        A writable array, zero beyond the value.
    """
    c = np.zeros(like.shape)
    c[0] = c0
    return c


def _sum_products(a: Coefficients, b: Coefficients, k: int) -> NDArray[np.float64]:
    """Return ``Σ a[j] b[k - j]`` for ``j = 0 ... k`` (one Cauchy coefficient)."""
    return np.asarray(np.sum(a[: k + 1] * b[k::-1], axis=0))


def _weighted(a: Coefficients, b: Coefficients, k: int) -> NDArray[np.float64]:
    """Return ``Σ j a[j] b[k - j]`` for ``j = 1 ... k`` (the ODE recurrences)."""
    j = np.arange(1, k + 1).reshape((-1,) + (1,) * (a.ndim - 1))
    return np.asarray(np.sum(j * a[1 : k + 1] * b[k - 1 :: -1][:k], axis=0))


# ----------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------
def _mul(a: Coefficients, b: Coefficients) -> Coefficients:
    """Product: the Cauchy product ``c[k] = Σ a[j] b[k - j]``."""
    c = np.zeros(a.shape)
    for k in range(a.shape[0]):
        c[k] = _sum_products(a, b, k)
    return c


def _div(a: Coefficients, b: Coefficients) -> Coefficients:
    """Quotient: ``c[k] = (a[k] - Σ_{j<k} c[j] b[k-j]) / b[0]``."""
    c = np.zeros(a.shape)
    for k in range(a.shape[0]):
        acc = a[k].copy()
        for j in range(k):
            acc -= c[j] * b[k - j]
        c[k] = acc / b[0]
    return c


def _exp(a: Coefficients, c0: NDArray[np.float64] | None = None) -> Coefficients:
    """``y = exp(u)``: ``y' = y u'`` (``c0`` overrides the value, for ``expm1``)."""
    c = _series(np.exp(a[0]), a)
    for k in range(1, a.shape[0]):
        c[k] = _weighted(a, c, k) / k
    if c0 is not None:
        c[0] = c0
    return c


def _log(a: Coefficients) -> Coefficients:
    """``y = log(u)``: ``u y' = u'``."""
    c = _series(np.log(a[0]), a)
    for k in range(1, a.shape[0]):
        # c[k] is still 0 here, so the j = k term of the sum vanishes.
        c[k] = (a[k] - _weighted(c, a, k) / k) / a[0]
    return c


def _power_constant(
    a: Coefficients, p: NDArray[np.float64], c0: NDArray[np.float64]
) -> Coefficients:
    """``y = u^p``, ``p`` constant: ``u y' = p y u'``.

    Valid wherever ``u ≠ 0``; at ``u = 0`` the derivatives are infinite or
    undefined (unless ``p`` is a whole number, handled by :func:`_power_whole`).
    """
    c = _series(c0, a)
    n = a.shape[0]
    for k in range(1, n):
        j = np.arange(1, k + 1).reshape((-1,) + (1,) * (a.ndim - 1))
        c[k] = np.sum((p * j - (k - j)) * a[1 : k + 1] * c[k - 1 :: -1][:k], axis=0) / (
            k * a[0]
        )
    return c


def _power_whole(a: Coefficients, p: int) -> Coefficients:
    """``y = u^p`` for a whole ``p``, by repeated squaring (exact even at ``u = 0``)."""
    result = _series(np.ones(a.shape[1:]), a)
    base = a
    k = abs(p)
    while k:
        if k & 1:
            result = _mul(result, base)
        k >>= 1
        if k:
            base = _mul(base, base)
    if p < 0:
        return _div(_series(np.ones(a.shape[1:]), a), result)
    return result


def _power(a: Coefficients, b: Coefficients) -> Coefficients:
    """``y = u^v``.

    Whole constant powers are exact (repeated squaring), other constant
    powers use their recurrence, and varying exponents ``exp(v log u)``.
    """
    if not np.any(b[1:]):
        p = b[0]
        first = p.flat[0] if p.size else 0.0
        if np.all(p == first) and float(first).is_integer() and abs(first) <= 64:
            return _power_whole(a, int(first))
        with np.errstate(all="ignore"):
            return _power_constant(a, p, np.power(a[0], p))
    return _exp(_mul(b, _log(a)))


def _sin_cos(a: Coefficients) -> tuple[Coefficients, Coefficients]:
    """``s = sin u``, ``c = cos u``: ``s' = c u'``, ``c' = -s u'``."""
    s = _series(np.sin(a[0]), a)
    c = _series(np.cos(a[0]), a)
    for k in range(1, a.shape[0]):
        s[k] = _weighted(a, c, k) / k
        c[k] = -_weighted(a, s, k) / k
    return s, c


def _sinh_cosh(a: Coefficients) -> tuple[Coefficients, Coefficients]:
    """``s = sinh u``, ``c = cosh u``: ``s' = c u'``, ``c' = s u'``."""
    s = _series(np.sinh(a[0]), a)
    c = _series(np.cosh(a[0]), a)
    for k in range(1, a.shape[0]):
        s[k] = _weighted(a, c, k) / k
        c[k] = _weighted(a, s, k) / k
    return s, c


def _riccati(a: Coefficients, c0: NDArray[np.float64], sign: float) -> Coefficients:
    """``y' = (1 + sign · y²) u'``: ``tan`` (sign +1) and ``tanh`` (sign -1).

    Solving the equation keeps ``tanh`` finite where ``sinh / cosh`` would
    overflow, and ``tan`` free of a separate division.
    """
    y = _series(c0, a)
    w = _series(1.0 + sign * c0 * c0, a)  # w = 1 + sign · y², updated as y grows
    for k in range(1, a.shape[0]):
        y[k] = _weighted(a, w, k) / k
        w[k] = sign * _sum_products(y, y, k)
    return y


def _from_derivative(
    a: Coefficients,
    c0: NDArray[np.float64],
    derivative: Callable[[Coefficients], Coefficients],
) -> Coefficients:
    """``y = f(u)`` from ``f'``: integrate ``y' = f'(u) u'`` term by term.

    Args:
        a: The coefficients of ``u``.
        c0: The values ``f(u)``.
        derivative: Computes the coefficients of ``f'(u)`` from those of
            ``u`` one order lower.

    Returns:
        The coefficients of ``y``.
    """
    n = a.shape[0] - 1
    c = _series(c0, a)
    if n == 0:
        return c
    slope = derivative(np.ascontiguousarray(a[:n]))
    k = np.arange(1, n + 1).reshape((-1,) + (1,) * (a.ndim - 1))
    du = a[1:] * k  # u'(x) as a series, one order lower
    c[1:] = _mul(slope, du) / k
    return c


def _constant_like(value: float, a: Coefficients) -> Coefficients:
    """A constant with the order and shape of ``a``."""
    return _series(np.full(a.shape[1:], value), a)


def _inverse_sqrt(
    a: Coefficients, inner: Callable[[Coefficients], Coefficients]
) -> Coefficients:
    """``inner(u) ** -1/2``, for the derivatives of the inverse functions."""
    q = inner(a)
    with np.errstate(all="ignore"):
        return _power_constant(q, np.array(-0.5), q[0] ** -0.5)


def _one_minus_square(a: Coefficients) -> Coefficients:
    return _constant_like(1.0, a) - _mul(a, a)


def _one_plus_square(a: Coefficients) -> Coefficients:
    return _constant_like(1.0, a) + _mul(a, a)


def _square_minus_one(a: Coefficients) -> Coefficients:
    return _mul(a, a) - _constant_like(1.0, a)


def _flat(op: Callable[..., Any]) -> Callable[..., Coefficients]:
    """A piecewise-constant function: its value, and zero derivatives."""

    def rule(*args: Coefficients) -> Coefficients:
        return _series(op(*(arg[0] for arg in args)), args[0])

    return rule


def _select(
    condition: NDArray[np.bool_], a: Coefficients, b: Coefficients
) -> Coefficients:
    """Take ``a`` where ``condition`` holds and ``b`` elsewhere, per point."""
    return np.where(condition[np.newaxis], a, b)


def _remainder(
    op: Callable[..., Any],
) -> Callable[[Coefficients, Coefficients], Coefficients]:
    """``u mod v = u - v · floor(u / v)``, the floor being locally constant."""

    def rule(a: Coefficients, b: Coefficients) -> Coefficients:
        quotient = (
            np.floor(a[0] / b[0]) if op is np.remainder else np.trunc(a[0] / b[0])
        )
        c: Coefficients = np.asarray(a - b * quotient[np.newaxis], dtype=np.float64)
        c[0] = op(a[0], b[0])
        return c

    return rule


def _arctan2(a: Coefficients, b: Coefficients) -> Coefficients:
    """``y = atan2(u, v)``: ``y' = (v u' - u v') / (u² + v²)``."""
    n = a.shape[0] - 1
    c = _series(np.arctan2(a[0], b[0]), a)
    if n == 0:
        return c
    k = np.arange(1, n + 1).reshape((-1,) + (1,) * (a.ndim - 1))
    du, dv = a[1:] * k, b[1:] * k
    low_a, low_b = np.ascontiguousarray(a[:n]), np.ascontiguousarray(b[:n])
    numerator = _mul(low_b, du) - _mul(low_a, dv)
    denominator = _mul(low_a, low_a) + _mul(low_b, low_b)
    c[1:] = _div(numerator, denominator) / k
    return c


def _wrap(rule: Callable[..., Coefficients]) -> Callable[..., Jet]:
    """Turn a rule on coefficient arrays into one on jets (errors silenced)."""

    def apply(*arrays: Coefficients) -> Jet:
        with np.errstate(all="ignore"):
            c = np.array(rule(*arrays), dtype=np.float64)
        # Where a value is undefined (log of a negative, a pole), so are all
        # its derivatives, whatever the recurrence made of them.
        undefined = ~np.isfinite(c[0])
        if undefined.any():
            c[:, undefined] = np.nan
        return Jet(c)

    return apply


_LN2: Final = math.log(2.0)
_LN10: Final = math.log(10.0)

_RAW_RULES: dict[np.ufunc, Callable[..., Coefficients]] = {
    np.add: lambda a, b: a + b,
    np.subtract: lambda a, b: a - b,
    np.negative: lambda a: -a,
    np.positive: lambda a: a.copy(),
    np.multiply: _mul,
    np.true_divide: _div,
    np.reciprocal: lambda a: _div(_constant_like(1.0, a), a),
    np.square: lambda a: _mul(a, a),
    np.power: _power,
    np.float_power: _power,
    np.sqrt: lambda a: _power_constant(a, np.array(0.5), np.sqrt(a[0])),
    np.cbrt: lambda a: _power_constant(a, np.array(1 / 3), np.cbrt(a[0])),
    np.exp: lambda a: _exp(a),
    np.exp2: lambda a: _exp(a * _LN2, np.exp2(a[0])),
    np.expm1: lambda a: _exp(a, np.expm1(a[0])),
    np.log: _log,
    np.log2: lambda a: _log(a) / _LN2,
    np.log10: lambda a: _log(a) / _LN10,
    np.log1p: lambda a: _from_derivative(
        a,
        np.log1p(a[0]),
        lambda u: _div(_constant_like(1.0, u), u + _constant_like(1.0, u)),
    ),
    np.sin: lambda a: _sin_cos(a)[0],
    np.cos: lambda a: _sin_cos(a)[1],
    np.tan: lambda a: _riccati(a, np.tan(a[0]), 1.0),
    np.sinh: lambda a: _sinh_cosh(a)[0],
    np.cosh: lambda a: _sinh_cosh(a)[1],
    np.tanh: lambda a: _riccati(a, np.tanh(a[0]), -1.0),
    np.arcsin: lambda a: _from_derivative(
        a, np.arcsin(a[0]), lambda u: _inverse_sqrt(u, _one_minus_square)
    ),
    np.arccos: lambda a: _from_derivative(
        a, np.arccos(a[0]), lambda u: -_inverse_sqrt(u, _one_minus_square)
    ),
    np.arctan: lambda a: _from_derivative(
        a,
        np.arctan(a[0]),
        lambda u: _div(_constant_like(1.0, u), _one_plus_square(u)),
    ),
    np.arcsinh: lambda a: _from_derivative(
        a, np.arcsinh(a[0]), lambda u: _inverse_sqrt(u, _one_plus_square)
    ),
    np.arccosh: lambda a: _from_derivative(
        a, np.arccosh(a[0]), lambda u: _inverse_sqrt(u, _square_minus_one)
    ),
    np.arctanh: lambda a: _from_derivative(
        a,
        np.arctanh(a[0]),
        lambda u: _div(_constant_like(1.0, u), _one_minus_square(u)),
    ),
    np.arctan2: _arctan2,
    np.hypot: lambda a, b: _power_constant(
        _mul(a, a) + _mul(b, b), np.array(0.5), np.hypot(a[0], b[0])
    ),
    np.deg2rad: lambda a: a * (math.pi / 180),
    np.rad2deg: lambda a: a * (180 / math.pi),
    np.absolute: lambda a: a * np.sign(a[0])[np.newaxis],
    np.fabs: lambda a: a * np.sign(a[0])[np.newaxis],
    np.sign: _flat(np.sign),
    np.floor: _flat(np.floor),
    np.ceil: _flat(np.ceil),
    np.trunc: _flat(np.trunc),
    np.rint: _flat(np.rint),
    np.floor_divide: _flat(np.floor_divide),
    np.heaviside: _flat(np.heaviside),
    np.remainder: _remainder(np.remainder),
    np.fmod: _remainder(np.fmod),
    np.maximum: lambda a, b: _select(a[0] >= b[0], a, b),
    np.minimum: lambda a, b: _select(a[0] <= b[0], a, b),
    np.fmax: lambda a, b: _select((a[0] >= b[0]) | np.isnan(b[0]), a, b),
    np.fmin: lambda a, b: _select((a[0] <= b[0]) | np.isnan(b[0]), a, b),
}

_RULES: dict[np.ufunc, Callable[..., Jet]] = {
    ufunc: _wrap(rule) for ufunc, rule in _RAW_RULES.items()
}

_VALUE_UFUNCS: Final = frozenset(
    {
        np.greater,
        np.greater_equal,
        np.less,
        np.less_equal,
        np.equal,
        np.not_equal,
        np.logical_and,
        np.logical_or,
        np.logical_xor,
        np.logical_not,
        np.isfinite,
        np.isinf,
        np.isnan,
        np.signbit,
    }
)
"""Ufuncs that act on the values only (comparisons and tests)."""


# ----------------------------------------------------------------------
# NumPy functions
# ----------------------------------------------------------------------
def _as_jet(item: object, order: int) -> Jet:
    """Return a jet for a value (constants have zero derivatives)."""
    return (
        item
        if isinstance(item, Jet)
        else Jet.constant(np.asarray(item, dtype=float), order)
    )


def _order_of(*items: object) -> int:
    orders = [item.order for item in items if isinstance(item, Jet)]
    return max(orders) if orders else 0


def _where(condition: object, a: object = None, b: object = None) -> Any:
    if a is None or b is None:
        return NotImplemented
    order = _order_of(condition, a, b)
    jets = _aligned([_as_jet(a, order), _as_jet(b, order)])
    if jets is None:
        return NotImplemented
    cond = np.broadcast_to(
        np.asarray(_values(condition), dtype=bool), jets[0].shape[1:]
    )
    return Jet(_select(cond, jets[0], jets[1]))


def _clip(a: object, low: object = None, high: object = None, **kwargs: object) -> Any:
    low = kwargs.pop("a_min", kwargs.pop("min", low))
    high = kwargs.pop("a_max", kwargs.pop("max", high))
    if kwargs:
        return NotImplemented
    result = a
    if high is not None:
        result = np.minimum(result, high)
    if low is not None:
        result = np.maximum(result, low)
    return result


def _like(fill: float | None) -> Callable[..., Any]:
    def make(
        a: object, fill_value: object = None, dtype: object = None, **kwargs: object
    ) -> Any:
        if kwargs or not isinstance(a, Jet):
            return NotImplemented
        value = fill if fill is not None else fill_value
        return Jet.constant(np.full(a.shape, _values(value), dtype=float), a.order)

    return make


def _piecewise(
    x: object,
    conditions: object,
    functions: Sequence[object],
    *args: object,
    **kw: object,
) -> Any:
    if not isinstance(x, Jet) or args or kw:
        return NotImplemented
    condition_list = (
        conditions if isinstance(conditions, (list, tuple)) else [conditions]
    )
    masks = [
        np.broadcast_to(np.asarray(_values(c), dtype=bool), x.shape)
        for c in condition_list
    ]
    if len(functions) == len(masks) + 1:
        masks.append(~np.any(masks, axis=0) if masks else np.ones(x.shape, dtype=bool))
    result = Jet.constant(np.zeros(x.shape), x.order)
    for mask, function in zip(masks, functions, strict=False):
        piece = function(x) if callable(function) else function
        jets = _aligned([result, _as_jet(piece, x.order)])
        if jets is None:
            return NotImplemented
        result = Jet(_select(mask, jets[1], jets[0]))
    return result


def _select_function(
    conditions: Sequence[object], choices: Sequence[object], default: object = 0
) -> Any:
    order = _order_of(*choices, default)
    result: Any = _as_jet(default, order)
    for condition, choice in reversed(list(zip(conditions, choices, strict=True))):
        result = _where(condition, choice, result)
    return result


_FUNCTIONS: dict[Callable[..., Any], Callable[..., Any]] = {
    np.where: _where,
    np.clip: _clip,
    np.zeros_like: _like(0.0),
    np.ones_like: _like(1.0),
    np.full_like: _like(None),
    np.empty_like: _like(0.0),
    np.piecewise: _piecewise,
    np.select: _select_function,
    np.shape: lambda a: a.shape,
    np.ndim: lambda a: a.ndim,
    np.size: lambda a, axis=None: a.size if axis is None else a.shape[axis],
}
