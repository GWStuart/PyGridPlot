"""Tests for Taylor arithmetic (:mod:`pygridplot.core.plot.jets`).

Each rule is checked against the derivatives worked out by hand, at many
points and several orders.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from pygridplot.core.plot.jets import Jet

Array = NDArray[np.float64]


def nth(func: Callable[[Jet], object], x: Array, n: int) -> Array:
    """Return the n-th derivative of ``func`` at ``x`` with jets."""
    result = func(Jet.variable(x, n))
    assert isinstance(result, Jet)
    return result.derivatives()[n]


def close(got: Array, want: Array, tolerance: float = 1e-12) -> None:
    """Assert agreement relative to the size of the values."""
    scale = np.maximum(np.abs(want), 1e-3 * np.max(np.abs(want)) + 1e-300)
    assert np.max(np.abs(got - want) / scale) < tolerance


X = np.linspace(-2.3, 2.3, 23)
POSITIVE = np.linspace(0.2, 4.0, 20)
UNIT = np.linspace(-0.9, 0.9, 19)
ABOVE_ONE = np.linspace(1.1, 4.0, 15)

CASES: list[
    tuple[str, Callable[[Jet], object], int, Callable[[Array], Array], Array]
] = [
    ("x + 3", lambda x: x + 3, 1, lambda x: 1 + 0 * x, X),
    ("3 - x", lambda x: 3 - x, 1, lambda x: -1 + 0 * x, X),
    ("x * x * x", lambda x: x * x * x, 3, lambda x: 6 + 0 * x, X),
    ("1 / x", lambda x: 1 / x, 3, lambda x: -6 / x**4, POSITIVE),
    ("x ** 5", lambda x: x**5, 4, lambda x: 120 * x, X),
    ("x ** -2", lambda x: x**-2, 2, lambda x: 6 / x**4, POSITIVE),
    ("x ** 2.5", lambda x: x**2.5, 2, lambda x: 3.75 * x**0.5, POSITIVE),
    (
        "x ** x",
        lambda x: x**x,
        2,
        lambda x: x**x * ((np.log(x) + 1) ** 2 + 1 / x),
        POSITIVE,
    ),
    ("2 ** x", lambda x: 2**x, 3, lambda x: 2**x * math.log(2) ** 3, X),
    ("sqrt", np.sqrt, 2, lambda x: -0.25 * x**-1.5, POSITIVE),
    ("cbrt", np.cbrt, 1, lambda x: 1 / (3 * np.cbrt(x) ** 2), X[X != 0]),
    ("square", np.square, 2, lambda x: 2 + 0 * x, X),
    ("reciprocal", np.reciprocal, 1, lambda x: -1 / x**2, POSITIVE),
    ("exp(2x)", lambda x: np.exp(2 * x), 12, lambda x: 2.0**12 * np.exp(2 * x), X),
    ("exp2", np.exp2, 2, lambda x: np.exp2(x) * math.log(2) ** 2, X),
    ("expm1", np.expm1, 1, np.exp, X),
    ("log", np.log, 3, lambda x: 2 / x**3, POSITIVE),
    ("log2", np.log2, 1, lambda x: 1 / (x * math.log(2)), POSITIVE),
    ("log10", np.log10, 1, lambda x: 1 / (x * math.log(10)), POSITIVE),
    ("log1p", np.log1p, 2, lambda x: -1 / (1 + x) ** 2, UNIT),
    ("sin(3x)", lambda x: np.sin(3 * x), 9, lambda x: 3.0**9 * np.cos(3 * x), X),
    ("cos", np.cos, 6, lambda x: -np.cos(x), X),
    (
        "tan",
        np.tan,
        3,
        lambda x: 2 * (1 + np.tan(x) ** 2) * (1 + 3 * np.tan(x) ** 2),
        UNIT,
    ),
    ("sinh", np.sinh, 3, np.cosh, X),
    ("cosh", np.cosh, 2, np.cosh, X),
    ("tanh", np.tanh, 2, lambda x: -2 * np.tanh(x) / np.cosh(x) ** 2, X),
    ("arcsin", np.arcsin, 2, lambda x: x / (1 - x**2) ** 1.5, UNIT),
    ("arccos", np.arccos, 1, lambda x: -1 / np.sqrt(1 - x**2), UNIT),
    ("arctan", np.arctan, 3, lambda x: (6 * x**2 - 2) / (1 + x**2) ** 3, X),
    ("arcsinh", np.arcsinh, 1, lambda x: 1 / np.sqrt(x**2 + 1), X),
    ("arccosh", np.arccosh, 1, lambda x: 1 / np.sqrt(x**2 - 1), ABOVE_ONE),
    ("arctanh", np.arctanh, 2, lambda x: 2 * x / (1 - x**2) ** 2, UNIT),
    ("arctan2(x, 2)", lambda x: np.arctan2(x, 2), 1, lambda x: 2 / (4 + x**2), X),
    ("arctan2(1, x)", lambda x: np.arctan2(1, x), 1, lambda x: -1 / (1 + x**2), X),
    ("hypot(x, 1)", lambda x: np.hypot(x, 1), 2, lambda x: (x**2 + 1) ** -1.5, X),
    ("degrees", np.rad2deg, 1, lambda x: 180 / math.pi + 0 * x, X),
    ("abs", np.abs, 1, np.sign, X),
    ("floor", np.floor, 1, lambda x: 0 * x, X),
    ("x mod 1", lambda x: x % 1, 1, lambda x: 1 + 0 * x, X),
    (
        "maximum(x, x^2)",
        lambda x: np.maximum(x, x**2),
        1,
        lambda x: np.where(x >= x**2, 1, 2 * x),
        X,
    ),
    (
        "where",
        lambda x: np.where(x > 0, np.sin(x), x**3),
        2,
        lambda x: np.where(x > 0, -np.sin(x), 6 * x),
        X,
    ),
    (
        "clip",
        lambda x: np.clip(x**3, -1, 2),
        1,
        lambda x: np.where(np.abs(x**3 - 0.5) < 1.5, 3 * x**2, 0),
        X,
    ),
    (
        "piecewise",
        lambda x: np.piecewise(x, [x < 0], [lambda t: t**2, lambda t: np.exp(t)]),
        1,
        lambda x: np.where(x < 0, 2 * x, np.exp(x)),
        X,
    ),
    (
        "select",
        lambda x: np.select([x < -1, x < 1], [x, x**2], np.cos(x)),
        1,
        lambda x: np.where(x < -1, 1, np.where(x < 1, 2 * x, -np.sin(x))),
        X,
    ),
    (
        "a mixed function",
        lambda x: np.sin(x) ** 2 * np.exp(-x / 3) + 1 / (1 + x**2),
        1,
        lambda x: (
            np.exp(-x / 3) * (np.sin(2 * x) - np.sin(x) ** 2 / 3)
            - 2 * x / (1 + x**2) ** 2
        ),
        X,
    ),
]


@pytest.mark.parametrize(
    ("name", "func", "n", "exact", "x"), CASES, ids=[c[0] for c in CASES]
)
def test_rules_give_exact_derivatives(
    name: str,
    func: Callable[[Jet], object],
    n: int,
    exact: Callable[[Array], Array],
    x: Array,
) -> None:
    close(nth(func, x, n), exact(x))


@settings(max_examples=60, deadline=None)
@given(
    coefficients=st.lists(st.integers(-9, 9), min_size=1, max_size=8),
    n=st.integers(0, 9),
)
def test_polynomials_match_their_exact_derivatives(
    coefficients: list[int], n: int
) -> None:
    def poly(x: Jet) -> object:
        result: object = 0
        for c in coefficients:  # Horner's rule, with jets
            result = result * x + c
        return result

    x = np.linspace(-3, 3, 13)
    want = np.polyval(np.polyder(np.array(coefficients, dtype=float), n), x)
    got = Jet.variable(x, n)
    value = poly(got)
    derivatives = value.derivatives()[n] if isinstance(value, Jet) else np.zeros_like(x)
    np.testing.assert_allclose(derivatives, want, rtol=1e-12, atol=1e-9)


def test_whole_powers_are_exact_where_the_base_is_zero() -> None:
    at_zero = Jet.variable(np.array([0.0]), 5)
    assert (at_zero**5).derivatives()[5][0] == 120.0
    assert (at_zero**2).derivatives()[2][0] == 2.0


def test_undefined_values_have_undefined_derivatives() -> None:
    x = Jet.variable(np.array([-1.0, 1.0]), 2)
    logged = np.log(x)
    assert np.isnan(logged.derivatives()[:, 0]).all()
    assert logged.derivatives()[1, 1] == 1.0
    assert np.isnan(np.arcsin(Jet.variable(np.array([2.0]), 1)).derivatives()).all()
    pole = (1 / Jet.variable(np.array([0.0]), 1)).derivatives()
    assert np.isnan(pole).all()


def test_comparisons_and_truth() -> None:
    x = Jet.variable(np.array([-1.0, 2.0]), 1)
    assert (x > 0).tolist() == [False, True]
    assert np.isfinite(x).all()
    single = Jet.variable(np.array(3.0), 1)
    assert bool(single > 2)
    with pytest.raises(ValueError, match="ambiguous"):
        bool(x)


def test_jets_refuse_to_become_numbers() -> None:
    x = Jet.variable(np.array(1.0), 1)
    with pytest.raises(TypeError):
        math.sin(x)  # type: ignore[call-overload]
    with pytest.raises(TypeError):
        float(x)
    with pytest.raises(TypeError):
        np.sum(Jet.variable(np.arange(3.0), 1))  # no rule: NumPy raises
    with pytest.raises(TypeError):
        np.gcd(x, 2)


def test_array_helpers() -> None:
    x = Jet.variable(np.linspace(0, 1, 4), 2)
    assert x.shape == (4,)
    assert len(x) == 4
    assert x[1:3].shape == (2,)
    assert np.zeros_like(x).derivatives().max() == 0.0
    filled = np.full_like(x, 7.0)
    assert filled.value.tolist() == [7.0] * 4
    assert x.astype(float).value.tolist() == x.value.tolist()
    with pytest.raises(TypeError):
        x.astype(int)
    assert np.shape(x) == (4,)
