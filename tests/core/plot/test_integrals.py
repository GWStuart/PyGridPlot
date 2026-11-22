"""Tests for :func:`pygridplot.integral` and :func:`pygridplot.antiderivative`."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

import pygridplot as pgp
from pygridplot.core.plot.derivatives import Derivative
from pygridplot.core.plot.integrals import Antiderivative

Array = NDArray[np.float64]
INF = math.inf


@pytest.mark.parametrize(
    ("f", "a", "b", "exact"),
    [
        (np.sin, 0, math.pi, 2.0),
        (lambda x: x**2, -1, 2, 3.0),
        (np.exp, -1, 3, math.exp(3) - math.exp(-1)),
        (lambda x: np.sin(x) / x, -1, 1, 1.8921661407343662),  # undefined at 0
        (np.floor, 0, 3.5, 4.5),
        (lambda x: 1 / np.sqrt(x), 0, 1, 2.0),
        (np.log, 0, 1, -1.0),
        (lambda x: np.exp(-(x**2)), -INF, INF, math.sqrt(math.pi)),
        (lambda x: 1 / x**2, 1, INF, 1.0),
        (np.exp, -INF, 0, 1.0),
        (lambda x: 1 / (1 + x**2), -INF, INF, math.pi),
        (lambda x: x**2 * np.exp(-x), 0, INF, 2.0),
        (lambda x: x**-1.5, 1, INF, 2.0),
        (math.sin, 0, math.pi, 2.0),  # scalar-only
        (lambda x: math.exp(x), -INF, 0, 1.0),
    ],
)
def test_integrals(f: Callable[[Any], Any], a: float, b: float, exact: float) -> None:
    assert pgp.integral(f, a, b) == pytest.approx(exact, rel=1e-10)


def test_odd_functions_over_the_whole_line_vanish() -> None:
    assert pgp.integral(lambda x: x * np.exp(-(x**2)), -INF, INF) == pytest.approx(
        0.0, abs=1e-15
    )


@pytest.mark.parametrize(
    ("f", "a", "b"),
    [
        (lambda x: 1 / x, -1, 1),  # diverges
        (lambda x: 1 / x, 0, 1),
        (lambda x: 1 / x, 1, INF),
        (lambda x: 1 / x**2, -1, 2),
        (np.tan, 0, 3),
        (np.sqrt, -1, 1),  # undefined left of 0
        (lambda x: np.log(x - 2), 0, 3),
        (lambda x: 0 * x + 1, 0, INF),
    ],
)
def test_integrals_that_do_not_exist_are_nan(
    f: Callable[[Any], Any], a: float, b: float
) -> None:
    assert math.isnan(pgp.integral(f, a, b))


def test_limits() -> None:
    assert pgp.integral(np.sin, math.pi, 0) == pytest.approx(-2.0, rel=1e-14)
    assert pgp.integral(np.log, -1.0, -1.0) == 0.0  # empty, even where undefined
    assert pgp.integral(np.exp, INF, INF) == 0.0
    assert pgp.integral(lambda x: np.exp(-x), INF, 0) == pytest.approx(-1.0)
    assert math.isnan(pgp.integral(np.sin, math.nan, 1.0))
    assert math.isnan(pgp.integral(np.sin, 0.0, math.nan))
    assert pgp.integral(np.cos, np.float32(0), np.int64(1)) == pytest.approx(
        math.sin(1)
    )
    assert isinstance(pgp.integral(np.cos, 0, 1), float)


def test_broken_functions_never_raise() -> None:
    def broken(x: object) -> float:
        raise ZeroDivisionError

    assert math.isnan(pgp.integral(broken, 0, 1))
    assert math.isnan(pgp.integral(lambda x: "text", 0, 1))  # type: ignore[arg-type, return-value]
    assert pgp.integral(lambda x: 2.0, 0, 3) == pytest.approx(6.0)  # a constant


def test_bad_arguments_are_rejected() -> None:
    with pytest.raises(TypeError, match="function"):
        pgp.integral(3.0, 0, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="limit"):
        pgp.integral(np.sin, "0", 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="limit"):
        pgp.integral(np.sin, 0, None)  # type: ignore[arg-type]


# --- Antiderivatives -------------------------------------------------------

X = np.linspace(-10.0, 10.0, 801)


def test_antiderivatives_match_closed_forms() -> None:
    np.testing.assert_allclose(pgp.antiderivative(np.cos)(X), np.sin(X), atol=1e-14)
    anti = pgp.antiderivative(lambda x: x**2, 1.0)
    np.testing.assert_allclose(anti(X), (X**3 - 1) / 3, rtol=1e-12, atol=1e-12)
    steps = pgp.antiderivative(np.floor)
    assert steps(2.5) == pytest.approx(2.0)  # 0 + 1 + 2 · 0.5
    assert steps(-1.5) == pytest.approx(2.0)  # -(-1 - 2 · 0.5)


def test_numbers_in_numbers_out() -> None:
    anti = pgp.antiderivative(np.cos)
    assert isinstance(anti(1.0), float)
    assert anti(1.0) == pytest.approx(math.sin(1.0), rel=1e-14)
    assert anti(np.ones((2, 3))).shape == (2, 3)
    assert anti(np.zeros(0)).shape == (0,)
    assert anti(0.0) == 0.0  # at the lower limit
    np.testing.assert_array_equal(anti(np.array([np.nan, 0.0])), [np.nan, 0.0])


def test_the_same_point_twice_and_unsorted_points() -> None:
    anti = pgp.antiderivative(np.exp)
    got = anti(np.array([2.0, -1.0, 2.0, 0.5]))
    np.testing.assert_allclose(got, np.exp([2.0, -1.0, 2.0, 0.5]) - 1, rtol=1e-13)
    assert got[0] == got[2]


def test_divergence_makes_the_antiderivative_undefined_beyond_it() -> None:
    log = pgp.antiderivative(lambda x: 1 / x, 1.0)
    got = log(np.array([-2.0, -0.5, 0.5, 1.0, math.e]))
    assert np.isnan(got[:2]).all()
    np.testing.assert_allclose(got[2:], [math.log(0.5), 0.0, 1.0], atol=1e-14)
    root = pgp.antiderivative(np.sqrt, 1.0)  # undefined left of 0
    assert np.isnan(root(-0.5))
    assert root(4.0) == pytest.approx(14 / 3, rel=1e-12)


def test_integrable_singularities_at_the_lower_limit() -> None:
    anti = pgp.antiderivative(lambda x: 1 / np.sqrt(np.abs(x)))
    np.testing.assert_allclose(
        anti(np.array([-4.0, -1.0, 1e-6, 1.0, 9.0])),
        [-4.0, -2.0, 2e-3, 2.0, 6.0],
        rtol=1e-9,
    )


def test_infinite_limits_and_points() -> None:
    normal = lambda x: np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)  # noqa: E731
    cdf = pgp.antiderivative(normal, -INF)
    got = cdf(np.array([-INF, -1.96, 0.0, 1.96, INF]))
    np.testing.assert_allclose(got, [0.0, 0.0249979, 0.5, 0.9750021, 1.0], atol=1e-7)
    assert cdf(INF) == pytest.approx(1.0, rel=1e-12)
    assert cdf(-INF) == 0.0
    tail = pgp.antiderivative(lambda x: np.exp(-x), INF)  # anti(x) = -∫ₓ^∞
    np.testing.assert_allclose(
        tail(np.array([0.0, 1.0, INF])), [-1.0, -math.exp(-1), 0.0]
    )
    anti = pgp.antiderivative(lambda x: 1 / (1 + x**2))
    assert anti(INF) == pytest.approx(math.pi / 2, rel=1e-12)
    assert anti(-INF) == pytest.approx(-math.pi / 2, rel=1e-12)
    assert np.isnan(pgp.antiderivative(lambda x: 0 * x + 1)(INF))  # diverges


def test_antiderivatives_of_scalar_only_and_broken_functions() -> None:
    anti = pgp.antiderivative(math.cos)
    np.testing.assert_allclose(
        anti(np.array([0.5, 2.0])), np.sin([0.5, 2.0]), rtol=1e-13
    )

    def broken(x: object) -> float:
        raise ValueError

    assert np.isnan(pgp.antiderivative(broken)(np.array([1.0, 2.0]))).all()


def test_derivatives_of_antiderivatives_are_the_function_exactly() -> None:
    f = lambda x: np.exp(-x) * np.sin(3 * x)  # noqa: E731
    anti = pgp.antiderivative(f, 2.0)
    df = pgp.derivative(anti)
    assert isinstance(df, Derivative)
    assert df.func is f
    assert df.order == 0
    np.testing.assert_array_equal(df(X), pgp.derivative(f, 0)(X))
    assert pgp.derivative(anti, 3).order == 2
    assert pgp.derivative(anti, 0).func is anti  # the antiderivative itself
    value, slope, bend = pgp.derivatives(anti, 1.0, 2)
    assert value == pytest.approx(anti(1.0))
    assert (slope, bend) == pgp.derivatives(f, 1.0, 1)
    assert pgp.derivatives(anti, X, 2).shape == (3, X.size)
    assert pgp.derivative(pgp.derivative(anti), 1)(1.0) == pytest.approx(
        pgp.derivatives(f, 1.0, 1)[1]
    )


def test_antiderivatives_inside_differentiated_functions() -> None:
    """A jet cannot pass through: such a function is differentiated numerically."""
    anti = pgp.antiderivative(np.cos)
    got = pgp.derivative(lambda x: x * anti(x))(np.array([0.5, 1.0]))
    want = np.sin([0.5, 1.0]) + np.array([0.5, 1.0]) * np.cos([0.5, 1.0])
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_antiderivative_arguments_are_validated() -> None:
    with pytest.raises(TypeError, match="function"):
        pgp.antiderivative(2.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="limit"):
        pgp.antiderivative(np.sin, "1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="NaN"):
        pgp.antiderivative(np.sin, math.nan)


def test_repr() -> None:
    anti = pgp.antiderivative(np.cos, 1.5)
    assert isinstance(anti, Antiderivative)
    assert repr(anti) == "antiderivative(cos, 1.5)"


@settings(max_examples=40, deadline=None)
@given(
    a=st.floats(-20, 20),
    b=st.floats(-20, 20),
    lower=st.floats(-20, 20),
    k=st.floats(0.1, 5),
)
def test_antiderivatives_agree_with_integrals(
    a: float, b: float, lower: float, k: float
) -> None:
    """F(b) - F(a) is the integral from a to b, whatever the lower limit."""
    f = lambda x: np.cos(k * x) + np.abs(x - 1.0)  # noqa: E731
    anti = pgp.antiderivative(f, lower)
    fa, fb = anti(np.array([a, b]))
    scale = pgp.integral(lambda x: np.abs(f(x)), min(a, b, lower), max(a, b, lower))
    assert abs((fb - fa) - pgp.integral(f, a, b)) <= 1e-9 * scale + 1e-12


@settings(max_examples=30, deadline=None)
@given(points=st.lists(st.floats(-30, 30), min_size=1, max_size=40))
def test_values_barely_depend_on_the_other_points(points: list[float]) -> None:
    """The same x evaluated with different neighbours agrees to rounding."""
    anti = pgp.antiderivative(lambda x: np.sin(x) * x)
    together = anti(np.array(points))
    alone = np.array([anti(p) for p in points])
    scale = 1 + np.abs(np.array(points)) ** 2
    assert np.all(np.abs(together - alone) <= 1e-10 * scale)
