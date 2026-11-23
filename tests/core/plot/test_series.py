"""Tests for :func:`pygridplot.taylor` and :func:`pygridplot.fourier`."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import pygridplot as pgp
from pygridplot.core.plot.series import MAX_TERMS, FourierSeries, TaylorPolynomial

X = np.linspace(-4.0, 4.0, 161)


# --- Taylor polynomials ------------------------------------------------------


def test_taylor_coefficients_and_values() -> None:
    p = pgp.taylor(np.sin, 0, 5)
    assert isinstance(p, TaylorPolynomial)
    np.testing.assert_allclose(
        p.coefficients, [0, 1, 0, -1 / 6, 0, 1 / 120], rtol=1e-15, atol=1e-17
    )
    assert p.degree == 5
    assert p.center == 0.0
    np.testing.assert_allclose(p(X), X - X**3 / 6 + X**5 / 120, rtol=1e-14, atol=1e-14)


def test_taylor_about_another_point() -> None:
    p = pgp.taylor(np.exp, 1.0, 8)
    np.testing.assert_allclose(
        p.coefficients, [math.e / math.factorial(k) for k in range(9)], rtol=1e-14
    )
    assert p(1.0) == pytest.approx(math.e, rel=1e-15)
    assert p(1.5) == pytest.approx(math.exp(1.5), rel=1e-7)


def test_polynomials_are_their_own_taylor_polynomials() -> None:
    f = lambda x: 2 - 3 * x + 0.5 * x**4  # noqa: E731
    p = pgp.taylor(f, -0.7, 6)
    np.testing.assert_allclose(p(X), f(X), rtol=1e-12, atol=1e-12)
    assert p.coefficients[5] == 0.0


def test_numbers_in_numbers_out() -> None:
    p = pgp.taylor(np.cos, 0.3, 4)
    assert isinstance(p(1.0), float)
    assert p(np.ones((2, 3))).shape == (2, 3)
    q = pgp.taylor(np.cos, 0.3, 0)  # a constant
    assert q(np.zeros(4)).shape == (4,)
    assert q(5.0) == pytest.approx(math.cos(0.3))


def test_taylor_polynomials_differentiate_exactly() -> None:
    """Their derivatives at the centre are the function's."""
    f = lambda x: np.exp(np.sin(x))  # noqa: E731
    p = pgp.taylor(f, 0.4, 6)
    np.testing.assert_allclose(
        pgp.derivatives(p, 0.4, 6), pgp.derivatives(f, 0.4, 6), rtol=1e-12
    )


def test_taylor_where_the_function_is_undefined() -> None:
    p = pgp.taylor(np.log, -1.0, 3)
    assert np.isnan(p.coefficients).all()
    assert math.isnan(p(2.0))


def test_math_functions_are_differentiated_numerically() -> None:
    p = pgp.taylor(math.sin, 0.0, 3)
    np.testing.assert_allclose(p.coefficients, [0, 1, 0, -1 / 6], atol=1e-6)


def test_taylor_coefficients_are_read_only() -> None:
    p = pgp.taylor(np.sin, 0, 3)
    with pytest.raises(ValueError, match="read-only"):
        p.coefficients[0] = 1.0


def test_taylor_arguments_are_validated() -> None:
    with pytest.raises(TypeError, match="function"):
        pgp.taylor(1.0, 0, 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="number"):
        pgp.taylor(np.sin, "0", 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        pgp.taylor(np.sin, math.inf, 2)
    for bad in (-1, 41, 2.5, True):
        with pytest.raises(ValueError, match="order"):
            pgp.taylor(np.sin, 0, bad)  # type: ignore[arg-type]


def test_taylor_repr() -> None:
    assert repr(pgp.taylor(np.sin, 0.5, 3)) == "taylor(sin, 0.5, 3)"


# --- Fourier series ----------------------------------------------------------


def test_square_wave() -> None:
    square = pgp.fourier(np.sign, 9)
    assert isinstance(square, FourierSeries)
    want = [4 / (k * math.pi) if k % 2 else 0.0 for k in range(10)]
    np.testing.assert_allclose(square.sines, want, atol=1e-11)
    np.testing.assert_allclose(square.cosines, 0.0, atol=1e-11)
    assert square.terms == 9
    assert square.period == pytest.approx(2 * math.pi)
    assert square.interval == (-math.pi, math.pi)


def test_sawtooth_and_triangle() -> None:
    saw = pgp.fourier(lambda x: x, 6)
    want = [0.0] + [2 * (-1) ** (k + 1) / k for k in range(1, 7)]
    np.testing.assert_allclose(saw.sines, want, atol=1e-11)
    triangle = pgp.fourier(np.abs, 5)
    want = [math.pi] + [-4 / (math.pi * k**2) if k % 2 else 0.0 for k in range(1, 6)]
    np.testing.assert_allclose(triangle.cosines, want, atol=1e-11)


def test_trigonometric_polynomials_are_reproduced() -> None:
    f = lambda x: 3 + np.cos(2 * x) - 0.5 * np.sin(5 * x)  # noqa: E731
    series = pgp.fourier(f, 7)
    np.testing.assert_allclose(series(X), f(X), atol=1e-12)
    np.testing.assert_allclose(series.cosines, [6, 0, 1, 0, 0, 0, 0, 0], atol=1e-12)


def test_other_intervals_and_periodicity() -> None:
    f = lambda x: x * (1 - x)  # noqa: E731
    wave = pgp.fourier(f, 40, (0, 1))
    assert wave.period == 1.0
    inside = np.linspace(0.05, 0.95, 19)
    np.testing.assert_allclose(wave(inside), f(inside), atol=1e-4)
    np.testing.assert_allclose(wave(inside + 7.0), wave(inside), atol=1e-12)
    np.testing.assert_allclose(wave(inside - 3.0), wave(inside), atol=1e-12)
    far = pgp.fourier(np.cos, 3, (1000.0, 1000.0 + 2 * math.pi))  # periodic: any period
    np.testing.assert_allclose(far.cosines, [0, 1, 0, 0], atol=1e-10)


def test_no_harmonics_is_the_mean() -> None:
    mean = pgp.fourier(lambda x: x**2, 0)
    assert mean(1.0) == pytest.approx(math.pi**2 / 3, rel=1e-12)
    assert mean(np.zeros((2, 2))).shape == (2, 2)


def test_numbers_in_numbers_out_and_undefined_points() -> None:
    square = pgp.fourier(np.sign, 5)
    assert isinstance(square(1.0), float)
    assert square(np.ones((3, 2))).shape == (3, 2)
    assert np.isnan(square(np.array([np.nan, np.inf]))).all()


def test_series_of_undefined_functions_are_nan() -> None:
    assert np.isnan(pgp.fourier(np.sqrt, 3).cosines).all()  # undefined left of 0
    assert math.isnan(pgp.fourier(np.sqrt, 3)(1.0))
    assert math.isnan(pgp.fourier(lambda x: 1 / x, 3)(1.0))  # diverges


def test_scalar_only_functions() -> None:
    series = pgp.fourier(lambda x: math.copysign(1.0, x), 3)
    np.testing.assert_allclose(
        series.sines, [0, 4 / math.pi, 0, 4 / (3 * math.pi)], atol=1e-10
    )


def test_fourier_series_differentiate_exactly() -> None:
    series = pgp.fourier(lambda x: np.cos(2 * x) + np.sin(3 * x), 4)
    slope = pgp.derivative(series)(X)
    np.testing.assert_allclose(
        slope, -2 * np.sin(2 * X) + 3 * np.cos(3 * X), atol=1e-10
    )


def test_many_terms_and_large_arrays() -> None:
    """Evaluation in blocks gives the same values as one point at a time."""
    series = pgp.fourier(np.sign, 300)
    xs = np.linspace(-10, 10, 2001)
    together = series(xs)
    np.testing.assert_allclose(
        together[::250], [series(x) for x in xs[::250]], atol=1e-13
    )
    assert MAX_TERMS >= 1000
    assert pgp.fourier(np.sign, MAX_TERMS).terms == MAX_TERMS


def test_amplitudes_are_the_spectrum() -> None:
    square = pgp.fourier(np.sign, 7)
    want = [0.0] + [4 / (k * math.pi) if k % 2 else 0.0 for k in range(1, 8)]
    np.testing.assert_allclose(square.amplitudes, want, atol=1e-11)
    shifted = pgp.fourier(lambda x: 2 + 3 * np.cos(x) - 4 * np.sin(x), 2)
    np.testing.assert_allclose(shifted.amplitudes, [2, 5, 0], atol=1e-11)
    shifted.amplitudes[0] = 99.0  # a new array each time
    assert shifted.amplitudes[0] == pytest.approx(2.0)


def test_fourier_coefficients_are_read_only() -> None:
    series = pgp.fourier(np.sign, 3)
    with pytest.raises(ValueError, match="read-only"):
        series.sines[1] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        series.cosines[1] = 0.0


def test_fourier_arguments_are_validated() -> None:
    with pytest.raises(TypeError, match="function"):
        pgp.fourier(2.0, 3)  # type: ignore[arg-type]
    for bad in (-1, MAX_TERMS + 1, 2.5, True):
        with pytest.raises(ValueError, match="whole number"):
            pgp.fourier(np.sin, bad)  # type: ignore[arg-type]
    for interval in ((1.0, 1.0), (2.0, 1.0), (0.0, math.inf), (0.0,), 5.0):
        with pytest.raises(ValueError, match="interval"):
            pgp.fourier(np.sin, 3, interval)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="number"):
        pgp.fourier(np.sin, 3, ("a", 1.0))  # type: ignore[arg-type]


def test_fourier_repr() -> None:
    assert repr(pgp.fourier(np.sign, 2, (0, 1))) == "fourier(sign, 2, (0.0, 1.0))"


@settings(max_examples=25, deadline=None)
@given(
    cosines=st.lists(st.floats(-3, 3), min_size=1, max_size=6),
    sines=st.lists(st.floats(-3, 3), min_size=6, max_size=6),
    start=st.floats(-10, 10),
    period=st.floats(0.5, 20),
)
def test_coefficients_of_trigonometric_polynomials_are_exact(
    cosines: list[float], sines: list[float], start: float, period: float
) -> None:
    omega = 2 * math.pi / period
    n = len(cosines)

    def f(x: np.ndarray) -> np.ndarray:
        total = 0.5 * cosines[0] + 0 * x
        for k in range(1, n):
            total = (
                total
                + cosines[k] * np.cos(k * omega * x)
                + sines[k] * np.sin(k * omega * x)
            )
        return total

    series = pgp.fourier(f, n + 1, (start, start + period))
    np.testing.assert_allclose(series.cosines[:n], cosines, atol=1e-9)
    np.testing.assert_allclose(series.sines[1:n], sines[1:n], atol=1e-9)
    np.testing.assert_allclose(series.cosines[n:], 0.0, atol=1e-9)
