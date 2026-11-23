"""Tests for :func:`pygridplot.derivative` and :func:`pygridplot.derivatives`."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import pygridplot as pgp
from pygridplot.core.plot.derivatives import Derivative

X = np.linspace(-3.0, 3.0, 31)


def test_derivatives_of_numpy_functions_are_exact() -> None:
    np.testing.assert_allclose(pgp.derivative(np.sin)(X), np.cos(X), rtol=0, atol=1e-15)
    np.testing.assert_allclose(pgp.derivative(np.sin, 4)(X), np.sin(X), atol=1e-14)
    np.testing.assert_allclose(
        pgp.derivative(lambda x: np.exp(-(x**2)), 2)(X),
        (4 * X**2 - 2) * np.exp(-(X**2)),
        atol=1e-15,
    )


def test_numbers_in_numbers_out() -> None:
    df = pgp.derivative(lambda x: x**3)
    assert df(2.0) == 12.0
    assert isinstance(df(2.0), float)
    values = np.ones((2, 3))
    assert df(values).shape == (2, 3)


def test_jumps_kinks_poles_and_edges_are_exact() -> None:
    """Where finite differences invent slopes, the derivative is exact."""
    assert pgp.derivative(np.floor)(1 + 1e-7) == 0.0
    assert pgp.derivative(np.abs)(1e-7) == 1.0
    assert pgp.derivative(np.tan)(1.5707) == pytest.approx(1 / math.cos(1.5707) ** 2)
    assert pgp.derivative(np.sqrt)(1e-9) == pytest.approx(0.5 / math.sqrt(1e-9))


def test_undefined_points_are_nan_and_nothing_raises() -> None:
    assert math.isnan(pgp.derivative(np.log)(-1.0))
    assert math.isnan(pgp.derivative(lambda x: 1 / x)(0.0))

    def broken(x: object) -> float:
        raise ZeroDivisionError

    assert math.isnan(pgp.derivative(broken)(1.0))
    assert math.isnan(pgp.derivative(lambda x: "text")(1.0))  # type: ignore[arg-type, return-value]


def test_constant_functions_have_zero_derivatives() -> None:
    assert pgp.derivative(lambda x: 3.0)(1.0) == 0.0
    assert pgp.derivative(lambda x: 0 * x + 2, 3)(X).max() == 0.0


def test_single_value_functions_are_differentiated_exactly_point_by_point() -> None:
    def ramp(x: float) -> float:
        if x > 0:  # a Python condition: needs one value at a time
            return np.sin(x)  # type: ignore[no-any-return]
        return x * x

    got = pgp.derivative(ramp)(X)
    np.testing.assert_allclose(got, np.where(X > 0, np.cos(X), 2 * X), atol=1e-15)


def test_math_functions_are_differentiated_numerically() -> None:
    got = pgp.derivative(lambda x: math.exp(math.sin(x)))(X)
    want = np.cos(X) * np.exp(np.sin(X))
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
    second = pgp.derivative(lambda x: math.sin(x), 2)(X)
    np.testing.assert_allclose(second, -np.sin(X), atol=1e-7)


def test_points_where_a_function_raises_are_undefined() -> None:
    def picky(x: float) -> float:
        if x < 0:
            raise ValueError("negative")
        return x**2

    got = pgp.derivative(picky)(X)
    assert np.isnan(got[X < 0]).all()
    np.testing.assert_allclose(got[X >= 0], 2 * X[X >= 0])


def test_derivatives_of_derivatives_add_their_orders() -> None:
    twice = pgp.derivative(pgp.derivative(np.sin), 2)
    assert isinstance(twice, Derivative)
    assert twice.order == 3
    assert twice.func is np.sin
    np.testing.assert_allclose(twice(X), -np.cos(X), atol=1e-15)
    assert repr(twice) == "derivative(sin, 3)"


def test_a_derivative_inside_another_function_is_differentiated_numerically() -> None:
    df = pgp.derivative(np.sin)
    got = pgp.derivative(lambda x: x * df(x))(X)  # d/dx (x cos x)
    np.testing.assert_allclose(got, np.cos(X) - X * np.sin(X), atol=1e-8)


def test_derivatives_gives_every_order_at_once() -> None:
    value, slope, curvature = pgp.derivatives(np.exp, 0.0, 2)
    assert (value, slope, curvature) == (1.0, 1.0, 1.0)
    table = pgp.derivatives(np.sin, X, 3)
    assert table.shape == (4, X.size)
    np.testing.assert_allclose(table[3], -np.cos(X), atol=1e-15)
    shifted = pgp.derivatives(pgp.derivative(np.sin), 0.0, 1)
    assert shifted == pytest.approx((1.0, 0.0))


def test_order_zero_is_the_function_itself() -> None:
    np.testing.assert_allclose(pgp.derivative(np.cos, 0)(X), np.cos(X))


@pytest.mark.parametrize("bad", [-1, 41, 1.5, True, "2"])
def test_orders_are_validated(bad: object) -> None:
    with pytest.raises(ValueError, match="order"):
        pgp.derivative(np.sin, bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="order"):
        pgp.derivatives(np.sin, 0.0, bad)  # type: ignore[arg-type]


def test_non_functions_are_rejected() -> None:
    with pytest.raises(TypeError, match="function"):
        pgp.derivative(3.0)  # type: ignore[arg-type]


def test_high_orders() -> None:
    got = pgp.derivative(lambda x: np.exp(3 * x), 30)(np.array([0.0, 0.5]))
    np.testing.assert_allclose(
        got, 3.0**30 * np.exp(3 * np.array([0.0, 0.5])), rtol=1e-12
    )


def plot_pixels(surface: object) -> NDArray[np.float64]:
    """Return the darkness of each pixel of a surface, shape (h, w)."""
    import pygame

    rgb = pygame.surfarray.array3d(surface).astype(float)
    return (255.0 - rgb.min(axis=2)).T / 255.0


@pytest.mark.usefixtures("pygame_display")
def test_derivatives_plot_like_functions() -> None:
    import pygame

    theme = pgp.GridTheme(
        show_minor=False, show_major=False, show_axes=False, show_labels=False
    )
    grid = pgp.Grid(pygame.Rect(0, 0, 320, 240), theme=theme)
    surface = pygame.Surface((320, 240))
    grid.draw(surface)
    grid.plot(surface, pgp.derivative(lambda x: x**2 / 2), color=(0, 0, 0))  # y = x
    ink = plot_pixels(surface)
    for x in (-1.0, 0.5, 1.5):
        px, py = (round(v) for v in grid.to_screen(x, x))
        assert ink[py, px] > 0.9


def test_results_that_are_not_derivatives_fall_back() -> None:
    """A function returning the wrong shape or kind is evaluated another way."""
    lengths = pgp.derivative(lambda x: np.zeros(3) if np.ndim(x) else x * 2.0)
    np.testing.assert_allclose(lengths(X), 2.0)  # point by point instead
    as_list = pgp.derivative(lambda x: [x])  # type: ignore[arg-type, return-value]
    assert np.isnan(as_list(X)).all()


def test_an_undefined_constant_has_undefined_derivatives() -> None:
    """Where a function is NaN, its derivatives are NaN too, even for a constant."""
    f = pgp.derivative(lambda x: np.where(x > 0, x**2, np.nan))
    np.testing.assert_array_equal(f(np.array([-1.0, 2.0])), [np.nan, 4.0])
    g = pgp.derivative(lambda x: x + np.where(x > 0, 1.0, np.inf), 2)
    assert np.isnan(g(np.array([-1.0]))).all()
