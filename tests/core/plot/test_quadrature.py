"""Tests for the adaptive quadrature (:mod:`pygridplot.core.plot.quadrature`)."""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from pygridplot.core.plot import quadrature
from pygridplot.core.plot.evaluation import evaluate_quietly
from pygridplot.core.plot.quadrature import (
    GAUSS,
    KRONROD,
    MAX_NEW_INTERVALS,
    NODES,
    Quadrature,
    integrate,
    nodes_of,
)

Array = NDArray[np.float64]


def run(
    f: Callable[[Array], Array], a: float, b: float, parts: int = 8, **kw: bool
) -> Quadrature:
    """Integrate a NumPy function over [a, b], from equal parts, as one piece."""
    edges = np.linspace(a, b, parts + 1)
    return integrate(
        lambda x: evaluate_quietly(f, x),
        edges[:-1],
        edges[1:],
        np.zeros(parts, dtype=np.intp),
        1,
        **kw,
    )


def value(f: Callable[[Array], Array], a: float, b: float) -> float:
    return float(run(f, a, b).values[0])


def counting(f: Callable[[Array], Array]) -> tuple[Callable[[Array], Array], list[int]]:
    """Wrap a function to count the points it is evaluated at, per call."""
    calls: list[int] = []

    def g(x: Array) -> Array:
        calls.append(x.size)
        return evaluate_quietly(f, x)

    return g, calls


def test_the_rules_are_exact_for_polynomials_of_their_degree() -> None:
    assert NODES.shape == KRONROD.shape == GAUSS.shape == (21,)
    assert KRONROD.sum() == pytest.approx(2.0, abs=1e-15)
    assert np.count_nonzero(GAUSS) == 10
    for degree in range(32):
        exact = 0.0 if degree % 2 else 2.0 / (degree + 1)
        assert KRONROD @ NODES**degree == pytest.approx(exact, abs=1e-15)
        if degree < 20:
            assert GAUSS @ NODES**degree == pytest.approx(exact, abs=1e-15)
    # Degree 20 is beyond the Gauss rule: the difference that estimates errors.
    assert abs(GAUSS @ NODES**20 - 2 / 21) > 1e-7


@pytest.mark.parametrize(
    ("f", "a", "b", "exact"),
    [
        (np.sin, 0.0, math.pi, 2.0),
        (np.exp, -1.0, 3.0, math.exp(3) - math.exp(-1)),
        (lambda x: 1 / (1 + x**2), -5.0, 5.0, 2 * math.atan(5)),
        (lambda x: np.sin(1000 * x), 0.0, 1.0, (1 - math.cos(1000)) / 1000),
        (np.floor, 0.0, 3.5, 4.5),  # jumps
        (np.abs, -1.0, 2.0, 2.5),  # a kink
        (
            lambda x: np.exp(-(((x - 0.3) / 1e-3) ** 2)),
            -1,
            1,
            math.sqrt(math.pi) * 1e-3,
        ),
    ],
)
def test_integrals_are_accurate(
    f: Callable[[Array], Array], a: float, b: float, exact: float
) -> None:
    result = run(f, a, b)
    assert result.values[0] == pytest.approx(exact, rel=1e-10, abs=1e-14)
    assert abs(result.values[0] - exact) <= max(result.errors[0], 1e-15)


@pytest.mark.parametrize(
    ("f", "a", "b", "exact"),
    [
        (lambda x: 1 / np.sqrt(x), 0.0, 1.0, 2.0),
        (np.log, 0.0, 1.0, -1.0),
        (lambda x: x**-0.9, 0.0, 1.0, 10.0),
        (lambda x: 1 / np.sqrt(np.abs(x)), -1.0, 4.0, 6.0),  # inside: a cut lands on it
    ],
)
def test_integrable_singularities_converge(
    f: Callable[[Array], Array], a: float, b: float, exact: float
) -> None:
    assert value(f, a, b) == pytest.approx(exact, rel=1e-9)


def test_singularities_away_from_zero_are_kept_if_accurate_enough() -> None:
    """Floating point cannot get closer to 1 than ~1e-13: the result is still good."""
    result = run(lambda x: 1 / np.sqrt(1 - x**2), -1.0, 1.0)
    assert result.values[0] == pytest.approx(math.pi, rel=1e-7)
    assert result.errors[0] > quadrature.TOLERANCE * result.magnitudes[0]  # missed
    assert result.errors[0] <= quadrature.ACCEPTABLE * result.magnitudes[0]


@pytest.mark.parametrize(
    "f",
    [lambda x: 1 / x, lambda x: 1 / x**2, np.tan, lambda x: 1 / (x - 0.3)],
)
def test_divergent_integrals_are_nan_and_give_up_quickly(
    f: Callable[[Array], Array],
) -> None:
    g, calls = counting(f)
    result = integrate(
        g, np.array([-0.5, 0.0]), np.array([0.0, 2.0]), np.zeros(2, np.intp), 1
    )
    assert math.isnan(result.values[0])
    assert len(calls) <= quadrature.STALL_ROUNDS + 4


def test_functions_undefined_on_a_stretch_give_nan_at_once() -> None:
    g, calls = counting(np.sqrt)
    result = integrate(g, np.array([-1.0]), np.array([1.0]), np.zeros(1, np.intp), 1)
    assert math.isnan(result.values[0])
    assert math.isinf(result.errors[0])
    assert len(calls) == 1


def test_a_single_undefined_point_does_not_matter() -> None:
    """sin(x)/x is undefined at 0, the middle node of [-1, 1]: it is cut away."""
    result = run(lambda x: np.sin(x) / x, -1.0, 1.0, parts=1)
    assert result.values[0] == pytest.approx(1.8921661407343662, rel=1e-14)


def test_functions_that_raise_or_are_scalar_only() -> None:
    def picky(x: float) -> float:
        if x == 0.0:
            raise ZeroDivisionError
        return math.sin(x) / x

    assert value(picky, -1.0, 1.0) == pytest.approx(1.8921661407343662, rel=1e-13)


def test_pieces_are_independent() -> None:
    """A divergent or undefined piece leaves the others as accurate as alone.

    Not bit for bit: how intervals are cut depends on how many are cut in
    all, which the other pieces change.
    """
    f = lambda x: 1 / x  # noqa: E731
    g = lambda x: evaluate_quietly(f, x)  # noqa: E731
    lo, hi = np.array([1.0, -1.0, 2.0, -3.0]), np.array([2.0, 1.0, 5.0, -2.0])
    together = integrate(g, lo, hi, np.arange(4), 4)
    assert math.isnan(together.values[1])
    for i in (0, 2, 3):
        alone = integrate(g, lo[i : i + 1], hi[i : i + 1], np.zeros(1, np.intp), 1)
        assert together.values[i] == pytest.approx(alone.values[0], rel=1e-14)
    np.testing.assert_allclose(
        together.values[[0, 2, 3]], [math.log(2), math.log(2.5), math.log(2 / 3)]
    )


def test_constant_and_zero_integrands() -> None:
    assert value(lambda x: 0 * x, -1.0, 1.0) == 0.0
    assert value(lambda x: 0 * x + 3.0, 0.0, 2.0) == pytest.approx(6.0, rel=1e-15)
    assert value(np.sin, -math.pi, math.pi) == pytest.approx(0.0, abs=1e-15)


def test_the_budget_bounds_the_work() -> None:
    """An integrand oscillating 160,000 times cannot be done: NaN, within budget."""
    g, calls = counting(lambda x: np.sin(1e6 * x))
    result = integrate(g, np.array([0.0]), np.array([1.0]), np.zeros(1, np.intp), 1)
    assert math.isnan(result.values[0])
    assert sum(calls) <= 23 * (1 + MAX_NEW_INTERVALS)  # 21 nodes and 2 probes each


def test_integration_is_deterministic() -> None:
    """The same values give the same evaluations and results, bit for bit."""
    f = lambda x: np.floor(x) + 1 / np.sqrt(np.abs(x - 0.7))  # noqa: E731
    first: list[Array] = []
    second: list[Array] = []

    def recording(into: list[Array]) -> Callable[[Array], Array]:
        def g(x: Array) -> Array:
            into.append(x.copy())
            return evaluate_quietly(f, x)

        return g

    lo, hi = np.array([-2.0, 0.0]), np.array([0.0, 3.0])
    a = integrate(recording(first), lo, hi, np.arange(2), 2)
    b = integrate(recording(second), lo, hi, np.arange(2), 2)
    assert len(first) == len(second) > 1
    for x, y in zip(first, second, strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a.values, b.values)


def test_samples_cover_each_piece_with_the_function_at_the_nodes() -> None:
    result = run(np.abs, -1.0, 2.0, samples=True)
    order = np.argsort(result.lo)
    lo, hi = result.lo[order], result.hi[order]
    assert lo[0] == -1.0
    assert hi[-1] == 2.0
    np.testing.assert_array_equal(lo[1:], hi[:-1])  # they tile the interval
    np.testing.assert_array_equal(
        result.samples, np.abs(nodes_of(result.lo, result.hi))
    )
    half = (result.hi - result.lo) / 2
    total = (result.samples @ KRONROD * half).sum()
    assert total == pytest.approx(result.values[0], rel=1e-14)
    assert run(np.abs, -1.0, 2.0).lo.size == 0  # not kept unless asked for


@settings(max_examples=60, deadline=None)
@given(
    coefficients=st.lists(
        st.floats(-10, 10, allow_subnormal=False), min_size=1, max_size=12
    ),
    a=st.floats(-50, 50),
    width=st.floats(1e-3, 100),
)
def test_polynomials_are_integrated_to_rounding(
    coefficients: list[float], a: float, width: float
) -> None:
    p = np.polynomial.Polynomial(coefficients)
    b = a + width
    # The reference in exact rational arithmetic: in floats it cancels itself.
    lo, hi = Fraction(a), Fraction(b)
    exact = sum(
        Fraction(c) * (hi ** (k + 1) - lo ** (k + 1)) / (k + 1)
        for k, c in enumerate(coefficients)
    )
    result = run(p, a, b)
    scale = result.magnitudes[0] + abs(float(exact))
    assert abs(float(Fraction(float(result.values[0])) - exact)) <= 1e-12 * scale


@settings(max_examples=40, deadline=None)
@given(
    a=st.floats(-20, 20),
    widths=st.lists(st.floats(1e-2, 10), min_size=2, max_size=5),
    k=st.floats(0.1, 30),
)
def test_integrals_add_up_over_neighbouring_pieces(
    a: float, widths: list[float], k: float
) -> None:
    f = lambda x: np.sin(k * x) + np.sign(x - 0.5)  # noqa: E731
    edges = a + np.concatenate([[0.0], np.cumsum(widths)])
    g = lambda x: evaluate_quietly(f, x)  # noqa: E731
    n = len(widths)
    pieces = integrate(g, edges[:-1], edges[1:], np.arange(n), n)
    whole = integrate(g, edges[:-1], edges[1:], np.zeros(n, np.intp), 1)
    scale = pieces.magnitudes.sum()
    assert abs(pieces.values.sum() - whole.values[0]) <= 1e-9 * scale


@settings(max_examples=60, deadline=None)
@given(
    end=st.integers(0, 8),
    offset=st.floats(-0.005, 0.005),
    kink=st.booleans(),
)
def test_jumps_and_kinks_beyond_the_outermost_nodes_are_seen(
    end: int, offset: float, kink: bool
) -> None:
    """Neither rule samples the last 0.43% of an interval: the ends are probed.

    The feature is placed right next to an end of one of the eight starting
    parts of [0, 1], where it would otherwise lie between the outermost node
    and the end, invisible to both rules. Exactly at the end it is a single
    point; between the probe and the end (4e-6 of the half width) it is
    not seen, which costs at most that much (not tested).
    """
    blind = (1 - quadrature.PROBE) / 16
    assume(offset == 0 or abs(offset) / 8 > blind)
    c = end / 8 + offset / 8
    if kink:
        f = lambda x: np.abs(x - c)  # noqa: E731
        exact = (c**2 + (1 - c) ** 2) / 2 if 0 <= c <= 1 else abs(0.5 - c)
    else:
        f = lambda x: np.where(x < c, 0.0, 1.0)  # noqa: E731
        exact = min(max(1 - c, 0.0), 1.0)
    assert value(f, 0.0, 1.0) == pytest.approx(exact, rel=1e-10, abs=1e-12)


def test_jumps_exactly_at_an_end_change_nothing() -> None:
    """The probes are inside the ends: a jump at a limit is a single point."""
    for f, a, b in [
        (np.floor, 0.0, 1.0),  # floor(1) = 1, but only at 1
        (lambda x: np.heaviside(x, 1.0), -1.0, 0.0),
        (lambda x: np.where(x < 1, 0.0, 1.0), 0.0, 1.0),
    ]:
        g, calls = counting(f)
        result = integrate(g, np.array([a]), np.array([b]), np.zeros(1, np.intp), 1)
        assert result.values[0] == 0.0
        assert len(calls) == 1  # no refinement towards the jump at all
    assert value(np.floor, 0.0, 3.0) == pytest.approx(3.0, rel=1e-12)


def test_singular_limits_are_probed_without_harm() -> None:
    g, calls = counting(lambda x: 1 / np.sqrt(x))
    result = integrate(g, np.array([0.0]), np.array([1.0]), np.zeros(1, np.intp), 1)
    assert calls[0] == 23  # the 21 nodes and a probe near each end
    assert result.values[0] == pytest.approx(2.0, rel=1e-10)
