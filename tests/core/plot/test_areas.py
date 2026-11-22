"""Tests for the area between two graphs (:mod:`pygridplot.core.plot.areas`)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from pygridplot.core.plot.areas import rings_between, section


def piece(x0: float, x1: float, f: object = None, n: int = 9) -> NDArray[np.float64]:
    """Return a piece of a graph over ``[x0, x1]`` (``y = f(x)``, default 1)."""
    xs = np.linspace(x0, x1, n)
    ys = f(xs) if callable(f) else np.ones_like(xs)
    return np.column_stack((xs, ys)).astype(np.float64)


def test_section_interpolates_its_ends() -> None:
    line = piece(0.0, 4.0, lambda x: 2 * x, n=5)  # x = 0, 1, 2, 3, 4
    cut = section(line, 0.5, 2.5)
    assert cut.tolist() == [[0.5, 1.0], [1.0, 2.0], [2.0, 4.0], [2.5, 5.0]]
    # Bounds on sample positions are not repeated.
    assert section(line, 1.0, 3.0).tolist() == [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]


def test_one_ring_where_both_graphs_are_defined() -> None:
    upper = [piece(-2.0, 3.0, lambda x: x + 5)]
    lower = [piece(-1.0, 4.0, lambda x: 0 * x)]
    (ring,) = rings_between(upper, lower)
    assert ring[:, 0].min() == -1.0
    assert ring[:, 0].max() == 3.0
    # The first graph forwards, then the second backwards.
    assert ring[0].tolist() == [-1.0, 4.0]
    assert ring[-1].tolist() == [-1.0, 0.0]
    assert np.all(np.diff(ring[: len(ring) // 2, 0]) > 0)


def test_rings_stop_at_every_break() -> None:
    """A jump or gap in either graph splits the area."""
    broken = [piece(-3.0, -1.0), piece(-1.0, 2.0), piece(2.5, 4.0)]
    whole = [piece(-5.0, 5.0, lambda x: 0 * x)]
    rings = rings_between(broken, whole)
    spans = [(r[:, 0].min(), r[:, 0].max()) for r in rings]
    assert spans == [(-3.0, -1.0), (-1.0, 2.0), (2.5, 4.0)]
    assert rings_between(whole, broken)[1][:, 0].max() == 2.0  # either order


def test_no_rings_without_overlap() -> None:
    assert rings_between([piece(0.0, 1.0)], [piece(2.0, 3.0)]) == []
    assert rings_between([piece(0.0, 1.0)], [piece(1.0, 3.0)]) == []  # touching
    assert rings_between([], [piece(0.0, 1.0)]) == []


def covered(intervals: list[tuple[float, float]], xs: NDArray) -> NDArray:
    """Whether each x lies strictly inside one of the intervals."""
    inside = np.zeros(xs.shape, dtype=bool)
    for low, high in intervals:
        inside |= (xs > low) & (xs < high)
    return inside


def pieces_from(cuts: list[float]) -> list[NDArray[np.float64]]:
    """Turn sorted cut points into pieces over every other interval."""
    ends = sorted(set(cuts))
    return [piece(a, b) for a, b in zip(ends[::2], ends[1::2], strict=False)]


@settings(max_examples=60, deadline=None)
@given(
    first=st.lists(st.floats(-10, 10), min_size=2, max_size=12),
    second=st.lists(st.floats(-10, 10), min_size=2, max_size=12),
)
def test_the_area_covers_exactly_where_both_are_defined(
    first: list[float], second: list[float]
) -> None:
    a, b = pieces_from(first), pieces_from(second)
    rings = rings_between(a, b)
    xs = np.linspace(-10, 10, 2001)
    expected = covered([(p[0, 0], p[-1, 0]) for p in a], xs) & covered(
        [(p[0, 0], p[-1, 0]) for p in b], xs
    )
    actual = covered([(r[:, 0].min(), r[:, 0].max()) for r in rings], xs)
    assert np.array_equal(actual, expected)
    starts = [r[0, 0] for r in rings]
    assert starts == sorted(starts)  # in order of x


def test_crossing_graphs_make_one_ring_with_both_lobes() -> None:
    rising = [piece(-2.0, 2.0, lambda x: x)]
    falling = [piece(-2.0, 2.0, lambda x: -x)]
    (ring,) = rings_between(rising, falling)
    x, y = ring[:, 0], ring[:, 1]
    lobes = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert lobes == pytest.approx(0.0)  # two lobes of 4, wound opposite ways
    assert np.abs(y).max() == 2.0
