"""Tests for arrows as polygons (:mod:`pygridplot.core.plot.arrows`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from pygridplot.core.plot.arrows import HEAD_ASPECT, arrow_rings, head_shape


def signed_area(ring: NDArray[np.float64]) -> float:
    """Return a ring's signed area."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def one(
    tail: tuple[float, float], tip: tuple[float, float], **kwargs: float
) -> NDArray:
    """Return the ring of a single arrow."""
    width = kwargs.get("width", 2.0)
    head = (kwargs.get("head_length", 10.0), kwargs.get("head_width", 8.0))
    (ring,) = arrow_rings(np.array([tail]), np.array([tip]), width, head)
    return ring


def test_head_shapes() -> None:
    assert head_shape(2.0, 12.0) == (12.0, HEAD_ASPECT * 12.0)
    assert head_shape(2.0, 0.0) == (0.0, 0.0)
    thin, thick = head_shape(1.0, None), head_shape(4.0, None)
    assert thin[0] < thick[0]  # automatic heads grow with the shaft
    assert thin[1] > 2 * 1.0  # and are always wider than it
    assert thick[1] > 2 * 4.0


def test_an_arrow_runs_from_tail_to_tip() -> None:
    ring = one((10.0, 20.0), (110.0, 20.0))
    assert ring.shape == (7, 2)
    np.testing.assert_allclose(ring[3], [110.0, 20.0])  # the tip
    np.testing.assert_allclose(ring[[0, 6]], [[10.0, 21.0], [10.0, 19.0]])  # tail
    np.testing.assert_allclose(ring[[1, 5]], [[100.0, 21.0], [100.0, 19.0]])
    np.testing.assert_allclose(ring[[2, 4]], [[100.0, 24.0], [100.0, 16.0]])  # barbs


def test_the_area_is_the_shaft_and_the_head() -> None:
    ring = one((0.0, 0.0), (60.0, 80.0), width=3.0)  # 100 px long
    shaft = 3.0 * (100.0 - 10.0)
    head = 0.5 * 8.0 * 10.0
    assert abs(signed_area(ring)) == pytest.approx(shaft + head)


def test_short_arrows_are_a_smaller_head() -> None:
    ring = one((0.0, 0.0), (5.0, 0.0))  # half as long as the head
    np.testing.assert_allclose(ring[[1, 5]], [[0.0, 1.0], [0.0, -1.0]])  # no shaft
    np.testing.assert_allclose(ring[[2, 4]], [[0.0, 2.0], [0.0, -2.0]])  # half width
    assert abs(signed_area(ring)) == pytest.approx(0.5 * 4.0 * 5.0)


def test_no_head_is_a_plain_segment() -> None:
    ring = one((0.0, 0.0), (0.0, 50.0), head_length=0.0, head_width=0.0)
    assert abs(signed_area(ring)) == pytest.approx(2.0 * 50.0)
    np.testing.assert_allclose(ring[3], [0.0, 50.0])


def test_arrows_without_a_direction_are_left_out() -> None:
    tails = np.array([[0.0, 0.0], [1.0, 1.0], [math.nan, 0.0], [3.0, 3.0]])
    tips = np.array([[5.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, math.inf]])
    rings = arrow_rings(tails, tips, 2.0, (4.0, 3.0))
    assert rings.shape == (1, 7, 2)
    np.testing.assert_allclose(rings[0, 3], [5.0, 0.0])
    assert arrow_rings(tails[:0], tips[:0], 2.0, (4.0, 3.0)).shape == (0, 7, 2)


@settings(max_examples=80, deadline=None)
@given(
    x0=st.floats(-500, 500),
    y0=st.floats(-500, 500),
    angle=st.floats(0, 2 * math.pi),
    length=st.floats(0.5, 400),
    width=st.floats(0.5, 8),
)
def test_arrows_have_the_right_area_and_all_wind_the_same_way(
    x0: float, y0: float, angle: float, length: float, width: float
) -> None:
    """Every arrow is a simple shape of the expected area, wound one way.

    Winding one way is what makes overlapping arrows merge under the
    nonzero rule, rather than cancel.
    """
    head = head_shape(width, None)
    tail = (x0, y0)
    tip = (x0 + length * math.cos(angle), y0 + length * math.sin(angle))
    ring = one(tail, tip, width=width, head_length=head[0], head_width=head[1])
    shrink = min(1.0, length / head[0])
    shaft = width * max(0.0, length - head[0])
    expected = shaft + 0.5 * head[1] * head[0] * shrink**2
    # Negative: clockwise in these coordinates (y up), for every arrow alike.
    assert signed_area(ring) == pytest.approx(-expected, rel=1e-6, abs=1e-6)
    np.testing.assert_allclose(ring[3], tip)
