"""Tests for following a curve with the cursor (:mod:`core.handles.tracking`)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from pygridplot.core.handles.tracking import nearest_on_curve
from pygridplot.core.plot.curves import CurvePiece, sample_curve_pieces

BOX = (-100.0, -100.0, 900.0, 700.0)
Points = tuple[NDArray[np.float64], NDArray[np.float64]]
Curve = Callable[[NDArray[np.float64]], Points]


def pieces(curve: Curve, t_range: tuple[float, float]) -> list[CurvePiece]:
    """Sample a curve given directly in screen coordinates."""
    return sample_curve_pieces(curve, t_range, BOX)


def circle(t: NDArray[np.float64]) -> Points:
    """A circle of radius 100 px about (400, 300), anticlockwise on screen."""
    return 400 + 100 * np.cos(t), 300 - 100 * np.sin(t)


def follow(
    sampled: list[CurvePiece],
    targets: list[tuple[float, float]],
    t: float,
    *,
    closed: bool = False,
) -> list[float]:
    """Drag through a sequence of targets, returning the parameter after each."""
    path = []
    for target in targets:
        found = nearest_on_curve(sampled, target, t, closed=closed, switch_px=10.0)
        assert found is not None
        t = found
        path.append(t)
    return path


def test_the_nearest_point_is_found_when_placing() -> None:
    sampled = pieces(circle, (0.0, 2 * math.pi))
    t = nearest_on_curve(sampled, (400.0, 100.0), None, closed=True, switch_px=10)
    assert t is not None
    assert t == pytest.approx(math.pi / 2, abs=1e-3)


def test_dragging_round_a_closed_curve_wraps_past_its_end() -> None:
    sampled = pieces(circle, (0.0, 2 * math.pi))
    angles = [k * 2 * math.pi / 40 for k in range(1, 81)]  # twice round
    targets = [(400 + 150 * math.cos(a), 300 - 150 * math.sin(a)) for a in angles]
    path = follow(sampled, targets, 0.0, closed=True)
    expected = [a % (2 * math.pi) for a in angles]
    gaps = [abs(p - e) for p, e in zip(path, expected, strict=True)]
    error_px = [100 * min(gap, 2 * math.pi - gap) for gap in gaps]
    # Exact up to the chords the curve was sampled into (see
    # test_constraints for the refined, exact result).
    assert max(error_px) < 0.5


def test_an_open_curve_stops_at_its_end() -> None:
    sampled = pieces(circle, (0.0, math.pi))  # the upper half
    path = follow(sampled, [(420.0, 330.0), (480.0, 340.0)], 0.3)
    assert path[-1] == 0.0  # clamped to the end, not jumping round


def test_a_figure_eight_is_followed_through_its_crossing() -> None:
    def eight(
        t: NDArray[np.float64],
    ) -> Points:
        return 400 + 200 * np.cos(t), 300 - 200 * np.sin(t) * np.cos(t)

    sampled = pieces(eight, (0.0, 2 * math.pi))
    ts = np.linspace(1.2, 1.95, 60)  # crosses the double point at t = pi / 2
    targets = [(float(a), float(b)) for a, b in zip(*eight(ts), strict=True)]
    path = follow(sampled, targets, 1.2, closed=True)
    assert np.all(np.diff(path) > 0)  # never jumps to the other branch
    assert path[-1] == pytest.approx(1.95, abs=1e-3)


def test_another_branch_is_taken_once_it_is_clearly_closer() -> None:
    # Two horizontal lines 40 px apart, one curve: y = 280 for t < 0, then 320.
    def lines(
        t: NDArray[np.float64],
    ) -> Points:
        return 400 + 100 * np.abs(t), np.where(t < 0, 280.0, 320.0)

    sampled = pieces(lines, (-2.0, 2.0))
    assert len(sampled) == 2  # broken at the jump
    near_own = follow(sampled, [(450.0, 295.0)], -0.5)  # 15 px from own, 25 from other
    assert near_own[0] < 0
    jumped = follow(sampled, [(450.0, 315.0)], -0.5)  # 35 px from own, 5 from other
    assert jumped[0] > 0


def test_a_spiral_is_followed_turn_by_turn() -> None:
    def spiral(
        t: NDArray[np.float64],
    ) -> Points:
        r = 10 * t
        return 400 + r * np.cos(t), 300 - r * np.sin(t)

    sampled = pieces(spiral, (0.0, 6 * math.pi))
    ts = np.linspace(2.0, 17.0, 300)
    # Aim a little outside the spiral, nearer the next turn than the gap allows.
    targets = [
        (400 + (10 * t + 20) * math.cos(t), 300 - (10 * t + 20) * math.sin(t))
        for t in ts
    ]
    path = follow(sampled, targets, 2.0)
    assert np.all(np.diff(path) > 0)
    assert abs(path[-1] - 17.0) < 0.2


def test_gaps_and_empty_curves() -> None:
    assert nearest_on_curve([], (0.0, 0.0), 0.0, closed=False, switch_px=10) is None

    def gappy(
        t: NDArray[np.float64],
    ) -> Points:
        y = np.where(np.abs(t) < 0.5, np.nan, 300.0)
        return 400 + 100 * t, y

    sampled = pieces(gappy, (-2.0, 2.0))
    # A parameter in the gap: the nearest point is taken instead.
    t = nearest_on_curve(sampled, (500.0, 300.0), 0.0, closed=False, switch_px=10)
    assert t == pytest.approx(1.0, abs=1e-6)


@settings(max_examples=50, deadline=None)
@given(
    start=st.floats(0.0, 2 * math.pi - 1e-6),
    angle=st.floats(0.0, 2 * math.pi),
    distance=st.floats(20.0, 300.0),
)
def test_the_point_found_on_a_circle_faces_the_target(
    start: float, angle: float, distance: float
) -> None:
    """On a circle there is no other branch: the result is always the nearest.

    Targets near the centre are left out: there, every point of the circle is
    almost equally near.
    """
    sampled = pieces(circle, (0.0, 2 * math.pi))
    target = (400 + distance * math.cos(angle), 300 - distance * math.sin(angle))
    t = nearest_on_curve(sampled, target, start, closed=True, switch_px=10)
    assert t is not None
    gap = abs(t - angle) % (2 * math.pi)
    # Within the chords' accuracy, magnified towards the centre.
    assert 100 * min(gap, 2 * math.pi - gap) < 0.5 * max(1.0, 100 / distance)
