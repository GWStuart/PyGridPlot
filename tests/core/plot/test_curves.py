"""Tests for sampling parametric curves (:mod:`pygridplot.core.plot.curves`)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from pygridplot.core.plot.curves import CurveSampling, sample_curve
from pygridplot.core.plot.geometry import Box

BOX: Box = (-2.5, -2.5, 801.5, 601.5)
"""An 800x600 plane with a small margin."""

Curve = Callable[[NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]]


def on_screen(
    fx: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    fy: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    scale: float = 50.0,
    origin: tuple[float, float] = (400.0, 300.0),
) -> Curve:
    """Map a world-space curve to screen pixels with a simple linear view."""
    ox, oy = origin
    return lambda t: (ox + scale * fx(t), oy - scale * fy(t))


def all_points(lines: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Join polylines into one array of points."""
    return np.concatenate(lines)


def segment_lengths(lines: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Return the length of every segment of every polyline."""
    return np.concatenate([np.hypot(*np.diff(line, axis=0).T) for line in lines])


def test_circle_is_exact_and_finely_sampled() -> None:
    circle = on_screen(lambda t: 3 * np.cos(t), lambda t: 3 * np.sin(t))
    (line,) = sample_curve(circle, (0.0, 2 * math.pi), BOX)
    radius = np.hypot(line[:, 0] - 400, line[:, 1] - 300)
    assert radius == pytest.approx(150.0)
    assert line[0] == pytest.approx(line[-1])  # closed
    # The chord between samples stays within the tolerance of the arc.
    half_angle = segment_lengths([line]).max() / 2 / 150.0
    assert 150.0 * (1 - math.cos(half_angle)) <= 0.25


@settings(max_examples=60, deadline=None)
@given(
    a=st.integers(1, 7),
    b=st.integers(1, 7),
    phase=st.floats(0.0, 3.0),
    size=st.floats(0.5, 5.0),
)
def test_lissajous_curves_stay_within_the_tolerance(
    a: int, b: int, phase: float, size: float
) -> None:
    """Halfway along every drawn segment, the true curve is near the chord."""
    fx = lambda t: size * np.sin(a * t + phase)  # noqa: E731
    fy = lambda t: size * np.sin(b * t)  # noqa: E731
    curve = on_screen(fx, fy)
    config = CurveSampling()
    calls = {"t": []}

    def recording(
        t: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        calls["t"].append(t.copy())
        return curve(t)

    (line,) = sample_curve(recording, (0.0, 2 * math.pi), BOX, config)
    t = np.sort(np.concatenate(calls["t"]))
    mid_t = 0.5 * (t[:-1] + t[1:])
    px, py = curve(mid_t)
    a_pts, b_pts = line[:-1], line[1:]
    mid = np.column_stack((px, py))
    d = b_pts - a_pts
    s = np.clip(
        np.einsum("ij,ij->i", mid - a_pts, d)
        / np.maximum(np.einsum("ij,ij->i", d, d), 1e-12),
        0,
        1,
    )
    error = np.hypot(*(mid - (a_pts + s[:, None] * d)).T)
    # Curves bend smoothly between samples, so a little slack covers the
    # midpoint test only looking at one point per segment.
    assert error.max() <= 2 * config.tolerance_px
    assert segment_lengths([line]).max() <= config.max_segment_px


def test_jumps_break_the_curve() -> None:
    step = on_screen(lambda t: t, np.floor)
    lines = sample_curve(step, (-3.5, 3.5), BOX)
    assert len(lines) == 8  # broken at -3, -2, ..., 3
    assert segment_lengths(lines).max() < 2.0  # no vertical connectors


def test_poles_break_the_curve_and_branches_leave_the_plane() -> None:
    curve = on_screen(np.tan, lambda t: 0.5 * t)  # x = tan t: poles at ±π/2
    lines = sample_curve(curve, (-2.0, 2.0), BOX)
    assert len(lines) == 3
    for line in lines:
        assert np.isfinite(line).all()
    left, middle, right = lines
    # Each side of a pole runs off the plane instead of stopping short:
    # tan t -> +inf just below a pole and -inf just above it.
    assert left[-1, 0] > BOX[2]
    assert middle[0, 0] < BOX[0]
    assert middle[-1, 0] > BOX[2]
    assert right[0, 0] < BOX[0]


def test_domain_edges_are_found_exactly() -> None:
    lobe = lambda t: 4 * np.sqrt(np.cos(2 * t))  # noqa: E731  lemniscate radius
    curve = on_screen(lambda t: lobe(t) * np.cos(t), lambda t: lobe(t) * np.sin(t))
    lines = sample_curve(curve, (0.0, 2 * math.pi), BOX)
    assert len(lines) == 3  # the right lobe is split by the start of the range
    ends = np.array([p for line in lines for p in (line[0], line[-1])])
    near_origin = np.hypot(ends[:, 0] - 400, ends[:, 1] - 300)
    assert np.sort(near_origin)[:4] == pytest.approx(np.zeros(4), abs=0.2)


def test_curves_far_off_screen_cost_little() -> None:
    calls = []

    def far(t: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        calls.append(t.size)
        return 5000 + 100 * np.cos(t), 5000 + 100 * np.sin(t)

    lines = sample_curve(far, (0.0, 2 * math.pi), BOX)
    assert sum(calls) == 256  # only the initial samples
    assert len(lines) == 1


def test_budget_is_respected() -> None:
    calls = []

    def wild(t: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        calls.append(t.size)
        return 400 + 50 * t, 300 - 200 * np.sin(1 / t)

    config = CurveSampling(max_samples=2000)
    sample_curve(wild, (-3.0, 3.0), BOX, config)
    assert sum(calls) <= 2000


def test_nothing_is_drawn_for_undefined_curves_or_bad_ranges() -> None:
    nowhere = lambda t: (np.full(t.shape, np.nan), t)  # noqa: E731
    assert sample_curve(nowhere, (0.0, 1.0), BOX) == []
    circle = on_screen(np.cos, np.sin)
    assert sample_curve(circle, (1.0, 1.0), BOX) == []
    assert sample_curve(circle, (0.0, math.inf), BOX) == []


@settings(max_examples=40, deadline=None)
@given(dx=st.integers(-60, 60), dy=st.integers(-60, 60))
def test_panning_moves_curves_rigidly(dx: int, dy: int) -> None:
    fx = lambda t: 4 * np.cos(3 * t) * np.cos(t)  # noqa: E731
    fy = lambda t: 4 * np.cos(3 * t) * np.sin(t)  # noqa: E731
    before = all_points(sample_curve(on_screen(fx, fy), (0.0, math.pi), BOX))
    moved = on_screen(fx, fy, origin=(400.0 + dx, 300.0 + dy))
    after = all_points(sample_curve(moved, (0.0, math.pi), BOX))
    inside = (
        (before[:, 0] > 100)
        & (before[:, 0] < 700)
        & (before[:, 1] > 100)
        & (before[:, 1] < 500)
    )
    shifted = {tuple(p) for p in np.round(before[inside] + (dx, dy), 6)}
    assert shifted <= {tuple(p) for p in np.round(after, 6)}


def test_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="initial_samples"):
        CurveSampling(initial_samples=1)
    with pytest.raises(ValueError, match="positive"):
        CurveSampling(tolerance_px=0.0)
    with pytest.raises(ValueError, match="max_rounds"):
        CurveSampling(max_rounds=-1)
