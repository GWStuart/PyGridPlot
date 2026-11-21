"""Tests for finding implicit curves (:mod:`pygridplot.core.plot.implicit`).

The sampler works in transformed coordinates with no view, so these tests
call it directly with functions of ``(u, w)``, which are plain ``(x, y)``
here. The region is an 800 × 600 px plane at 50 px per unit, sampled every
2 px, as a default plane would be.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from pygridplot.core.plot.implicit import (
    ImplicitContours,
    ImplicitSampling,
    candidate_cells,
    sample_implicit,
)

Function = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]

SCALE = 50.0
"""Pixels per unit."""
STEP = (2 / SCALE, 2 / SCALE)
"""The fine grid step: 2 px."""
U_RANGE = (-8.0, 8.0)
W_RANGE = (-6.0, 6.0)


def sample(
    func: Function,
    config: ImplicitSampling | None = None,
    u_range: tuple[float, float] = U_RANGE,
    w_range: tuple[float, float] = W_RANGE,
) -> ImplicitContours | None:
    """Sample a curve over the standard region."""
    return sample_implicit(func, u_range, w_range, STEP, config or ImplicitSampling())


def world_lines(contours: ImplicitContours | None) -> list[NDArray[np.float64]]:
    """Return a result's polylines in (transformed) world coordinates."""
    assert contours is not None
    (i0, j0), (su, sw) = contours.origin, contours.step
    return [
        np.column_stack(((i0 + line[:, 0]) * su, (j0 + line[:, 1]) * sw))
        for line in contours.lines
    ]


def all_points(lines: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Join polylines into one array of points."""
    return np.concatenate(lines)


def longest_segment_px(lines: list[NDArray[np.float64]]) -> float:
    """Return the longest segment of any polyline, in pixels."""
    return max(float(np.hypot(*np.diff(line, axis=0).T).max()) for line in lines)


# --- Accuracy and topology -------------------------------------------------


def test_circle_is_one_closed_accurate_line() -> None:
    (line,) = world_lines(sample(lambda x, y: x**2 + y**2 - 9))
    radius = np.hypot(line[:, 0], line[:, 1])
    assert np.abs(radius - 3).max() * SCALE < 0.01  # within 1/100 px
    assert line[0] == pytest.approx(line[-1])  # closed
    # Consecutive vertices are on neighbouring fine-grid edges.
    assert longest_segment_px([line]) * SCALE <= 2 * math.sqrt(2) + 1e-9


@pytest.mark.parametrize(
    ("name", "func"),
    [
        ("heart", lambda x, y: (x**2 + y**2 - 4) ** 3 - 4 * x**2 * y**3),
        ("folium", lambda x, y: x**3 + y**3 - 6 * x * y),
        ("lemniscate", lambda x, y: (x**2 + y**2) ** 2 - 8 * (x**2 - y**2)),
        ("cubic", lambda x, y: y**2 - x**3 + 2 * x - 1),
    ],
)
def test_classic_curves_are_accurate(name: str, func: Function) -> None:
    points = all_points(world_lines(sample(func)))
    # A vertex is within a small fraction of a pixel of the curve: |F| is at
    # most the gradient times that distance.
    eps = 1e-6
    x, y = points[:, 0], points[:, 1]
    gradient = np.hypot(
        (func(x + eps, y) - func(x - eps, y)) / (2 * eps),
        (func(x, y + eps) - func(x, y - eps)) / (2 * eps),
    )
    distance_px = np.abs(func(x, y)) / np.maximum(gradient, 1e-12) * SCALE
    # Away from singular points, where the gradient vanishes.
    regular = gradient > 0.5
    assert regular.mean() > 0.9
    assert distance_px[regular].max() < 0.1, name


def test_crossing_curves_are_drawn_through_the_crossing() -> None:
    points = all_points(world_lines(sample(lambda x, y: x * y)))
    on_x_axis = np.abs(points[:, 1]) < 1e-9
    on_y_axis = np.abs(points[:, 0]) < 1e-9
    assert (on_x_axis | on_y_axis).all()
    assert points[on_x_axis, 0].min() < -7.9
    assert points[on_x_axis, 0].max() > 7.9
    assert points[on_y_axis, 1].min() < -5.9
    assert points[on_y_axis, 1].max() > 5.9


def test_lemniscate_reaches_its_double_point() -> None:
    lines = world_lines(sample(lambda x, y: (x**2 + y**2) ** 2 - 8 * (x**2 - y**2)))
    nearest = np.hypot(*all_points(lines).T).min()
    assert nearest * SCALE < 2.0  # within a fine cell of the origin


def test_separate_pieces_are_separate_lines() -> None:
    # Two circles and a line between them.
    def func(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        return ((x - 4) ** 2 + y**2 - 1) * ((x + 4) ** 2 + y**2 - 1) * (y - 3)

    lines = world_lines(sample(func))
    assert len(lines) == 3


@settings(max_examples=40, deadline=None)
@given(
    cx=st.floats(-6.0, 6.0),
    cy=st.floats(-4.0, 4.0),
    radius_px=st.floats(1.5, 12.0),
)
def test_small_loops_are_found(cx: float, cy: float, radius_px: float) -> None:
    """Loops much smaller than a coarse cell (8 px) are found, and unbroken."""
    r = radius_px / SCALE
    lines = world_lines(sample(lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 - r**2))
    assert len(lines) == 1
    radius = np.hypot(lines[0][:, 0] - cx, lines[0][:, 1] - cy)
    assert np.abs(radius - r).max() * SCALE < 0.25


def test_loops_are_unbroken_wherever_they_lie() -> None:
    """No crossing of a loop is rejected, at any sub-pixel position.

    Edges nearly tangent to a loop hold a root near one end, which takes more
    bisection steps to confirm than a typical crossing; none may be rejected
    as if it were a jump.
    """
    r = 6 / SCALE
    for cx in np.linspace(0.0, 2 * STEP[0], 15, endpoint=False):
        for cy in np.linspace(0.0, 2 * STEP[1], 15, endpoint=False):
            result = sample(
                lambda x, y, a=cx, b=cy: (x - a) ** 2 + (y - b) ** 2 - r**2,
                u_range=(-1, 1),
                w_range=(-1, 1),
            )
            assert result is not None
            assert len(result.lines) == 1, (cx, cy)


def test_nothing_for_functions_without_zeros() -> None:
    assert sample(lambda x, y: x**2 + y**2 + 1) is None
    assert sample(lambda x, y: 0 * x + 1) is None
    assert sample(lambda x, y: x * np.nan) is None
    assert sample(lambda x, y: 1 / (x * y)) is None  # poles only
    assert sample(lambda x, y: (x - 100) ** 2 + y**2 - 1) is None  # off screen


def test_identically_zero_draws_nothing() -> None:
    assert sample(lambda x, y: 0 * x) is None


@pytest.mark.parametrize("offset", [0.0, 1e-3, 0.5 * STEP[0]])
def test_touching_zeros_draw_nothing_wherever_the_grid_lies(offset: float) -> None:
    """``F`` must change sign: a curve where it only touches zero is never drawn.

    With no offset the grid's nodes lie exactly on the curve and ``F`` is
    exactly zero there; that must not make a difference.
    """
    assert sample(lambda x, y: (x - y - offset) ** 2) is None
    assert sample(lambda x, y: (y - offset) ** 2 + 0 * x) is None
    assert sample(lambda x, y: (x - offset) ** 2 + y**2) is None  # a single point
    assert sample(lambda x, y: np.maximum(0.0, x - offset) + 0 * y) is None


@pytest.mark.parametrize("offset", [0.0, 1e-3, 0.5 * STEP[0]])
def test_crossings_through_nodes_are_drawn_unbroken(offset: float) -> None:
    (diagonal,) = world_lines(sample(lambda x, y: x - y - offset))
    assert np.abs(diagonal[:, 0] - diagonal[:, 1] - offset).max() < 1e-9
    assert np.ptp(diagonal[:, 0]) > 11.9  # corner to corner
    (column,) = world_lines(sample(lambda x, y: x - offset + 0 * y))
    assert np.ptp(column[:, 1]) > 11.9


def test_extreme_magnitudes_do_not_matter() -> None:
    def circle(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        return x**2 + y**2 - 9

    expected = world_lines(sample(circle))
    for factor in (1e-300, 1e300):
        lines = world_lines(sample(lambda x, y, f=factor: f * circle(x, y)))
        np.testing.assert_allclose(all_points(lines), all_points(expected))


# --- Several levels --------------------------------------------------------


def test_levels_give_one_curve_each() -> None:
    result = sample_implicit(
        lambda x, y: x**2 + y**2, U_RANGE, W_RANGE, STEP, levels=(1.0, 4.0, 9.0)
    )
    lines = world_lines(result)
    assert len(lines) == 3
    radii = sorted(float(np.hypot(*line.T).mean()) for line in lines)
    assert radii == pytest.approx([1.0, 2.0, 3.0], abs=1e-4)


def test_levels_match_separate_calls() -> None:
    def height(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sin(x) + np.cos(y) + 0.1 * x

    levels = (-1.0, -0.25, 0.5, 1.2)
    together = sample_implicit(height, (-3, 3), (-2, 2), STEP, levels=levels)
    separate = [
        sample_implicit(height, (-3, 3), (-2, 2), STEP, levels=(c,)) for c in levels
    ]

    def keys(lines: list[NDArray[np.float64]]) -> set[tuple[float, float]]:
        points = all_points(lines)
        return {(float(a), float(b)) for a, b in np.round(points / STEP[0], 6)}

    expected: set[tuple[float, float]] = set()
    for result in separate:
        expected |= keys(world_lines(result))
    assert keys(world_lines(together)) == expected


def test_levels_are_validated() -> None:
    circle: Function = lambda x, y: x**2 + y**2  # noqa: E731
    with pytest.raises(ValueError, match="levels"):
        sample_implicit(circle, U_RANGE, W_RANGE, STEP, levels=(1.0, math.nan))
    assert sample_implicit(circle, U_RANGE, W_RANGE, STEP, levels=()) is None
    # A level the function never reaches is simply not drawn.
    only = sample_implicit(circle, U_RANGE, W_RANGE, STEP, levels=(-1.0, 4.0))
    assert len(world_lines(only)) == 1


# --- Poles, jumps and domain edges -----------------------------------------


def test_poles_are_not_joined() -> None:
    lines = world_lines(sample(lambda x, y: y - np.tan(x)))
    # Five whole branches (centred on -2π ... 2π) and the ends of those
    # centred on ±3π, which reach in to |x| = 2.5π ≈ 7.85.
    assert len(lines) == 7
    points = all_points(lines)
    # Every vertex is on y = tan(x): x is arctan(y) plus a multiple of pi.
    offset = (points[:, 0] - np.arctan(points[:, 1])) / math.pi
    assert np.abs(offset - np.round(offset)).max() * math.pi * SCALE < 0.25
    assert longest_segment_px(lines) * SCALE <= 2 * math.sqrt(2) + 1e-9


def test_jumps_are_not_joined() -> None:
    lines = world_lines(sample(lambda x, y: y - np.floor(x)))
    points = all_points(lines)
    # Every vertex is on a step: y is a whole number and floor(x) == y.
    assert np.abs(points[:, 1] - np.round(points[:, 1])).max() < 1e-9
    assert np.all(np.floor(points[:, 0] + 1e-9) >= np.round(points[:, 1]))
    for line in lines:  # each piece is one horizontal step
        assert np.ptp(line[:, 1]) < 1e-9


def test_jumps_through_zero_are_curves() -> None:
    """``sign(x)`` jumps across 0, but is exactly 0 on the line x = 0."""
    (line,) = world_lines(sample(lambda x, y: np.sign(x) + 0 * y))
    assert np.abs(line[:, 0]).max() < 1e-12


def test_curves_stop_at_domain_edges() -> None:
    lines = world_lines(sample(lambda x, y: y - np.sqrt(x)))
    (line,) = lines
    assert line[:, 0].min() >= 0.0
    assert line[:, 0].min() * SCALE < 4.0  # within about a fine cell of x = 0
    # The distance to the curve x = y² (smooth, unlike sqrt at 0) is tiny.
    x, y = line[:, 0], line[:, 1]
    distance = np.abs(x - y**2) / np.hypot(1.0, 2 * y)
    assert distance.max() * SCALE < 0.1


def test_infinite_values_count_as_undefined() -> None:
    assert sample(lambda x, y: np.where(x > 0, np.inf, -1.0) + 0 * y) is None


def test_refine_steps_zero_keeps_false_crossings() -> None:
    """Without the check, a pole is drawn as a line: the check is what removes it."""
    unchecked = world_lines(
        sample(lambda x, y: y - np.tan(x), ImplicitSampling(refine_steps=0))
    )
    points = all_points(unchecked)
    offset = (points[:, 0] - np.arctan(points[:, 1])) / math.pi
    assert np.abs(offset - np.round(offset)).max() > 0.2  # far off the curve


# --- Panning, budget and configuration -------------------------------------


@settings(max_examples=40, deadline=None)
@given(di=st.integers(-80, 80), dj=st.integers(-80, 80))
def test_panning_moves_curves_rigidly(di: int, dj: int) -> None:
    """Shifting the curve by whole fine steps shifts every vertex exactly.

    Shifting the curve instead of the region is the same thing seen from the
    curve: the grid stays anchored at multiples of the step.
    """
    du, dw = di * STEP[0], dj * STEP[1]

    def folium(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        return x**3 + y**3 - 6 * x * y + 0.5

    before = all_points(world_lines(sample(folium)))
    after = all_points(world_lines(sample(lambda x, y: folium(x - du, y - dw))))
    inside = (np.abs(before[:, 0] + du) < 7) & (np.abs(before[:, 1] + dw) < 5)
    inside &= (np.abs(before[:, 0]) < 7) & (np.abs(before[:, 1]) < 5)

    def keys(points: NDArray[np.float64]) -> set[tuple[float, float]]:
        rounded = np.round(points / STEP[0], 5) + 0.0  # + 0.0 turns -0 into 0
        return {(float(a), float(b)) for a, b in rounded}

    assert keys(before[inside] + (du, dw)) <= keys(after)


@pytest.mark.parametrize("budget", [1000, 5000, 20000, 100_000])
def test_budget_is_respected(budget: int) -> None:
    config = ImplicitSampling(max_samples=budget)
    for func in (
        lambda x, y: x**2 + y**2 - 9,
        lambda x, y: np.sin(3 * x) * np.sin(3 * y) - 0.2,
        lambda x, y: y - np.tan(x),
    ):
        calls = 0

        def counted(
            x: NDArray[np.float64], y: NDArray[np.float64], f: Function = func
        ) -> NDArray[np.float64]:
            nonlocal calls
            calls += x.size
            return f(x, y)

        result = sample(counted, config)
        assert result is not None
        assert calls == result.evaluations <= budget


def test_small_budgets_coarsen_the_grid_but_keep_the_curve() -> None:
    result = sample(lambda x, y: x**2 + y**2 - 9, ImplicitSampling(max_samples=2000))
    assert result is not None
    assert result.step[0] > STEP[0]  # coarser than asked
    (line,) = world_lines(result)
    assert np.abs(np.hypot(*line.T) - 3).max() * SCALE < 2.0


def test_huge_regions_coarsen_the_coarse_grid() -> None:
    """A 40,000 × 30,000 px region still fits the budget."""
    result = sample(
        lambda x, y: x**2 + y**2 - 90_000, u_range=(-400, 400), w_range=(-300, 300)
    )
    assert result is not None
    assert result.evaluations <= ImplicitSampling().max_samples
    assert len(result.lines) == 1


def test_unusable_regions_give_nothing() -> None:
    circle: Function = lambda x, y: x**2 + y**2 - 9  # noqa: E731
    assert sample(circle, u_range=(-math.inf, 1.0)) is None
    assert sample(circle, w_range=(math.nan, 1.0)) is None
    assert sample_implicit(circle, U_RANGE, W_RANGE, (0.0, 0.04)) is None
    assert sample(circle, u_range=(-1e12, 1e12)) is None  # too many steps


def test_the_coarse_search_misses_nothing() -> None:
    def busy(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sin(3 * x) * np.sin(3 * y) - 0.2

    results = [
        sample(busy, ImplicitSampling(coarse_factor=factor)) for factor in (8, 2, 1)
    ]
    # The fine grid is the same 2 px grid each time: only the search for the
    # curve differs (factor 1 evaluates it everywhere), and it misses nothing.
    keys = []
    for result in results:
        assert result is not None
        assert result.step == pytest.approx(STEP)
        points = all_points(world_lines(result))
        inside = (np.abs(points[:, 0]) < 7.5) & (np.abs(points[:, 1]) < 5.5)
        keys.append({tuple(p) for p in np.round(points[inside] / STEP[0], 6)})
    assert keys[0] == keys[1] == keys[2]


def test_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="resolution_px"):
        ImplicitSampling(resolution_px=0.1)
    with pytest.raises(ValueError, match="coarse_factor"):
        ImplicitSampling(coarse_factor=3)
    with pytest.raises(ValueError, match="refine_steps"):
        ImplicitSampling(refine_steps=-1)
    with pytest.raises(ValueError, match="max_samples"):
        ImplicitSampling(max_samples=10)


# --- Candidate cells -------------------------------------------------------


def test_candidate_cells() -> None:
    values = np.array(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, -1.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        ]
    )
    cells = candidate_cells(values)
    assert cells.shape == (3, 6)
    # The sign change at the -1 node, widened by one cell.
    assert cells[:, 3:].all()
    assert not cells[:, :2].any()


def test_domain_edges_are_candidates() -> None:
    values = np.full((4, 8), 5.0)
    values[:, :2] = np.nan
    cells = candidate_cells(values)
    assert cells[:, :3].all()  # the cells with undefined corners, and widened
    assert not cells[:, 4:].any()


def test_near_misses_are_candidates() -> None:
    """A dip towards zero between nodes, with no sign change, is a candidate."""
    x = np.arange(12.0)
    values = np.add.outer(np.zeros(5), np.abs(x - 5.5) + 0.1)  # minimum 0.6
    cells = candidate_cells(values)
    assert cells[:, 4:7].all()  # the dip, and widened
    assert not cells[:, :2].any()
    assert not cells[:, 9:].any()
