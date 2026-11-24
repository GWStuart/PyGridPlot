"""Tests for dash patterns: their settings, layout, anchoring and clipping."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from pygridplot.core.plot.dashes import (
    ANCHOR_PERIODS,
    MAX_MARGIN_PX,
    DashLattice,
    DashPattern,
    check_dash,
    dash_lattice,
    dash_lengths,
    dash_margin,
    dash_paths,
    dash_pattern,
    swatch_dashes,
)
from pygridplot.core.view import View

BOX = (-10.0, -10.0, 610.0, 410.0)
RECT = (0, 0, 600, 400)
LATTICE = DashLattice(x0=100.0, dx=80.0, y0=50.0, dy=80.0)
DASHED = dash_pattern("dashed", 2.5)
DOTTED = dash_pattern("dotted", 2.5)
assert DASHED is not None
assert DOTTED is not None


def dashes_of(segments: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    """Group consecutive segments into dashes (a dash is several along a bend)."""
    breaks = np.linalg.norm(segments[1:, 0] - segments[:-1, 1], axis=1) > 1e-6
    starts = np.flatnonzero(np.r_[True, breaks])
    return np.split(segments, starts[1:])


def lengths(groups: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """The centre-line length of each dash."""
    return np.array([np.linalg.norm(g[:, 1] - g[:, 0], axis=1).sum() for g in groups])


def straight(x0: float, x1: float, y: float = 200.0) -> NDArray[np.float64]:
    """A horizontal path, with a vertex every pixel (as a sampled curve has)."""
    xs = np.unique(np.r_[np.arange(x0, x1), x1])
    return np.column_stack((xs, np.full(xs.shape, y)))


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
def test_dash_settings_are_names_or_pairs_of_lengths() -> None:
    assert check_dash("dashed") == "dashed"
    assert check_dash([10, 5]) == (10.0, 5.0)  # a hashable tuple
    assert check_dash(np.array([8, 3, 0, 3])) == (8.0, 3.0, 0.0, 3.0)
    for bad in ("dashes", (5,), (5, 3, 2), (5, 0), (-1, 4), (math.inf, 4), 7, None):
        with pytest.raises(ValueError, match="dash"):
            check_dash(bad)


def test_named_patterns_grow_with_the_width() -> None:
    assert dash_lengths("solid", 2.5) is None
    assert dash_pattern("solid", 2.5) is None
    thin = dash_lengths("dashed", 1.0)
    thick = dash_lengths("dashed", 6.0)
    assert thin is not None
    assert thick is not None
    assert thick[0] > 2 * thin[0]
    assert thick[1] > 2 * thin[1]
    # Hairlines are drawn a pixel wide, so their patterns are a pixel's.
    assert dash_lengths("dotted", 0.3) == dash_lengths("dotted", 1.0)
    assert dash_lengths((12, 6), 4.0) == (12.0, 6.0)  # lengths given are kept


def test_patterns_are_measured_along_the_centre_line() -> None:
    # A 10 px dash of a 2 px line: an 8 px centre line with round ends.
    pattern = dash_pattern((10, 4), 2.0)
    assert pattern == DashPattern(14.0, (0.0,), (4.0,))
    # Dashes shorter than the width are dots, as wide as the line.
    dot_dash = dash_pattern((0, 4, 12, 4), 3.0)
    assert dot_dash is not None
    assert dot_dash.period == 20.0
    assert dot_dash.centres == (0.0, 10.0)  # dot, 4 px gap, 12 px dash
    assert dot_dash.halves == (0.0, 4.5)


def test_samplers_look_further_for_longer_patterns_up_to_a_limit() -> None:
    assert dash_margin(DASHED) == pytest.approx(2 * ANCHOR_PERIODS * DASHED.period)
    huge = dash_pattern((400, 400), 2.5)
    assert huge is not None
    assert dash_margin(huge) == MAX_MARGIN_PX


def test_anchor_lines_are_powers_of_two_four_to_eight_periods_apart() -> None:
    for scale in (3.0, 50.0, 77.7, 4000.0):
        view = View(RECT, (1234.5, -3.25), (scale, scale / 3), equal_aspect=False)
        lattice = dash_lattice(view, DASHED)
        for spacing, axis_scale in ((lattice.dx, scale), (lattice.dy, scale / 3)):
            periods = spacing / DASHED.period
            assert ANCHOR_PERIODS <= periods < 2 * ANCHOR_PERIODS
            step = spacing / axis_scale  # in world units
            assert math.log2(step) == round(math.log2(step))


def test_anchor_lines_move_with_the_view() -> None:
    view = View(RECT, (0.37, -1.2), (55.0, 55.0))
    before = dash_lattice(view, DASHED)
    view.pan_pixels(23, -11)
    after = dash_lattice(view, DASHED)
    # Some vertical line is 23 px further right, some horizontal 11 px higher.
    shift_x = (after.x0 - before.x0 - 23) / before.dx
    shift_y = (after.y0 - before.y0 + 11) / before.dy
    assert shift_x == pytest.approx(round(shift_x), abs=1e-9)
    assert shift_y == pytest.approx(round(shift_y), abs=1e-9)


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
def test_a_straight_line_gets_regular_dashes_and_gaps() -> None:
    segments = dash_paths([straight(0, 600)], DASHED, LATTICE, BOX)
    assert segments is not None
    assert segments.shape[1:] == (2, 2)
    groups = dashes_of(segments)
    starts = np.array([g[0, 0, 0] for g in groups])
    ends = np.array([g[-1, 1, 0] for g in groups])
    # Between the first vertical line (x = 20) and the last (x = 580), the
    # stretches 80 px long hold 5 periods of 16.5 px, stretched by 80 / 82.5.
    fitted = (starts > 21) & (ends < 579)
    stretch = 80 / 82.5
    assert (ends - starts)[fitted] == pytest.approx(7.0 * stretch, rel=1e-9)
    gaps = starts[1:] - ends[:-1]
    assert gaps[fitted[1:] & fitted[:-1]] == pytest.approx(9.5 * stretch, rel=1e-9)
    # Before the first line, the pattern runs at its own length.
    assert (ends - starts)[0] == pytest.approx(7.0)


def test_every_anchor_is_in_the_middle_of_a_dash() -> None:
    segments = dash_paths([straight(0, 600)], DASHED, LATTICE, BOX)
    assert segments is not None
    for line in (180.0, 260.0, 340.0, 420.0):  # vertical anchor lines
        covering = [g for g in dashes_of(segments) if g[0, 0, 0] < line < g[-1, 1, 0]]
        assert len(covering) == 1
        middle = (covering[0][0, 0, 0] + covering[0][-1, 1, 0]) / 2
        assert middle == pytest.approx(line, abs=1e-9)


def test_dots_are_segments_of_no_length_evenly_spaced() -> None:
    segments = dash_paths([straight(0, 600)], DOTTED, LATTICE, BOX)
    assert segments is not None
    assert np.array_equal(segments[:, 0], segments[:, 1])
    xs = np.sort(segments[:, 0, 0])
    assert np.all(np.diff(xs) > 0)  # no dot drawn twice
    inner = np.diff(xs[(xs > 100) & (xs < 500)])
    assert inner == pytest.approx(DOTTED.period, rel=0.1)


def test_only_dashes_inside_the_box_are_returned() -> None:
    far = np.array([[-1e12, 200.0], [300.0, 200.0], [300.0, 1e12]])
    segments = dash_paths([far], DASHED, LATTICE, BOX)
    assert segments is not None
    assert len(segments) < 60  # not millions of dashes off screen
    x_min, y_min, x_max, y_max = BOX
    assert np.all((segments[..., 0] >= x_min) & (segments[..., 0] <= x_max))
    assert np.all((segments[..., 1] >= y_min) & (segments[..., 1] <= y_max))


def test_nothing_visible_gives_none_and_single_points_are_dots() -> None:
    assert dash_paths([], DASHED, LATTICE, BOX) is None
    assert dash_paths([straight(1000, 1200)], DASHED, LATTICE, BOX) is None
    dot = dash_paths([np.array([[5.0, 6.0]])], DASHED, LATTICE, BOX)
    assert dot is not None
    assert np.array_equal(dot, [[[5.0, 6.0], [5.0, 6.0]]])
    assert dash_paths([np.array([[5.0, 1e4]])], DASHED, LATTICE, BOX) is None


def test_lines_with_fixed_ends_start_and_end_with_a_dash() -> None:
    path = straight(13.0, 517.0)
    segments = dash_paths([path], DASHED, LATTICE, BOX, ends=[(True, True)])
    assert segments is not None
    groups = dashes_of(segments)
    assert groups[0][0, 0, 0] == pytest.approx(13.0)
    assert groups[-1][-1, 1, 0] == pytest.approx(517.0)
    dots = dash_paths([path], DOTTED, LATTICE, BOX, ends=[(True, True)])
    assert dots is not None
    assert dots[:, 0, 0].min() == pytest.approx(13.0)
    assert dots[:, 0, 0].max() == pytest.approx(517.0)  # the last dot is drawn


def test_corners_of_long_sides_are_in_a_dash() -> None:
    triangle = np.array([[50.0, 50.0], [450.0, 90.0], [230.0, 380.0], [50.0, 50.0]])
    segments = dash_paths([triangle], DASHED, LATTICE, BOX, corners=True)
    assert segments is not None
    for corner in triangle[:3]:
        touching = np.linalg.norm(segments[:, 0] - corner, axis=1) < 1e-6
        touching |= np.linalg.norm(segments[:, 1] - corner, axis=1) < 1e-6
        assert touching.sum() == 2  # the dash goes on round the corner


def test_dense_steps_are_dashed_like_a_curve() -> None:
    """Corners count only between sides a period long, so steps 1 px wide are dashed."""
    xs = np.arange(0, 400, 1.0)
    steps = np.repeat(np.column_stack((xs, 200 + 30 * np.sin(xs / 40))), 2, axis=0)
    steps[1::2, 0] += 1.0  # a horizontal and a vertical side per pixel
    segments = dash_paths([steps], DASHED, LATTICE, BOX, corners=True)
    assert segments is not None
    assert 15 < len(dashes_of(segments)) < 60  # dashes, not a solid line


def test_a_closed_loop_without_anchors_has_whole_periods() -> None:
    t = np.linspace(0, 2 * np.pi, 200)
    # About 8.0 periods round, outside the window where a period grows in.
    loop = np.column_stack((300 + 21 * np.cos(t), 200 + 21 * np.sin(t)))
    loop[-1] = loop[0]
    segments = dash_paths([loop], DASHED, None, BOX)
    assert segments is not None
    dash = lengths(dashes_of(segments))
    # The first dash is split where the loop starts and ends.
    pieces = np.r_[dash[0] + dash[-1], dash[1:-1]]
    assert pieces == pytest.approx(pieces.mean(), rel=1e-6)


def test_a_stretch_too_short_for_a_gap_is_drawn_whole() -> None:
    lattice = DashLattice(x0=100.0, dx=80.0, y0=50.0, dy=80.0)
    # Fixed ends 3 px either side of the line at x = 180: no room for a gap.
    path = straight(177.0, 183.0)
    segments = dash_paths([path], DASHED, lattice, BOX, ends=[(True, True)])
    assert segments is not None
    assert lengths(dashes_of(segments)) == pytest.approx([6.0])


def covered(segments: NDArray[np.float64], resolution: float) -> NDArray[np.bool_]:
    """Which cells of a horizontal line from x = 0 the dashes cover."""
    cells = np.zeros(int(200 / resolution), dtype=bool)
    for a, b in segments[:, :, 0]:
        cells[round(a / resolution) : round(b / resolution) + 1] = True
    return cells


def test_growing_stretches_change_their_dashes_smoothly() -> None:
    """Between two anchors, a stretch growing a little moves its dashes a little.

    Laying the pattern out afresh for each length would move dashes by up to
    half a period whenever the number of periods changes; instead a period
    grows in, so the change is in proportion to the growth (several times
    it while a period grows in, as the dashes after it make room).
    """
    step, resolution = 0.02, 0.01
    previous = None
    worst = 0.0
    for length in np.arange(60.0, 140.0, step):
        segments = dash_paths(
            [straight(0.0, length)], DASHED, None, BOX, ends=[(True, True)]
        )
        assert segments is not None
        cells = covered(segments, resolution)
        if previous is not None:
            worst = max(worst, float(np.sum(cells != previous)) * resolution)
        previous = cells
    assert worst < 60 * step  # a jump would change several pixels at once


def test_swatches_start_and_end_with_whole_dashes() -> None:
    for pattern in (DASHED, DOTTED, dash_pattern((4, 2, 0, 2), 1.5)):
        assert pattern is not None
        segments = swatch_dashes((0.0, 5.0), (22.0, 5.0), pattern)
        assert segments[:, :, 0].min() == pytest.approx(0.0)
        assert segments[:, :, 0].max() == pytest.approx(22.0)
        assert len(dashes_of(segments)) >= 2
        assert np.all(segments[:, :, 1] == 5.0)


# ----------------------------------------------------------------------
# Rigid panning
# ----------------------------------------------------------------------
def shifted(lattice: DashLattice, dx: int, dy: int, step: int) -> DashLattice:
    """The lattice after a pan, measured from a different line."""
    return DashLattice(
        lattice.x0 + dx + step * lattice.dx, lattice.dx, lattice.y0 + dy, lattice.dy
    )


def canonical(segments: NDArray[np.float64] | None) -> NDArray[np.float64]:
    """The dashes in an order and form that rounding cannot change.

    Slivers of dash a rounding error long (where a dash ends on the boundary
    between two stretches) lie inside the round end of the dash beside them,
    so they are left out; dots are compared as points.
    """
    if segments is None:
        return np.zeros((0, 4))
    flat = segments.reshape(-1, 4)
    size = np.hypot(flat[:, 2] - flat[:, 0], flat[:, 3] - flat[:, 1])
    if size.max() > 1e-6:
        flat = flat[size > 1e-6]
    flat = np.round(flat, 6)
    return flat[np.lexsort(flat.T[::-1])]


@settings(max_examples=80, deadline=None)
@given(
    points=st.lists(
        st.tuples(
            st.floats(-200, 800).map(lambda v: round(v, 3)),
            st.floats(-200, 600).map(lambda v: round(v, 3)),
        ),
        min_size=2,
        max_size=12,
    ),
    dx=st.integers(-150, 150),
    dy=st.integers(-150, 150),
    step=st.integers(-3, 3),
    closed=st.booleans(),
    dotted=st.booleans(),
)
def test_panning_moves_every_dash_with_the_path(
    points: list[tuple[float, float]],
    dx: int,
    dy: int,
    step: int,
    closed: bool,
    dotted: bool,
) -> None:
    pattern = DOTTED if dotted else DASHED
    path = np.array(points, dtype=np.float64)
    if closed:
        path = np.vstack((path, path[:1]))
    big = (-1e4, -1e4, 1e4, 1e4)  # all of the path, wherever it is panned to
    options = {"ends": [(True, True)], "corners": True}
    before = dash_paths([path], pattern, LATTICE, big, **options)  # type: ignore[arg-type]
    after = dash_paths(
        [path + np.array([dx, dy])],
        pattern,
        shifted(LATTICE, dx, dy, step),
        big,
        **options,  # type: ignore[arg-type]
    )
    moved = canonical(after) - np.array([dx, dy, dx, dy])
    np.testing.assert_allclose(moved, canonical(before), atol=1e-5)


def test_panning_keeps_crossings_at_vertices_on_their_lines() -> None:
    """Vertices exactly on anchor lines (as a contour's are) keep their anchors."""
    # Every vertex is on a vertical or a horizontal line of the lattice.
    path = np.array([[100.0, 50.0], [180.0, 75.0], [260.0, 130.0], [260.0, 290.0]])
    reference = dash_paths([path], DASHED, LATTICE, BOX)
    assert reference is not None
    for dx, dy in [(13, 9), (-7, 3), (101, -57)]:
        after = dash_paths(
            [path + np.array([dx + 0.1, dy + 0.3]) - np.array([0.1, 0.3])],
            DASHED,
            shifted(LATTICE, dx, dy, 0),
            (BOX[0] + dx, BOX[1] + dy, BOX[2] + dx, BOX[3] + dy),
        )
        assert after is not None
        np.testing.assert_allclose(after - [dx, dy], reference, atol=1e-6)
