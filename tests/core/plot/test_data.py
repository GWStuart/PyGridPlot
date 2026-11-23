"""Tests for data plots: per-point styles, error bars, steps and bars.

Covers :mod:`pygridplot.core.plot.data` and the builders that draw with it
(:func:`~pygridplot.core.plot.builders.point_markers`,
:func:`~pygridplot.core.plot.builders.error_bar_stroke`,
:func:`~pygridplot.core.plot.builders.polyline_stroke` with steps and
:func:`~pygridplot.core.plot.builders.bar_shapes`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pygridplot.core.coords import ScreenRect
from pygridplot.core.plot.builders import (
    bar_shapes,
    error_bar_stroke,
    point_markers,
    polyline_stroke,
)
from pygridplot.core.plot.data import (
    AUTO_BAR_FRACTION,
    as_bars,
    as_baselines,
    as_errors,
    auto_bar_width,
    check_step,
    check_width,
    data_range,
    is_one_color,
    per_point_colors,
    per_point_radii,
    step_vertices,
)
from pygridplot.core.plot.style import BarStyle, LineStyle, PointStyle
from pygridplot.core.view import View

RECT = ScreenRect(0, 0, 400, 300)
RED = (200, 0, 0)


def view(**kwargs: object) -> View:
    """A 400 x 300 view centred on the origin, 50 px per unit by default."""
    center = kwargs.pop("center", (0.0, 0.0))
    scale = kwargs.pop("scale", 50.0)
    return View(RECT, center, scale, **kwargs)  # type: ignore[arg-type]


# --- Colours and sizes per point -------------------------------------------------


def test_one_colour_or_a_colour_per_point() -> None:
    assert is_one_color("red")
    assert is_one_color((255, 0, 0))
    assert is_one_color([10, 20, 30, 40])
    assert is_one_color(np.array([1, 2, 3]))
    assert not is_one_color(["red", "green", "blue"])  # three colours
    assert not is_one_color([(255, 0, 0)])
    assert not is_one_color(3)
    assert per_point_colors(None, 3) is None
    assert per_point_colors((255, 0, 0), 3) is None  # one colour, three points


def test_numeric_colours_become_rgba_bytes() -> None:
    colors = per_point_colors([(255, 0, 0), (0, 0, 255, 128), (0.4, 9.6, 0)], 3)
    assert isinstance(colors, np.ndarray)
    assert colors.dtype == np.uint8
    assert colors.tolist() == [[255, 0, 0, 255], [0, 0, 255, 128], [0, 10, 0, 255]]
    array = per_point_colors(np.array([[1, 2, 3], [4, 5, 6]]), 2)
    assert isinstance(array, np.ndarray)
    assert array.tolist() == [[1, 2, 3, 255], [4, 5, 6, 255]]
    empty = per_point_colors([], 0)
    assert isinstance(empty, np.ndarray)
    assert empty.shape == (0, 4)


def test_named_colours_are_kept_as_given() -> None:
    assert per_point_colors(["red", (0, 0, 255)], 2) == ("red", (0, 0, 255))


@pytest.mark.parametrize(
    ("color", "count", "error"),
    [
        (["red", "green"], 3, ValueError),  # two colours for three points
        ([(0, 0, 300), (0, 0, 0)], 2, ValueError),  # out of range
        ([(0, 0, math.nan), (0, 0, 0)], 2, ValueError),
        ([object(), "red"], 2, TypeError),
        (3.5, 1, TypeError),
        (b"red", 3, TypeError),
    ],
)
def test_bad_colours_are_rejected(color: object, count: int, error: type) -> None:
    with pytest.raises(error):
        per_point_colors(color, count)


def test_sizes_per_point() -> None:
    assert per_point_radii(None, 3) is None
    assert per_point_radii(4.5, 3) is None
    sizes = per_point_radii([1, math.nan, 3], 3)
    assert sizes is not None
    assert sizes.tolist() == [1.0, 0.0, 3.0]  # NaN: no marker
    with pytest.raises(ValueError, match="2 sizes for 3"):
        per_point_radii([1, 2], 3)
    with pytest.raises(ValueError, match="non-negative"):
        per_point_radii([1, -2], 2)
    with pytest.raises(ValueError, match="non-negative"):
        per_point_radii([1, math.inf], 2)


# --- Error bars --------------------------------------------------------------------


def test_errors_take_three_shapes() -> None:
    one = as_errors(0.5, 3, "y_error")
    assert one is not None
    assert one.tolist() == [[0.5] * 3, [0.5] * 3]
    each = as_errors([1, 2, 3], 3, "y_error")
    assert each is not None
    assert each.tolist() == [[1, 2, 3], [1, 2, 3]]
    sides = as_errors([[1, 2, 3], [4, 5, 6]], 3, "y_error")
    assert sides is not None
    assert sides.tolist() == [[1, 2, 3], [4, 5, 6]]
    # Two numbers for two points are one each, the same below and above.
    pair = as_errors([1, 2], 2, "y_error")
    assert pair is not None
    assert pair.tolist() == [[1, 2], [1, 2]]
    assert as_errors(None, 3, "y_error") is None


def test_bad_errors_are_rejected() -> None:
    with pytest.raises(ValueError, match="x_error"):
        as_errors([1, 2], 3, "x_error")
    with pytest.raises(ValueError, match="non-negative"):
        as_errors([1, -1], 2, "y_error")
    with pytest.raises(ValueError, match="numbers"):
        as_errors("big", 2, "y_error")


def test_error_bars_span_their_lengths_with_caps() -> None:
    v = view()
    style = PointStyle(error_cap=8.0, error_width=2.0)
    errors = as_errors([[1.0], [0.5]], 1, "y_error")
    stroke = error_bar_stroke(v, [(1.0, 0.0)], style, RED, y_error=errors)
    assert stroke is not None
    assert stroke.width == 2.0
    bar, low_cap, high_cap = stroke.paths
    px, _ = v.to_screen(1.0, 0.0)
    assert bar[:, 0].tolist() == [px, px]
    assert bar[:, 1].tolist() == pytest.approx(
        [v.to_screen_y(-1.0), v.to_screen_y(0.5)]
    )
    assert low_cap[:, 0].tolist() == [px - 4, px + 4]  # 8 px across
    assert low_cap[0, 1] == pytest.approx(v.to_screen_y(-1.0))
    assert high_cap[0, 1] == pytest.approx(v.to_screen_y(0.5))
    # x error bars run across x, with upright caps.
    x_errors = as_errors(0.5, 1, "x_error")
    across = error_bar_stroke(v, [(0.0, 1.0)], style, RED, x_error=x_errors)
    assert across is not None
    bar = across.paths[0]
    assert bar[:, 1].tolist() == [v.to_screen_y(1.0)] * 2
    assert np.ptp(across.paths[1][:, 1]) == 8
    no_caps = error_bar_stroke(
        v, [(1.0, 0.0)], PointStyle(error_cap=0), RED, y_error=errors
    )
    assert no_caps is not None
    assert len(no_caps.paths) == 1


def test_error_bars_leave_out_what_cannot_be_drawn() -> None:
    v = view()
    errors = as_errors([math.nan, 0.0, 1.0], 3, "y_error")
    stroke = error_bar_stroke(
        v, [(0, 0), (1, 0), (math.nan, 0)], PointStyle(), RED, y_error=errors
    )
    assert stroke is None  # no length, a zero length, a point with no x
    assert error_bar_stroke(v, [(0, 0)], PointStyle(), RED) is None
    far = error_bar_stroke(
        v, [(100.0, 0.0)], PointStyle(), RED, y_error=as_errors(1, 1, "y")
    )
    assert far is None  # off the plane


def test_error_bars_below_zero_on_a_log_axis_run_off_the_bottom() -> None:
    v = View(RECT, (0.0, 1.0), (50.0, 150.0), equal_aspect=False, log_y=True)
    errors = as_errors([[2.0], [1.0]], 1, "y_error")  # from 1 - 2 to 1 + 1
    stroke = error_bar_stroke(v, [(0.0, 1.0)], PointStyle(), RED, y_error=errors)
    assert stroke is not None
    bar = stroke.paths[0]
    assert max(bar[:, 1]) >= RECT.bottom - 0.5  # reaches the bottom edge
    assert min(bar[:, 1]) == pytest.approx(v.to_screen_y(2.0))
    assert len(stroke.paths) == 2  # a cap at the top only


@settings(max_examples=60, deadline=None)
@given(  # small enough that the bars stay inside the plane
    cx=st.floats(-0.5, 0.5),
    cy=st.floats(-0.5, 0.5),
    dx=st.integers(-20, 20),
    dy=st.integers(-20, 20),
)
def test_error_bars_move_rigidly(cx: float, cy: float, dx: int, dy: int) -> None:
    points = [(0.5, 0.25), (-1.0, 1.5)]
    errors = as_errors([[0.3, 0.6], [0.2, 0.1]], 2, "y_error")
    v = view(center=(cx, cy))
    before = error_bar_stroke(v, points, PointStyle(), RED, y_error=errors)
    v.pan_pixels(dx, dy)
    after = error_bar_stroke(v, points, PointStyle(), RED, y_error=errors)
    assert before is not None
    assert after is not None
    for a, b in zip(before.paths, after.paths, strict=True):
        np.testing.assert_allclose(b, a + np.array([dx, dy]), atol=1e-6)


# --- Markers per point ---------------------------------------------------------------


def test_markers_keep_their_colours_and_sizes_when_some_are_hidden() -> None:
    v = view()
    colors = per_point_colors(["red", "green", "blue"], 3)
    radii = np.array([3.0, 5.0, 0.0])
    points = [(0, 0), (100, 0), (1, 1)]  # the second is off the plane
    markers = point_markers(v, points, PointStyle(), RED, colors=colors, radii=radii)
    assert markers is not None
    assert markers.colors == ("red",)  # the third has no size
    assert markers.radii is not None
    assert markers.radii.tolist() == [3.0]
    byte_colors = per_point_colors([(1, 2, 3), (4, 5, 6), (7, 8, 9)], 3)
    shown = point_markers(v, points, PointStyle(), RED, colors=byte_colors)
    assert shown is not None
    assert isinstance(shown.colors, np.ndarray)
    assert shown.colors.tolist() == [[1, 2, 3, 255], [7, 8, 9, 255]]
    assert point_markers(v, points, PointStyle(radius=0), RED) is None


def test_big_markers_are_kept_while_they_reach_the_plane() -> None:
    v = view()
    edge = v.to_world(RECT.right + 8.0, 150.0)  # 8 px beyond the right edge
    radii = np.array([12.0])
    assert point_markers(v, [edge], PointStyle(), RED, radii=radii) is not None
    assert point_markers(v, [edge], PointStyle(radius=3), RED) is None


# --- Steps -----------------------------------------------------------------------


def test_step_kinds() -> None:
    assert check_step(None) is None
    assert check_step("post") == "post"
    with pytest.raises(ValueError, match="step"):
        check_step("after")


@pytest.mark.parametrize(
    ("step", "u", "w"),
    [
        ("post", [0, 1, 1, 3, 3], [5, 5, 6, 6, 7]),
        ("pre", [0, 0, 1, 1, 3], [5, 6, 6, 7, 7]),
        ("mid", [0, 0.5, 0.5, 2, 2, 3], [5, 5, 6, 6, 7, 7]),
    ],
)
def test_steps_go_across_then_up(step: str, u: list[float], w: list[float]) -> None:
    su, sw = step_vertices(np.array([0.0, 1.0, 3.0]), np.array([5.0, 6.0, 7.0]), step)  # type: ignore[arg-type]
    assert su.tolist() == u
    assert sw.tolist() == w


def test_steps_keep_the_breaks_of_a_broken_line() -> None:
    u = np.array([0.0, 1.0, np.nan, 2.0, 3.0, 4.0])
    w = np.array([1.0, 2.0, 0.0, 3.0, np.nan, 5.0])
    su, sw = step_vertices(u, w, "post")
    assert np.isnan(su).sum() == 2
    assert su[:3].tolist() == [0.0, 1.0, 1.0]
    assert sw[:3].tolist() == [1.0, 1.0, 2.0]
    assert su[-1] == 4.0  # a lone vertex stays a dot
    empty_u, _ = step_vertices(np.array([np.nan]), np.array([np.nan]), "pre")
    assert empty_u.shape == (0,)


def test_mid_steps_are_half_way_on_screen_on_a_log_axis() -> None:
    v = View(RECT, (10.0, 0.0), (150.0, 50.0), equal_aspect=False, log_x=True)
    stroke = polyline_stroke(
        v, [(1.0, 0.0), (100.0, 1.0)], LineStyle(), RED, step="mid"
    )
    assert stroke is not None
    path = stroke.paths[0]
    middle = (v.to_screen_x(1.0) + v.to_screen_x(100.0)) / 2
    assert path[1, 0] == pytest.approx(middle)  # at 10, not 50.5
    assert path[1, 0] == pytest.approx(v.to_screen_x(10.0))


# --- Bars -----------------------------------------------------------------------


def test_bars_from_heights_or_pairs() -> None:
    assert as_bars([3, 1, 4]).tolist() == [[0, 3], [1, 1], [2, 4]]
    assert as_bars([3, 1], start=-1.5).tolist() == [[-1.5, 3], [-0.5, 1]]
    assert as_bars([(2019, 3.1), (2021, 2.4)]).tolist() == [[2019, 3.1], [2021, 2.4]]
    assert as_bars([]).shape == (0, 2)
    with pytest.raises(ValueError, match="pairs"):
        as_bars([[1, 2, 3]])
    with pytest.raises(ValueError, match="start"):
        as_bars([1], start=math.inf)
    with pytest.raises(ValueError, match="numbers"):
        as_bars(["a", "b"])


def test_automatic_bar_widths_leave_gaps() -> None:
    assert auto_bar_width(np.array([0.0, 1.0, 2.0])) == AUTO_BAR_FRACTION
    assert auto_bar_width(np.array([5.0, 0.0, 0.5, np.nan])) == pytest.approx(0.4)
    assert auto_bar_width(np.array([7.0])) == AUTO_BAR_FRACTION
    assert auto_bar_width(np.array([7.0, 7.0])) == AUTO_BAR_FRACTION
    assert check_width(None) is None
    assert check_width(2) == 2.0
    for bad in (0, -1, math.nan, True, "1"):
        with pytest.raises(ValueError, match="width"):
            check_width(bad)


def test_bars_are_rectangles_from_the_baseline() -> None:
    v = view()
    fill, outline = bar_shapes(v, as_bars([2.0, -1.0]), BarStyle(), RED)
    assert fill is not None
    assert outline is not None
    up, down = fill.rings
    assert up[:, 0].min() == pytest.approx(v.to_screen_x(-0.4))
    assert up[:, 0].max() == pytest.approx(v.to_screen_x(0.4))
    assert sorted(set(up[:, 1].tolist())) == pytest.approx(
        [v.to_screen_y(2.0), v.to_screen_y(0.0)]
    )
    assert down[:, 1].max() == pytest.approx(v.to_screen_y(-1.0))  # hangs down
    assert fill.opacity == BarStyle().opacity
    assert outline.opacity == 1.0
    assert outline.width == BarStyle().edge_width
    assert all(path.shape == (5, 2) for path in outline.paths)  # closed rings
    lifted, _ = bar_shapes(v, as_bars([2.0]), BarStyle(), RED, baseline=1.0, width=0.5)
    assert lifted is not None
    ring = lifted.rings[0]
    assert ring[:, 1].max() == pytest.approx(v.to_screen_y(1.0))
    assert np.ptp(ring[:, 0]) == pytest.approx(25.0)


def test_bars_are_left_out_or_clamped_as_needed() -> None:
    v = view()
    data = np.array([[0.0, 0.0], [1.0, math.nan], [100.0, 1.0], [2.0, 1e9]])
    fill, _ = bar_shapes(v, data, BarStyle(), RED)
    assert fill is not None
    (tall,) = fill.rings  # no height, no top, off the plane: left out
    assert tall[:, 1].min() >= RECT.top - 0.5 - 10  # clamped just beyond the top
    assert bar_shapes(v, data[:3], BarStyle(), RED) == (None, None)
    no_fill, edges = bar_shapes(v, data, BarStyle(opacity=0), RED)
    assert no_fill is None
    assert edges is not None
    assert bar_shapes(v, data, BarStyle(edge_width=0), RED)[1] is None


def test_bars_on_log_axes() -> None:
    v = View(RECT, (3.0, 10.0), 150.0, log_x=True, log_y=True)
    fill, _ = bar_shapes(v, np.array([[1.0, 10.0], [10.0, 100.0]]), BarStyle(), RED)
    assert fill is not None
    first, second = fill.rings
    # Equal on screen on a log x axis, and down to the bottom from 0.
    assert np.ptp(first[:, 0]) == pytest.approx(np.ptp(second[:, 0]))
    assert first[:, 1].max() >= RECT.bottom - 0.5


@settings(max_examples=60, deadline=None)
@given(  # small enough that the bars stay inside the plane
    cx=st.floats(-0.5, 0.5),
    cy=st.floats(-0.5, 0.5),
    dx=st.integers(-20, 20),
    dy=st.integers(-20, 20),
)
def test_bars_move_rigidly(cx: float, cy: float, dx: int, dy: int) -> None:
    bars = as_bars([0.5, 1.0, -0.75], start=-1)
    v = view(center=(cx, cy))
    before, _ = bar_shapes(v, bars, BarStyle(), RED)
    v.pan_pixels(dx, dy)
    after, _ = bar_shapes(v, bars, BarStyle(), RED)
    assert before is not None
    assert after is not None
    for a, b in zip(before.rings, after.rings, strict=True):
        np.testing.assert_allclose(b, a + np.array([dx, dy]), atol=1e-6)


# --- Ranges that show data ---------------------------------------------------------


def test_data_ranges_show_the_values_with_a_margin() -> None:
    assert data_range([1, 5, 3]) == pytest.approx((0.8, 5.2))
    assert data_range([1, 5], margin=0) == (1.0, 5.0)
    assert data_range([2, 3], include=0) == pytest.approx((-0.15, 3.15))
    assert data_range(np.array([[1.0, 2.0], [3.0, math.nan]])) == pytest.approx(
        (0.9, 3.1)
    )
    assert data_range([1, 1000], log=True) == pytest.approx((10**-0.15, 10**3.15))
    assert data_range([-1, 0, 10, 100], log=True, margin=0) == pytest.approx((10, 100))


def test_data_ranges_of_one_value_or_none() -> None:
    assert data_range([7, 7]) == pytest.approx((6.3, 7.7))
    assert data_range([0.0]) == (-1.0, 1.0)
    assert data_range([10.0], log=True) == pytest.approx((10**0.5, 10**1.5))
    assert data_range([]) is None
    assert data_range([math.nan, math.inf]) is None
    assert data_range([-3.0], log=True) is None
    for bad in (-0.1, math.inf, "wide"):
        with pytest.raises(ValueError, match="margin"):
            data_range([1, 2], margin=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="numbers"):
        data_range(["a"])


# --- Stacked, floating and horizontal bars -------------------------------------------


def test_baselines_one_for_all_or_one_per_bar() -> None:
    assert as_baselines(2, 3) == 2.0
    bases = as_baselines([1, 2, math.nan], 3)
    assert isinstance(bases, np.ndarray)
    assert np.isnan(bases[2])
    with pytest.raises(ValueError, match="2 values for 3"):
        as_baselines([1, 2], 3)
    with pytest.raises(ValueError, match="finite"):
        as_baselines(math.inf, 3)
    with pytest.raises(ValueError, match="finite"):
        as_baselines([1, math.inf], 2)


def test_bars_stand_on_their_own_baselines() -> None:
    v = view()
    a, b = np.array([1.0, 2.0]), np.array([0.5, 1.0])
    fill, _ = bar_shapes(v, as_bars(a + b), BarStyle(), RED, baseline=a)
    assert fill is not None
    for ring, low, high in zip(fill.rings, a, a + b, strict=True):
        assert sorted(set(ring[:, 1].tolist())) == pytest.approx(
            [v.to_screen_y(high), v.to_screen_y(low)]
        )
    # A NaN baseline leaves its bar out.
    one, _ = bar_shapes(
        v, as_bars([1.0, 2.0]), BarStyle(), RED, baseline=np.array([0, math.nan])
    )
    assert one is not None
    assert len(one.rings) == 1


def test_horizontal_bars_lie_along_x() -> None:
    v = view()
    fill, _ = bar_shapes(
        v, as_bars([2.0, -1.0], start=1), BarStyle(), RED, horizontal=True, baseline=0.5
    )
    assert fill is not None
    first, second = fill.rings
    assert sorted(set(first[:, 0].tolist())) == pytest.approx(
        [v.to_screen_x(0.5), v.to_screen_x(2.0)]
    )
    assert np.ptp(first[:, 1]) == pytest.approx(0.8 * 50)  # 0.8 of a unit tall
    assert first[:, 1].mean() == pytest.approx(v.to_screen_y(1.0))  # at y = 1
    assert second[:, 0].min() == pytest.approx(v.to_screen_x(-1.0))  # to the left
    log = View(RECT, (10.0, 0.0), (150.0, 50.0), equal_aspect=False, log_x=True)
    lying, _ = bar_shapes(log, as_bars([100.0]), BarStyle(), RED, horizontal=True)
    assert lying is not None
    assert lying.rings[0][:, 0].min() <= RECT.left - 0.5  # from 0: the left edge
