"""Tests for :mod:`pygridplot.core.view`: transforms, pan, zoom and limits."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pygridplot.core.coords import ScreenRect
from pygridplot.core.view import View, ViewLimits, ViewState

RECT = ScreenRect(0, 0, 800, 600)


def make_view(**kwargs: object) -> View:
    """Return an 800x600 view, overriding constructor arguments as given."""
    return View(kwargs.pop("rect", RECT), **kwargs)  # type: ignore[arg-type]


def test_origin_is_a_whole_pixel_at_the_middle() -> None:
    assert make_view().origin == (400.0, 300.0)
    assert make_view(rect=ScreenRect(10, 20, 101, 51)).origin == (60.0, 45.0)


def test_world_origin_maps_to_origin_pixel_and_y_points_up() -> None:
    view = make_view()
    assert view.to_screen(0.0, 0.0) == (400.0, 300.0)
    px, py = view.to_screen(1.0, 1.0)
    assert px == 450.0  # x grows to the right
    assert py == 250.0  # y grows upwards, i.e. towards smaller screen y


def test_transforms_accept_arrays() -> None:
    view = make_view(center=(2.0, -1.0), scale=10.0)
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([-1.0, 0.0, 1.0])
    px, py = view.to_screen(xs, ys)
    assert isinstance(px, np.ndarray)
    np.testing.assert_allclose(px, [380.0, 390.0, 400.0])
    np.testing.assert_allclose(py, [300.0, 290.0, 280.0])
    wx, wy = view.to_world(px, py)
    np.testing.assert_allclose(wx, xs)
    np.testing.assert_allclose(wy, ys)


@given(
    cx=st.floats(-1e6, 1e6),
    cy=st.floats(-1e6, 1e6),
    scale=st.floats(1e-3, 1e5),
    px=st.floats(-1000, 2000),
    py=st.floats(-1000, 2000),
)
def test_screen_world_round_trip(
    cx: float, cy: float, scale: float, px: float, py: float
) -> None:
    view = make_view(center=(cx, cy), scale=scale)
    x, y = view.to_world(px, py)
    back_x, back_y = view.to_screen(x, y)
    tolerance = 1e-6 * max(1.0, abs(cx) * scale, abs(cy) * scale)
    assert back_x == pytest.approx(px, abs=tolerance)
    assert back_y == pytest.approx(py, abs=tolerance)


def test_bounds_cover_the_rect_in_pixel_centre_convention() -> None:
    view = make_view(scale=100.0)
    b = view.bounds
    # Pixels 0..799 cover continuous x in [-0.5, 799.5).
    assert b.x_min == pytest.approx((-0.5 - 400) / 100)
    assert b.x_max == pytest.approx((799.5 - 400) / 100)
    assert b.y_max == pytest.approx((300 + 0.5) / 100)
    assert b.y_min == pytest.approx(-(599.5 - 300) / 100)
    assert b.width == pytest.approx(8.0)
    assert b.height == pytest.approx(6.0)


def test_pan_pixels_moves_content_with_the_mouse() -> None:
    view = make_view(scale=50.0)
    before = view.to_screen(1.0, 1.0)
    view.pan_pixels(30.0, -20.0)
    after = view.to_screen(1.0, 1.0)
    assert after[0] == pytest.approx(before[0] + 30.0)
    assert after[1] == pytest.approx(before[1] - 20.0)


def test_pin_places_a_world_point_exactly_under_a_pixel() -> None:
    view = make_view(scale=37.0, center=(5.0, 7.0))
    view.pin((1.25, -3.5), (123.0, 456.0))
    px, py = view.to_screen(1.25, -3.5)
    assert px == pytest.approx(123.0)
    assert py == pytest.approx(456.0)


@given(
    factor=st.floats(0.05, 20.0),
    px=st.floats(0, 800),
    py=st.floats(0, 600),
    cx=st.floats(-1e3, 1e3),
    cy=st.floats(-1e3, 1e3),
)
def test_zoom_at_keeps_the_point_under_the_cursor_fixed(
    factor: float, px: float, py: float, cx: float, cy: float
) -> None:
    view = make_view(center=(cx, cy), scale=40.0)
    before = view.to_world(px, py)
    view.zoom_at(px, py, factor)
    after = view.to_world(px, py)
    assert after[0] == pytest.approx(before[0], rel=1e-9, abs=1e-9)
    assert after[1] == pytest.approx(before[1], rel=1e-9, abs=1e-9)
    assert view.scale == pytest.approx(40.0 * factor)


def test_zoom_is_clamped_but_still_keeps_the_cursor_point() -> None:
    limits = ViewLimits(min_scale=1.0, max_scale=1000.0)
    view = make_view(scale=500.0, limits=limits)
    anchor = view.to_world(100.0, 100.0)
    applied = view.zoom_at(100.0, 100.0, 10.0)
    assert view.scale == 1000.0
    assert applied == pytest.approx(2.0)
    assert view.to_world(100.0, 100.0) == pytest.approx(anchor)
    view.zoom(1e-9)
    assert view.scale == 1.0


def test_relative_precision_limit_depends_on_the_centre() -> None:
    limits = ViewLimits(max_scale=1e14, min_relative_pixel=1e-12)
    view = make_view(center=(1e6, 0.0), limits=limits)
    view.zoom(1e30)
    assert view.scale == pytest.approx(1.0 / (1e6 * 1e-12))
    # Near the origin the absolute maximum applies instead.
    near_origin = make_view(limits=limits)
    near_origin.zoom(1e30)
    assert near_origin.scale == pytest.approx(1e14)


def test_moving_the_centre_far_away_re_clamps_the_scale() -> None:
    view = make_view(scale=1e12)
    view.center = (1e8, 0.0)
    assert view.scale <= 1.0 / (1e8 * view.limits.min_relative_pixel) * (1 + 1e-12)


def test_centre_is_clamped() -> None:
    view = make_view(limits=ViewLimits(max_abs_center=100.0))
    view.center = (1e9, -1e9)
    assert view.center == (100.0, -100.0)
    view.pan_world(1e9, 0.0)
    assert view.center_x == 100.0


def test_single_axis_zoom_requires_unequal_aspect() -> None:
    with pytest.raises(ValueError, match="equal_aspect"):
        make_view().zoom_at(10, 10, 2.0, axis="x")
    view = make_view(equal_aspect=False, scale=(10.0, 20.0))
    view.zoom_at(400, 300, 2.0, axis="x")
    assert (view.scale_x, view.scale_y) == (20.0, 20.0)
    view.zoom_at(400, 300, 3.0, axis="y")
    assert (view.scale_x, view.scale_y) == (20.0, 60.0)


def test_scale_property_is_ambiguous_for_unequal_axes() -> None:
    view = make_view(equal_aspect=False, scale=(10.0, 20.0))
    with pytest.raises(ValueError, match="scale_x"):
        _ = view.scale


def test_unequal_scales_rejected_with_equal_aspect() -> None:
    with pytest.raises(ValueError, match="equal_aspect"):
        make_view(scale=(1.0, 2.0))


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_invalid_scale_and_zoom_factors_raise(bad: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        make_view(scale=bad)
    with pytest.raises(ValueError, match="zoom factor"):
        make_view().zoom(bad)


def test_resizing_keeps_centre_and_scale() -> None:
    view = make_view(center=(3.0, 4.0), scale=25.0)
    view.rect = ScreenRect(0, 0, 1600, 1200)
    assert view.center == (3.0, 4.0)
    assert view.scale == 25.0
    assert view.to_screen(3.0, 4.0) == (800.0, 600.0)


def test_fit_with_equal_aspect_uses_the_tighter_range() -> None:
    view = make_view()
    view.fit(x_range=(-4.0, 4.0), y_range=(-1.0, 1.0))
    assert view.center == (0.0, 0.0)  # the origin pixel stays a whole pixel
    # The shorter half-extent of an 800 px side is 399.5 px.
    assert view.scale == pytest.approx(min(399.5 / 4, 299.5 / 1))
    b = view.bounds
    assert b.x_min <= -4.0
    assert b.x_max >= 4.0
    assert b.y_min <= -1.0
    assert b.y_max >= 1.0


def test_fit_with_unequal_aspect_fills_both_ranges() -> None:
    view = make_view(equal_aspect=False)
    view.fit(x_range=(0.0, 8.0), y_range=(10.0, 16.0))
    assert view.center == (4.0, 13.0)
    assert view.scale_x == pytest.approx(399.5 / 4)
    assert view.scale_y == pytest.approx(299.5 / 3)
    view.fit(y_range=(0.0, 1.0))
    assert view.scale_x == pytest.approx(399.5 / 4)
    assert view.scale_y == pytest.approx(299.5 / 0.5)


def test_fit_rejects_empty_ranges() -> None:
    with pytest.raises(ValueError, match="low < high"):
        make_view().fit(x_range=(1.0, 1.0))


def test_state_restore_and_copy() -> None:
    view = make_view(center=(1.0, 2.0), scale=3.0)
    saved = view.state()
    assert saved == ViewState(1.0, 2.0, 3.0, 3.0)
    clone = view.copy()
    view.pan_world(10.0, 10.0)
    view.zoom(2.0)
    assert clone.center == (1.0, 2.0)
    view.restore(saved)
    assert view.state() == saved


def test_multiples_to_screen_matches_naive_transform() -> None:
    view = make_view(center=(0.37, -12.5), scale=123.0)
    indices = np.arange(-20, 21, dtype=np.int64)
    step = 0.25
    precise_x = view.multiples_to_screen("x", step, indices)
    precise_y = view.multiples_to_screen("y", step, indices)
    np.testing.assert_allclose(precise_x, view.to_screen_x(indices * step))
    np.testing.assert_allclose(precise_y, view.to_screen_y(indices * step))


def test_multiples_to_screen_keeps_even_spacing_when_deeply_zoomed() -> None:
    # Centre far from the origin and a tiny step (beyond the default limits):
    # naive subtraction would produce uneven spacing; the relative formula
    # keeps the gaps even.
    limits = ViewLimits(max_scale=1e15, min_relative_pixel=1e-16)
    view = make_view(center=(12345.678, 0.0), scale=1e11, limits=limits)
    step = 1e-9
    nearest = round(12345.678 / step)
    indices = np.arange(nearest - 40, nearest + 41, dtype=np.int64)
    positions = view.multiples_to_screen("x", step, indices)
    gaps = np.diff(positions)
    np.testing.assert_allclose(gaps, step * 1e11, rtol=1e-3)


# ----------------------------------------------------------------------
# Logarithmic axes
# ----------------------------------------------------------------------
def log_log(**kwargs: object) -> View:
    """Return an 800x600 log-log view."""
    return make_view(log_x=True, log_y=True, **kwargs)


def test_log_axes_default_to_a_centre_of_one_and_decades() -> None:
    view = log_log()
    assert view.center == (1.0, 1.0)
    assert view.scale == 150.0  # pixels per decade
    assert view.equal_aspect
    assert view.to_screen(10.0, 100.0) == pytest.approx((550.0, 0.0))
    assert view.axis_center("x") == 0.0  # transformed: log10(1)


def test_mixed_axes_have_independent_default_scales() -> None:
    view = make_view(log_y=True)
    assert not view.equal_aspect
    assert (view.scale_x, view.scale_y) == (50.0, 150.0)
    assert view.center == (0.0, 1.0)


def test_log_axis_rejects_a_centre_it_cannot_show() -> None:
    with pytest.raises(ValueError, match="positive"):
        log_log(center=(0.0, 1.0))
    view = log_log()
    with pytest.raises(ValueError, match="positive"):
        view.center = (1.0, -2.0)
    with pytest.raises(ValueError, match="positive"):
        view.pan_world(-5.0, 0.0)


def test_log_bounds_are_positive_and_transformed_bounds_are_decades() -> None:
    view = log_log(center=(100.0, 0.1), scale=100.0)
    bounds = view.bounds
    assert bounds.x_min == pytest.approx(10**-2.005)
    assert bounds.x_max == pytest.approx(10**5.995)
    t = view.transformed_bounds
    assert (t.x_min, t.x_max) == pytest.approx((-2.005, 5.995))
    assert (t.y_min, t.y_max) == pytest.approx((-3.995, 2.005))  # y points up


@given(
    x=st.floats(1e-200, 1e200),
    y=st.floats(1e-200, 1e200),
    scale=st.floats(1.0, 1e6),
)
def test_log_round_trip(x: float, y: float, scale: float) -> None:
    view = log_log(center=(3.0, 0.02), scale=scale)
    px, py = view.to_screen(x, y)
    assert view.to_world(px, py) == pytest.approx((x, y), rel=1e-9)


def test_values_a_log_axis_cannot_show_go_off_screen() -> None:
    view = log_log()
    assert math.isnan(view.to_screen_x(-1.0))
    assert view.to_screen_x(0.0) == -math.inf  # infinitely far to the left
    assert view.to_screen_y(0.0) == math.inf  # infinitely far down


def test_zoom_at_keeps_the_cursor_value_on_a_log_axis() -> None:
    view = log_log(center=(5.0, 5.0))
    before = view.to_world(123.0, 456.0)
    view.zoom_at(123.0, 456.0, 3.7)
    assert view.to_world(123.0, 456.0) == pytest.approx(before, rel=1e-12)
    assert view.scale == pytest.approx(150.0 * 3.7)


def test_fit_on_log_axes_works_in_decades() -> None:
    view = make_view(log_x=True, equal_aspect=False)
    view.fit(x_range=(1.0, 1000.0), y_range=(-1.0, 1.0))
    assert view.center_x == pytest.approx(10**1.5)
    assert view.scale_x == pytest.approx(399.5 / 1.5)  # pixels per decade
    assert view.bounds.x_min <= 1.0
    assert view.bounds.x_max >= 1000.0
    with pytest.raises(ValueError, match="positive"):
        view.fit(x_range=(0.0, 10.0))


def test_log_axes_limit_their_centre_and_zoom() -> None:
    view = log_log()
    view.pan_pixels(-1e9, 0.0)
    assert view.axis_center("x") == 300.0  # 10**300 at most
    view.set_scale(1e-6)
    assert view.scale == 1.0  # at least a pixel per decade
    view.set_scale(1e20)
    assert view.scale < 1e14  # still precise near the (huge) centre


def test_set_log_keeps_the_visible_range() -> None:
    view = make_view(center=(50.0, 0.0), scale=10.0, equal_aspect=False)
    for log in (True, False):
        low, high = view.bounds.range("x")  # first 9.95 to 89.95
        view.set_log("x", log=log)
        assert view.log_x is log
        # The old range still spans the plane, with at most a pixel to spare.
        left, right = view.to_screen_x(low), view.to_screen_x(high)
        assert -0.5 - 1e-6 <= left <= 0.5 + 1e-6
        assert 798.5 - 1e-6 <= right <= 799.5 + 1e-6


def test_switching_to_log_with_negatives_visible_keeps_the_top_decades() -> None:
    view = make_view(equal_aspect=False)  # x from -8 to 8
    view.set_log("x", log=True)
    x_min, x_max = view.bounds.range("x")
    assert x_max >= 7.99 * (1 - 1e-12)
    assert x_min <= 7.99e-3 * (1 + 1e-12)
    assert (x_min, x_max) == pytest.approx((7.99e-3, 7.99), rel=0.01)
    view = make_view(center=(-100.0, 0.0), equal_aspect=False)  # nothing positive
    view.set_log("x", log=True)
    assert view.center_x == 1.0
    assert view.scale_x == 150.0


def test_state_restore_ignores_centres_a_log_axis_cannot_show() -> None:
    linear = make_view(center=(-3.0, 4.0))
    view = log_log(center=(7.0, 7.0))
    view.restore(linear.state())
    assert view.center_x == pytest.approx(7.0)  # -3 is not shown on a log axis
    assert view.center_y == pytest.approx(4.0)


def test_equal_aspect_can_be_switched_on_later() -> None:
    view = make_view(scale=(20.0, 80.0), equal_aspect=False)
    view.equal_aspect = True
    assert view.scale == 20.0  # the tighter of the two


def test_copy_keeps_the_axis_kinds() -> None:
    view = make_view(log_y=True)
    clone = view.copy()
    assert clone.log_y
    assert not clone.log_x
    assert "log_y=True" in repr(clone)
