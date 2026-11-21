"""Tests for draggable points and their constraints (:mod:`core.handles`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pygridplot.core.coords import ScreenRect
from pygridplot.core.handles.point import DraggablePoint
from pygridplot.core.view import View

RECT = ScreenRect(0, 0, 800, 600)


def make_view(**kwargs: object) -> View:
    """Return an 800x600 view, 50 px per unit about the origin by default."""
    kwargs.setdefault("scale", 50.0)
    return View(RECT, **kwargs)  # type: ignore[arg-type]


def unit_circle(view: View, **kwargs: object) -> DraggablePoint:
    """Return a point on the unit circle."""
    return DraggablePoint(view, curve=(np.cos, np.sin, (0, 2 * math.pi)), **kwargs)  # type: ignore[arg-type]


# --- Free and locked points ----------------------------------------------------


def test_free_points_follow_the_target() -> None:
    view = make_view()
    point = DraggablePoint(view, (1.0, 2.0))
    assert point.drag_to(500, 200)
    assert point.position == pytest.approx((2.0, 2.0))
    assert not point.drag_to(500, 200)  # no change, no move


def test_locked_points_keep_their_locked_coordinate() -> None:
    view = make_view()
    vertical = DraggablePoint(view, (1.0, 2.0), lock="x")
    vertical.drag_to(600, 450)
    assert vertical.position == pytest.approx((1.0, -3.0))
    horizontal = DraggablePoint(view, (1.0, 2.0), lock="y")
    horizontal.drag_to(600, 450)
    assert horizontal.position == pytest.approx((4.0, 2.0))


def test_bounds_clamp_dragging_and_setting() -> None:
    view = make_view()
    point = DraggablePoint(view, (0.5, 0.5), x_range=(0, 1), y_range=(-1, 1))
    point.drag_to(700, 0)  # far beyond both bounds
    assert point.position == (1.0, 1.0)
    point.position = (-5.0, 0.25)
    assert point.position == (0.0, 0.25)
    start = DraggablePoint(view, (3.0, 0.0), x_range=(0, 1))
    assert start.x == 1.0  # the start is clamped too


def test_locks_restrict_only_dragging() -> None:
    point = DraggablePoint(make_view(), (1.0, 2.0), lock="x")
    point.position = (3.0, 4.0)
    assert point.position == (3.0, 4.0)


@settings(max_examples=60, deadline=None)
@given(
    dx=st.integers(-200, 200),
    dy=st.integers(-200, 200),
    scale=st.floats(1e-3, 1e6),
)
def test_dragging_by_whole_pixels_moves_by_that_many_pixels(
    dx: int, dy: int, scale: float
) -> None:
    view = make_view(center=(0.3, -0.7), scale=scale)
    point = DraggablePoint(view, (0.3, -0.7))
    sx, sy = point.screen_position
    point.drag_to(sx + dx, sy + dy)
    nx, ny = point.screen_position
    assert nx - sx == pytest.approx(dx, abs=1e-6)
    assert ny - sy == pytest.approx(dy, abs=1e-6)


def test_log_axes_keep_points_positive() -> None:
    view = make_view(log_x=True, log_y=True, center=(10.0, 10.0), scale=100.0)
    point = DraggablePoint(view, (10.0, 10.0))
    point.drag_to(500, 200)  # one decade right and up
    assert point.position == pytest.approx((100.0, 100.0))
    bounded = DraggablePoint(view, (10.0, 10.0), x_range=(-5.0, 50.0))
    bounded.drag_to(0, 300)  # four decades left: 1e-3, within the bounds
    assert bounded.x == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="log axis"):
        DraggablePoint(view, (-1.0, 1.0))


# --- Points on curves ----------------------------------------------------------


def test_points_on_curves_stay_exactly_on_them() -> None:
    view = make_view()
    point = unit_circle(view)
    assert point.position == pytest.approx((1.0, 0.0))
    assert point.t == 0.0
    for angle in np.linspace(0.1, 6.2, 25):
        point.drag_to(400 + 150 * math.cos(angle), 300 - 150 * math.sin(angle))
        assert math.hypot(point.x, point.y) == pytest.approx(1.0, abs=1e-12)
        # Refined against the curve itself: well within 0.01 px on screen.
        assert 50 * abs(point.t - angle) < 0.01  # type: ignore[operator]


def test_points_on_curves_start_where_asked() -> None:
    view = make_view()
    assert unit_circle(view, t=math.pi).position == pytest.approx((-1.0, 0.0))
    near = unit_circle(view, position=(0.0, 3.0))
    assert near.t == pytest.approx(math.pi / 2, abs=1e-5)


def test_setting_t_and_position_on_curves() -> None:
    view = make_view()
    point = unit_circle(view)
    point.t = 10.0  # clamped to the range
    assert point.t == 2 * math.pi
    point.position = (-2.0, -2.0)
    assert point.t == pytest.approx(5 * math.pi / 4, abs=1e-5)


def test_curves_can_change_and_keep_the_parameter() -> None:
    view = make_view()
    point = unit_circle(view, t=math.pi / 2)
    point.curve = (lambda t: 2 * np.cos(t), lambda t: 2 * np.sin(t), (0, 2 * math.pi))
    assert point.position == pytest.approx((0.0, 2.0))
    assert point.curve is not None


def test_curves_follow_log_axes_as_seen() -> None:
    view = make_view(log_x=True, log_y=True, center=(1.0, 1.0), scale=100.0)
    # y = x², a straight line on log-log axes, for x in [0.1, 10].
    point = DraggablePoint(view, curve=(lambda t: t, lambda t: t**2, (0.1, 10.0)))
    point.drag_to(450, 200)  # perpendicular foot on the line of slope 2
    assert point.y == pytest.approx(point.x**2)
    log_x = math.log10(point.x)
    assert log_x == pytest.approx((0.5 + 2 * 1.0) / 5, abs=1e-4)


def test_undefined_curves() -> None:
    view = make_view()
    point = DraggablePoint(view, curve=(np.sqrt, np.sqrt, (-1.0, 1.0)))
    assert not point.visible  # undefined at t = -1
    assert point.drag_to(450, 250)  # but it can still be dragged onto the curve
    assert point.visible
    assert point.x == pytest.approx(point.y)


# --- Validation and state ---------------------------------------------------


def test_invalid_settings_are_rejected() -> None:
    view = make_view()
    curve = (np.cos, np.sin, (0, 1))
    with pytest.raises(ValueError, match="locked"):
        DraggablePoint(view, curve=curve, lock="x")
    with pytest.raises(ValueError, match="t_range"):
        DraggablePoint(view, curve=curve, x_range=(0, 1))
    with pytest.raises(ValueError, match="position"):
        DraggablePoint(view)
    with pytest.raises(ValueError, match="lock"):
        DraggablePoint(view, (0, 0), lock="z")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="x_range"):
        DraggablePoint(view, (0, 0), x_range=(1, 0))
    with pytest.raises(ValueError, match="coordinates"):
        DraggablePoint(view, (0, 0), coordinates="sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="curve"):
        DraggablePoint(view, curve=(np.cos, np.sin))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only for points on a curve"):
        DraggablePoint(view, (0, 0), t=1.0)
    free = DraggablePoint(view, (0, 0))
    with pytest.raises(ValueError, match="parameter"):
        free.t = 1.0
    with pytest.raises(ValueError, match="curve"):
        free.curve = curve


def test_moved_reports_changes_between_frames() -> None:
    point = DraggablePoint(make_view(), (0.0, 0.0))
    point.begin_frame()
    assert not point.moved
    point.drag_to(450, 300)
    point.begin_frame()
    assert point.moved
    point.begin_frame()
    assert not point.moved


def test_remove_calls_back_once() -> None:
    removed = []
    point = DraggablePoint(make_view(), (0.0, 0.0), on_remove=removed.append)
    point.remove()
    point.remove()
    assert removed == [point]
