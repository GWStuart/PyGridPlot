"""Tests of points: fixed, derived from other objects, and dragged."""

from __future__ import annotations

import gc
import math
import weakref

import numpy as np
import pygame
import pytest

import pygridplot as pgp

from ..graph.conftest import drag, make_plane, mouse, pixels


def screen(plane: pgp.Plane, point: pgp.Point) -> tuple[float, float]:
    """Return where a point is on the screen."""
    return plane.to_screen(point.x, point.y)


def move_away(plane: pgp.Plane) -> None:
    """Move the cursor off every point (so that none shows a hover ring)."""
    plane.handle_event(
        pygame.event.Event(
            pygame.MOUSEMOTION, pos=(0, 0), rel=(0, 0), buttons=(0, 0, 0)
        )
    )


# --- Fixed and derived points ---------------------------------------------------


def test_a_fixed_point() -> None:
    p = pgp.Point((1, 2))
    assert (p.x, p.y, p.position, p.source) == (1.0, 2.0, (1.0, 2.0), (1.0, 2.0))
    p.x = 3
    assert p.position == (3.0, 2.0)
    p.position = (-1, 0.5)
    assert p.position == (-1.0, 0.5)
    assert not p.draggable
    assert p.handles(make_plane()[0]) == ()


def test_derived_points_follow_what_they_use() -> None:
    q = pgp.Point((1, 2))
    f = pgp.Function(lambda x: x**2)
    whole = pgp.Point(lambda: (q.x, f(q.x)))
    parts = pgp.Point((lambda: q.y, 5))
    text = pgp.Point(("1 + 1", "sqrt(9)"))
    assert whole.position == (1.0, 1.0)
    assert parts.position == (2.0, 5.0)
    assert text.position == (2.0, 3.0)
    q.position = (3, -1)
    assert whole.position == (3.0, 9.0)
    assert parts.position == (-1.0, 5.0)
    assert callable(whole.source)


def test_a_derived_point_that_fails_is_undefined() -> None:
    def broken() -> tuple[float, float]:
        raise ZeroDivisionError

    p = pgp.Point(broken)
    assert math.isnan(p.x)
    assert math.isnan(p.y)
    plane, surface = make_plane()
    plane.add(p)
    plane.draw(surface)  # draws nothing, does not raise


def test_a_derived_point_is_drawn_where_it_is() -> None:
    q = pgp.Point((1, 2))
    plane, surface = make_plane()
    plane.add(pgp.Point(lambda: (q.x + 1, q.y), color="black"))
    q.position = (0, 1)
    plane.draw(surface)
    expected_plane, expected = make_plane()
    expected_plane.points(expected, [(1, 1)], color="black")
    assert pixels(surface) == pixels(expected)


# --- Draggable points -----------------------------------------------------------


def test_dragging_moves_the_point_and_what_uses_it_in_the_same_frame() -> None:
    plane, surface = make_plane()
    a = pgp.Point((1, 1), draggable=True)
    f = pgp.Function(lambda x: a.y + 0 * x, color="black")
    plane.add(f, a)  # f before a: it must still see this frame's drag
    plane.draw(surface)
    sx, sy = screen(plane, a)
    scale = plane.view.scale
    assert drag(plane, (sx, sy), (sx + 2 * scale, sy - scale))
    move_away(plane)
    plane.draw(surface)
    assert a.position == pytest.approx((3.0, 2.0))
    expected_plane, expected = make_plane()
    expected_plane.plot(expected, lambda x: 2.0 + 0 * x, color="black")
    handle = expected_plane.draggable((3.0, 2.0), color=pgp.LIGHT.palette[1])
    expected_plane.draw_draggables(expected, handle)
    assert pixels(surface) == pixels(expected)


def test_setting_a_draggable_point_moves_its_handle() -> None:
    plane, surface = make_plane()
    a = plane.add(pgp.Point((1, 1), draggable=True, x_range=(0, 2)))
    plane.draw(surface)
    a.x = 5  # kept within its bounds
    plane.draw(surface)
    assert a.position == (2.0, 1.0)
    assert plane.draggables[0].position == (2.0, 1.0)


def test_draggable_can_be_switched() -> None:
    plane, surface = make_plane()
    a = plane.add(pgp.Point((1, 1)))
    plane.draw(surface)
    assert plane.draggables == ()
    a.draggable = True
    plane.draw(surface)
    assert len(plane.draggables) == 1
    a.draggable = False
    assert plane.draggables == ()  # at once
    plane.draw(surface)
    assert a.position == (1.0, 1.0)
    derived = pgp.Point(lambda: (0, 0))
    with pytest.raises(ValueError, match="must be numbers"):
        derived.draggable = True


def test_constraints_need_a_draggable_point() -> None:
    with pytest.raises(ValueError, match="pass draggable=True"):
        pgp.Point((0, 0), lock="x")
    p = pgp.Point((0, 0))
    with pytest.raises(ValueError, match="for draggable points"):
        p.lock = "x"
    with pytest.raises(ValueError, match="cannot also be locked"):
        pgp.Point(curve=pgp.Function(np.sin), lock="x", draggable=True)
    with pytest.raises(ValueError, match="needs a position"):
        pgp.Point(draggable=True)


def test_a_point_on_a_function_object() -> None:
    plane, surface = make_plane()
    f = pgp.Function("x^2 / 4")
    p = pgp.Point(curve=f, t=2, draggable=True)
    plane.add(f, p)
    plane.draw(surface)
    assert p.position == pytest.approx((2.0, 1.0))
    sx, sy = screen(plane, p)
    drag(plane, (sx, sy), (sx - 40, sy), (sx - 80, sy))
    plane.draw(surface)
    assert p.t is not None
    assert p.t < 2
    assert p.position == pytest.approx((p.t, p.t**2 / 4))
    f.func = "x^3 / 8"  # it follows the changed function
    plane.draw(surface)
    assert p.position == pytest.approx((p.t, p.t**3 / 8))


def test_a_point_on_a_parametric_or_polar_object_or_its_own_curve() -> None:
    plane, surface = make_plane()
    circle = pgp.ParametricCurve(lambda t: 2 * np.cos(t), lambda t: 2 * np.sin(t))
    rose = pgp.PolarCurve(lambda th: 1 + 0 * th)
    on_circle = pgp.Point(curve=circle, t=math.pi / 2, draggable=True)
    on_rose = pgp.Point(curve=rose, draggable=True)
    own = pgp.Point(curve=("t", "t^2", (-1, 1)), t=0.5, draggable=True)
    plane.add(on_circle, on_rose, own)  # the curves need not be drawn
    plane.draw(surface)
    assert on_circle.position == pytest.approx((0.0, 2.0), abs=1e-12)
    assert on_rose.position == pytest.approx((1.0, 0.0))
    assert own.position == pytest.approx((0.5, 0.25))
    own.t = -1
    plane.draw(surface)
    assert own.position == pytest.approx((-1.0, 1.0))


def test_a_curve_by_name_needs_a_graph() -> None:
    plane, surface = make_plane()
    p = plane.add(pgp.Point(curve="f", draggable=True))
    with pytest.warns(pgp.PlotWarning, match="needs the point to be in a graph"):
        plane.draw(surface)
    assert plane.draggables == ()
    assert p.error is not None


def test_a_curve_with_an_error_stops_the_point() -> None:
    plane, surface = make_plane()
    f = pgp.Function("q x")
    p = plane.add(pgp.Point(curve=f, draggable=True))
    with pytest.warns(pgp.PlotWarning, match="has an error"):
        plane.draw(surface)
    assert p.error is not None
    f.func = "x"
    plane.draw(surface)
    assert p.error is None
    assert p.position == pytest.approx((0.0, 0.0))


def test_hidden_or_disabled_points_cannot_be_grabbed() -> None:
    plane, surface = make_plane()
    p = plane.add(pgp.Point((0, 0), draggable=True))
    plane.draw(surface)

    def grabs() -> bool:
        at = screen(plane, p)
        plane.handle_event(mouse(pygame.MOUSEBUTTONDOWN, at))
        grabbed = plane.dragged is not None
        plane.handle_event(mouse(pygame.MOUSEBUTTONUP, at))
        return grabbed

    assert grabs()
    p.visible = False
    plane.draw(surface)
    assert not grabs()
    p.visible = True
    p.enabled = False
    plane.draw(surface)
    assert not grabs()


def test_a_point_on_two_planes_stays_in_step() -> None:
    p = pgp.Point((0, 0), draggable=True)
    first, first_surface = make_plane()
    second, second_surface = make_plane()
    first.add(p)
    second.add(p)
    first.draw(first_surface)
    second.draw(second_surface)
    sx, sy = screen(first, p)
    drag(first, (sx, sy), (sx + 40, sy - 20))
    first.draw(first_surface)
    second.draw(second_surface)
    assert second.draggables[0].position == first.draggables[0].position == p.position
    assert p.dragging is False


def test_points_do_not_keep_planes_alive() -> None:
    p = pgp.Point((0, 0), draggable=True)
    plane, surface = make_plane()
    plane.add(p)
    plane.draw(surface)
    alive = weakref.ref(plane)
    del plane
    gc.collect()
    assert alive() is None
    assert p.handles(make_plane()[0]) == ()
