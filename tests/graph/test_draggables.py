"""Tests of draggable points in graphs: the item and the plane's point in step."""

from __future__ import annotations

import gc
import math
import weakref

import numpy as np
import pygame
import pytest

import pygridplot as pgp
from pygridplot.graph import DraggableItem, PlotItem

from .conftest import drag, frame, make_plane, mouse, pixels


def screen(plane: pgp.Plane, point: DraggableItem) -> tuple[float, float]:
    """Return where an item's point is on the screen."""
    return plane.to_screen(point.x, point.y)


def grabs(plane: pgp.Plane, at: tuple[float, float]) -> bool:
    """Return whether pressing at a screen position grabs a point (then release)."""
    plane.handle_event(mouse(pygame.MOUSEBUTTONDOWN, at))
    grabbed = plane.dragged is not None
    plane.handle_event(mouse(pygame.MOUSEBUTTONUP, at))
    return grabbed


def test_the_point_appears_on_the_plane_when_drawn() -> None:
    graph = pgp.Graph()
    point = graph.draggable((1, 2), name="P")
    plane, surface = make_plane()
    assert plane.draggables == ()
    graph.draw(plane, surface)
    (point,) = plane.draggables
    assert point.position == (1.0, 2.0)
    assert point.dragging is False


def test_dragging_moves_the_item_and_what_uses_it() -> None:
    graph = pgp.Graph()
    point = graph.draggable((1, 1), name="P")
    graph.add("y = P.y + 0x")
    plane, surface = make_plane()
    graph.draw(plane, surface)
    sx, sy = screen(plane, point)
    scale = plane.view.scale
    assert drag(plane, (sx, sy), (sx + 1.5 * scale, sy - 0.5 * scale))
    away = pygame.event.Event(
        pygame.MOUSEMOTION, pos=(0, 0), rel=(0, 0), buttons=(0, 0, 0)
    )
    plane.handle_event(away)  # no hover ring, to compare with a fresh plane
    frame(plane, surface, graph)
    assert point.position == pytest.approx((2.5, 1.5))
    assert graph.evaluate("P.x") == pytest.approx(2.5)
    # The plot used the new position in the same frame.
    expected_plane, expected = make_plane()
    expected_plane.plot(expected, lambda x: 1.5 + 0 * x, color=pgp.LIGHT.palette[1])
    handle = expected_plane.draggable((2.5, 1.5), color=pgp.LIGHT.palette[0])
    expected_plane.draw_draggables(expected, handle)
    assert pixels(surface) == pixels(expected)


def test_setting_the_position_moves_the_point() -> None:
    graph = pgp.Graph()
    point = graph.draggable((1, 1))
    plane, surface = make_plane()
    graph.draw(plane, surface)
    point.position = (-2, 0.5)
    frame(plane, surface, graph)
    assert plane.draggables[0].position == (-2.0, 0.5)
    assert point.position == (-2.0, 0.5)


def test_the_position_is_kept_within_the_bounds() -> None:
    graph = pgp.Graph()
    point = graph.draggable((1, 1), x_range=(0, 2), lock="y")
    plane, surface = make_plane()
    graph.draw(plane, surface)
    sx, sy = screen(plane, point)
    drag(plane, (sx, sy), (sx + 300, sy - 80))
    frame(plane, surface, graph)
    assert point.position == (2.0, 1.0)  # dragged: y locked, x bounded
    point.position = (-5, 3)  # from code, the locked coordinate may change too
    frame(plane, surface, graph)
    assert point.position == (0.0, 3.0)


def test_changing_the_constraint_makes_a_new_point() -> None:
    graph = pgp.Graph()
    point = graph.draggable((1, 1))
    plane, surface = make_plane()
    graph.draw(plane, surface)
    first = plane.draggables[0]
    point.lock = "x"
    frame(plane, surface, graph)
    (second,) = plane.draggables
    assert second is not first
    assert second.lock == "x"
    assert second.position == (1.0, 1.0)
    sx, sy = screen(plane, point)
    drag(plane, (sx, sy), (sx + 50, sy - 50))
    frame(plane, surface, graph)
    assert point.x == 1.0
    assert point.y > 1.0


def test_a_point_on_a_named_plot() -> None:
    graph = pgp.Graph()
    f = graph.add("f(x) = x^2 / 4")
    dot = graph.draggable(curve="f", t=2.0, name="D")
    plane, surface = make_plane()
    graph.draw(plane, surface)
    assert dot.position == pytest.approx((2.0, 1.0))
    sx, sy = screen(plane, dot)
    drag(plane, (sx, sy), (sx - 40, sy), (sx - 80, sy))
    frame(plane, surface, graph)
    assert dot.t is not None
    assert dot.t < 2.0
    assert dot.position == pytest.approx((dot.t, dot.t**2 / 4))
    assert isinstance(f, PlotItem)
    f.expression = "x^3 / 8"  # the point follows the changed curve
    frame(plane, surface, graph)
    assert dot.position == pytest.approx((dot.t, dot.t**3 / 8))


def test_a_point_on_a_plot_without_a_parameter_starts_at_zero() -> None:
    graph = pgp.Graph()
    graph.add("f(x) = cos(x)")
    dot = graph.draggable(curve="f")
    assert dot.position is None
    assert math.isnan(dot.x)
    plane, surface = make_plane()
    graph.draw(plane, surface)
    assert dot.position == pytest.approx((0.0, 1.0))
    assert dot.t == pytest.approx(0.0)


def test_a_point_on_a_plot_can_go_anywhere_on_screen() -> None:
    graph = pgp.Graph()
    graph.add("f(x) = 0x")
    dot = graph.draggable(curve="f", t=0.0)
    plane, surface = make_plane()
    graph.draw(plane, surface)
    edge = plane.rect.x + plane.rect.width - 2
    right = plane.to_world(edge, 0)[0]
    sx, sy = screen(plane, dot)
    drag(plane, (sx, sy), (edge, sy))
    frame(plane, surface, graph)
    assert dot.x == pytest.approx(right, abs=0.05)


def test_a_point_on_a_named_parametric_or_polar_curve() -> None:
    graph = pgp.Graph()
    graph.add("C = (2cos(t), 2sin(t))")
    graph.polar("1 + 0θ", name="R")
    on_c = graph.draggable(curve="C", t=math.pi / 2)
    on_r = graph.draggable(curve="R")
    plane, surface = make_plane()
    graph.draw(plane, surface)
    assert on_c.position == pytest.approx((0.0, 2.0), abs=1e-12)
    assert on_r.position == pytest.approx((1.0, 0.0))  # the start of θ's range
    sx, sy = screen(plane, on_c)
    drag(plane, (sx, sy), (sx - 30, sy + 10), (sx - 60, sy + 30))
    frame(plane, surface, graph)
    assert math.hypot(on_c.x, on_c.y) == pytest.approx(2.0)
    assert on_c.t == pytest.approx(math.atan2(on_c.y, on_c.x))


def test_a_point_on_its_own_curve() -> None:
    graph = pgp.Graph()
    graph.add("k = 3")
    dot = graph.draggable(curve=("t", "k t^2", (-1, 1)), t=0.5)
    plane, surface = make_plane()
    graph.draw(plane, surface)
    assert dot.position == pytest.approx((0.5, 0.75))
    dot.t = -1
    frame(plane, surface, graph)
    assert dot.position == pytest.approx((-1.0, 3.0))
    graph["k"].value = 1  # type: ignore[attr-defined]
    frame(plane, surface, graph)
    assert dot.position == pytest.approx((-1.0, 1.0))
    with pytest.raises(ValueError, match="only points on a curve"):
        graph.draggable((0, 0)).t = 1.0


def test_curve_errors() -> None:
    graph = pgp.Graph()
    graph.add("a = 1")
    unknown = graph.draggable(curve="nope")
    wrong = graph.draggable(curve="a")
    plane, surface = make_plane()
    with pytest.warns(pgp.GraphWarning):
        graph.draw(plane, surface)
    assert unknown.error == "unknown curve 'nope'"
    assert wrong.error is not None
    assert "a draggable point can follow" in wrong.error
    assert plane.draggables == ()


def test_points_and_curves_cannot_mix_constraints() -> None:
    graph = pgp.Graph()
    with pytest.raises(ValueError, match="cannot also be locked"):
        graph.draggable(curve="f", lock="x")
    with pytest.raises(ValueError, match="bounded by the curve"):
        graph.draggable(curve="f", x_range=(0, 1))
    with pytest.raises(ValueError, match="needs a position"):
        graph.draggable()


def test_a_graph_on_two_planes_keeps_both_in_step() -> None:
    graph = pgp.Graph()
    point = graph.draggable((0, 0), name="P")
    first, first_surface = make_plane()
    second, second_surface = make_plane()
    graph.draw(first, first_surface)
    graph.draw(second, second_surface)
    sx, sy = screen(first, point)
    drag(first, (sx, sy), (sx + 40, sy - 20))
    frame(first, first_surface, graph)
    frame(second, second_surface, graph)
    assert first.draggables[0].position == second.draggables[0].position
    assert point.position == first.draggables[0].position


def test_hidden_or_disabled_points() -> None:
    graph = pgp.Graph()
    point = graph.draggable((0, 0))
    plane, surface = make_plane()
    graph.draw(plane, surface)
    point.visible = False
    frame(plane, surface, graph)
    _, blank = make_plane()
    assert pixels(surface) == pixels(blank)
    assert not grabs(plane, screen(plane, point))
    point.visible = True
    point.enabled = False
    frame(plane, surface, graph)
    assert pixels(surface) != pixels(blank)  # drawn...
    assert not grabs(plane, screen(plane, point))  # ...but not draggable
    point.enabled = True
    frame(plane, surface, graph)
    assert grabs(plane, screen(plane, point))


def test_removing_or_detaching_takes_the_points_off_the_plane() -> None:
    graph = pgp.Graph()
    point = graph.draggable((0, 0))
    other = graph.draggable((1, 1))
    plane, surface = make_plane()
    graph.draw(plane, surface)
    assert len(plane.draggables) == 2
    graph.remove(point)
    assert len(plane.draggables) == 1
    graph.detach(plane)
    assert plane.draggables == ()
    graph.draw(plane, surface)  # drawn again: made again
    assert plane.draggables[0].position == other.position


def test_points_follow_the_theme_and_palette() -> None:
    graph = pgp.Graph()
    graph.add("y = x")
    graph.draggable((1, 1), color=None)
    plane, surface = make_plane(theme=pgp.DARK)
    graph.draw(plane, surface)
    assert plane.draggables[0].color == pgp.DARK.palette[1]
    graph.items[1].color = "red"  # type: ignore[attr-defined]
    graph.draw(plane, surface)
    assert plane.draggables[0].color == "red"


def test_a_point_drives_a_tangent_line() -> None:
    """The classic: a point on a curve and the tangent through it."""
    graph = pgp.Graph()
    graph.add("f(x) = sin(x)")
    graph.draggable(curve="f", t=1.0, name="P")
    graph.add("y = f'(P.x)(x - P.x) + P.y")
    plane, surface = make_plane()
    graph.draw(plane, surface)
    tangent = graph.function("f")
    assert graph.errors == {}
    slope = graph.evaluate("f'(P.x)")
    assert slope == pytest.approx(np.cos(1.0))
    assert tangent(1.0) == pytest.approx(np.sin(1.0))


def test_a_graph_does_not_keep_planes_alive() -> None:
    """A plane that is no longer used is freed, though the graph lives on."""
    graph = pgp.Graph()
    graph.draggable((0, 0))
    plane, surface = make_plane()
    graph.draw(plane, surface)
    alive = weakref.ref(plane)
    del plane
    gc.collect()
    assert alive() is None
    graph.draw(make_plane()[0], surface)  # and it still draws on a new one
