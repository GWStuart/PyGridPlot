"""End-to-end tests of draggable points on a :class:`pygridplot.Plane`.

Mouse events are synthesised, as a user's drag would produce them, and fed
through :meth:`Plane.handle_event`; drawing goes to headless surfaces.
"""

from __future__ import annotations

import gc
import math
import weakref

import numpy as np
import pygame
import pytest

import pygridplot as pgp
from pygridplot.dragging import HAND_CURSOR

pytestmark = pytest.mark.usefixtures("pygame_display")


@pytest.fixture(autouse=True)
def fresh_cursor() -> None:
    """Start every test with no plane asking for the hand cursor.

    Planes of earlier tests may still be waiting to be garbage-collected
    with a point hovered.
    """
    HAND_CURSOR.reset()


SIZE = (400, 300)


def mouse(kind: int, pos: tuple[float, float], button: int = 1) -> pygame.event.Event:
    """Build a mouse button or motion event at a (rounded) position."""
    at = (round(pos[0]), round(pos[1]))
    if kind == pygame.MOUSEMOTION:
        return pygame.event.Event(kind, pos=at, rel=(0, 0), buttons=(1, 0, 0))
    return pygame.event.Event(kind, pos=at, button=button)


def wheel(y: float) -> pygame.event.Event:
    """Build a wheel event scrolling ``y`` notches."""
    return pygame.event.Event(
        pygame.MOUSEWHEEL, x=0, y=int(y), precise_x=0.0, precise_y=y, flipped=False
    )


def make_plane(**kwargs: object) -> tuple[pgp.Plane, pygame.Surface]:
    """Return a 400x300 plane (instant easing) and a surface it has been drawn on."""
    kwargs.setdefault("interaction", pgp.InteractionConfig(smoothing=0))
    plane = pgp.Plane(pygame.Rect(0, 0, *SIZE), **kwargs)  # type: ignore[arg-type]
    surface = pygame.Surface(SIZE)
    plane.draw_grid(surface)
    return plane, surface


def drag(
    plane: pgp.Plane, start: tuple[float, float], *moves: tuple[float, float]
) -> bool:
    """Press at ``start``, move through ``moves`` and release.

    Returns:
        Whether the press was used.
    """
    used = plane.handle_event(mouse(pygame.MOUSEBUTTONDOWN, start))
    for move in moves:
        plane.handle_event(mouse(pygame.MOUSEMOTION, move))
    plane.handle_event(mouse(pygame.MOUSEBUTTONUP, moves[-1] if moves else start))
    return used


# --- Grabbing and dragging -----------------------------------------------------


def test_pressing_on_a_point_drags_it_instead_of_panning() -> None:
    plane, _ = make_plane()
    point = plane.draggable((1.0, 1.0))
    sx, sy = point.screen_position
    assert drag(plane, (sx + 3, sy - 2), (sx + 53, sy - 52))
    assert point.position == pytest.approx((2.0, 2.0))
    assert plane.view.center == (0.0, 0.0)  # the plane did not pan
    assert plane.dragged is None


def test_pressing_elsewhere_pans() -> None:
    plane, _ = make_plane()
    point = plane.draggable((1.0, 1.0))
    drag(plane, (100, 100), (150, 100))
    assert plane.view.center == pytest.approx((-1.0, 0.0))
    assert point.position == (1.0, 1.0)


def test_grabbing_off_centre_does_not_jump() -> None:
    plane, _ = make_plane()
    point = plane.draggable((0.0, 0.0))
    sx, sy = point.screen_position
    plane.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx + 5, sy + 4)))
    plane.handle_event(mouse(pygame.MOUSEMOTION, (sx + 5, sy + 4)))
    assert point.position == (0.0, 0.0)


def test_the_nearest_point_wins_and_ties_go_to_the_newest() -> None:
    plane, _ = make_plane()
    first = plane.draggable((0.0, 0.0))
    second = plane.draggable((0.0, 0.0))
    near = plane.draggable((0.2, 0.0))  # 10 px to the right
    sx, sy = first.screen_position
    plane.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx + 2, sy)))
    assert plane.dragged is second
    plane.handle_event(mouse(pygame.MOUSEBUTTONUP, (sx + 2, sy)))
    plane.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx + 8, sy)))
    assert plane.dragged is near


def test_points_are_grabbed_within_the_margin_only() -> None:
    plane, _ = make_plane()
    point = plane.draggable((0.0, 0.0))  # radius 6 + outline 2 + margin 6
    sx, sy = point.screen_position
    plane.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx + 13, sy)))
    assert plane.dragged is point
    plane.handle_event(mouse(pygame.MOUSEBUTTONUP, (sx + 13, sy)))
    plane.interaction = pgp.InteractionConfig(smoothing=0, grab_margin_px=0)
    plane.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx + 13, sy)))
    assert plane.dragged is None  # this press pans instead


def test_disabled_points_are_not_grabbed() -> None:
    plane, _ = make_plane()
    point = plane.draggable((0.0, 0.0), enabled=False)
    sx, sy = point.screen_position
    drag(plane, (sx, sy), (sx + 50, sy))
    assert point.position == (0.0, 0.0)
    assert plane.view.center != (0.0, 0.0)  # it panned instead


def test_points_drag_on_a_plane_without_pan_and_zoom() -> None:
    plane, _ = make_plane(interactive=False)
    point = plane.draggable((0.0, 0.0), lock="y")
    sx, sy = point.screen_position
    drag(plane, (sx, sy), (sx + 100, sy + 80))
    assert point.position == pytest.approx((2.0, 0.0))
    assert not drag(plane, (20, 20), (60, 60))  # elsewhere: nothing happens
    assert plane.view.center == (0.0, 0.0)


def test_drags_stay_inside_the_plane() -> None:
    plane, _ = make_plane()
    point = plane.draggable((0.0, 0.0))
    sx, sy = point.screen_position
    drag(plane, (sx, sy), (5000, -5000))
    px, py = point.screen_position
    assert 0 <= px <= SIZE[0] - 1
    assert 0 <= py <= SIZE[1] - 1


def test_leaving_the_window_ends_a_drag() -> None:
    plane, _ = make_plane()
    point = plane.draggable((0.0, 0.0))
    sx, sy = point.screen_position
    plane.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx, sy)))
    assert point.dragging
    plane.handle_event(pygame.event.Event(pygame.WINDOWLEAVE))
    assert not point.dragging
    assert plane.dragged is None


def test_zooming_mid_drag_keeps_the_point_under_the_cursor() -> None:
    plane, surface = make_plane()
    point = plane.draggable((1.0, 1.0))
    sx, sy = point.screen_position
    plane.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx, sy)))
    plane.handle_event(mouse(pygame.MOUSEMOTION, (sx + 40, sy + 10)))
    plane.handle_event(wheel(3))
    plane.draw_grid(surface)  # applies the zoom and re-drags the point
    assert plane.view.scale > 50
    assert point.screen_position == pytest.approx((round(sx + 40), round(sy + 10)))


def test_points_on_curves_follow_the_cursor_round_the_curve() -> None:
    plane, _ = make_plane()
    point = plane.draggable(curve=(np.cos, np.sin, (0, 2 * math.pi)))
    sx, sy = point.screen_position
    cx, cy = plane.to_screen(0.0, 0.0)
    moves = [
        (cx + 80 * math.cos(a), cy - 80 * math.sin(a)) for a in np.linspace(0, 6, 40)
    ]
    drag(plane, (sx, sy), *moves)
    assert math.hypot(point.x, point.y) == pytest.approx(1.0)
    assert point.t == pytest.approx(6.0, abs=0.02)


def test_removed_points_are_forgotten_even_mid_drag() -> None:
    plane, surface = make_plane()
    point = plane.draggable((0.0, 0.0))
    other = plane.draggable((2.0, 0.0))
    sx, sy = point.screen_position
    plane.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx, sy)))
    point.remove()
    assert plane.draggables == (other,)
    assert plane.dragged is None
    plane.handle_event(mouse(pygame.MOUSEMOTION, (sx + 30, sy)))
    plane.draw_grid(surface)
    assert point.position == (0.0, 0.0)
    with pytest.raises(ValueError, match="not a draggable point"):
        plane.draw_draggables(surface, point)


def test_hovering_sets_and_restores_the_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(pygame.mouse, "get_cursor", lambda: "arrow")
    monkeypatch.setattr(pygame.mouse, "set_cursor", calls.append)
    plane, _ = make_plane()
    point = plane.draggable((0.0, 0.0))
    sx, sy = point.screen_position
    plane.handle_event(mouse(pygame.MOUSEMOTION, (sx + 1, sy)))
    assert point.hovered
    assert calls == [pygame.SYSTEM_CURSOR_HAND]
    plane.handle_event(mouse(pygame.MOUSEMOTION, (sx + 60, sy)))
    assert not point.hovered
    assert calls == [pygame.SYSTEM_CURSOR_HAND, "arrow"]


def test_planes_share_the_hand_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two planes in one window agree on the cursor.

    Moving from a point on one plane to a point on another keeps the hand,
    and leaving both restores the original cursor, whatever the order in
    which the planes see the events.
    """
    calls: list[object] = []
    current: list[object] = ["arrow"]

    def set_cursor(cursor: object) -> None:
        calls.append(cursor)
        current[0] = cursor

    monkeypatch.setattr(pygame.mouse, "get_cursor", lambda: current[0])
    monkeypatch.setattr(pygame.mouse, "set_cursor", set_cursor)
    left = pgp.Plane(pygame.Rect(0, 0, 200, 300))
    right = pgp.Plane(pygame.Rect(200, 0, 200, 300))
    surface = pygame.Surface(SIZE)
    left.draw_grid(surface)
    right.draw_grid(surface)
    a = left.draggable((0.0, 0.0))
    b = right.draggable((0.0, 0.0))
    for pos in (a.screen_position, b.screen_position, (10.0, 10.0)):
        event = mouse(pygame.MOUSEMOTION, pos)
        for plane in (right, left):  # the "wrong" order for the move to b
            plane.handle_event(event)
    assert calls == [pygame.SYSTEM_CURSOR_HAND, "arrow"]
    assert current[0] == "arrow"


def test_moved_is_stable_through_a_frame() -> None:
    plane, surface = make_plane()
    point = plane.draggable((0.0, 0.0))
    sx, sy = point.screen_position
    drag(plane, (sx, sy), (sx + 20, sy))
    plane.draw_grid(surface)
    assert point.moved
    plane.draw_grid(surface)
    assert not point.moved


# --- Drawing -------------------------------------------------------------------


def ink(surface: pygame.Surface) -> np.ndarray:
    """Return how far each pixel is from the background, shape (h, w)."""
    rgb = pygame.surfarray.array3d(surface).astype(float)
    return (255.0 - rgb.min(axis=2)).T / 255.0


def blank_plane() -> tuple[pgp.Plane, pygame.Surface]:
    """A plane with an empty white grid."""
    theme = pgp.GridTheme(
        show_minor=False, show_major=False, show_axes=False, show_labels=False
    )
    return make_plane(theme=theme)


def test_points_are_drawn_in_palette_colours_with_an_outline() -> None:
    plane, surface = blank_plane()
    first = plane.draggable((0.0, 0.0))
    second = plane.draggable((2.0, 0.0), coordinates="never")
    plane.draw_draggables(surface)
    for index, point in enumerate((first, second)):
        sx, sy = (round(v) for v in point.screen_position)
        assert tuple(surface.get_at((sx, sy)))[:3] == plane.theme.palette[index]
        assert tuple(surface.get_at((sx + 7, sy)))[:3] == (255, 255, 255)  # outline
    assert ink(surface)[:, : SIZE[0] // 2 - 12].max() == 0.0  # nothing else


def test_hovered_points_get_a_ring_and_their_coordinates() -> None:
    plane, surface = blank_plane()
    point = plane.draggable((0.5, 0.5))
    sx, sy = point.screen_position
    plane.draw_draggables(surface)
    quiet = ink(surface)
    plane.handle_event(mouse(pygame.MOUSEMOTION, (sx, sy)))
    plane.draw_grid(surface)
    plane.draw_draggables(surface)
    lit = ink(surface)
    x, y = round(sx), round(sy)
    assert quiet[y, x + 11] == 0.0
    assert 0.05 < lit[y, x + 11] < 0.6  # the translucent ring
    label = lit[y - 30 : y - 8, x + 10 : x + 80]  # above right: "(0.5, 0.5)"
    assert label.max() > 0.5
    assert quiet[y - 30 : y - 8, x + 10 : x + 80].max() == 0.0


def test_coordinates_can_always_show() -> None:
    plane, surface = blank_plane()
    point = plane.draggable((0.5, 0.5), coordinates="always")
    plane.draw_draggables(surface)
    x, y = (round(v) for v in point.screen_position)
    assert ink(surface)[y - 30 : y - 8, x + 10 : x + 80].max() > 0.5


def test_off_screen_points_show_no_coordinates() -> None:
    plane, surface = blank_plane()
    plane.draggable((100.0, 0.0), coordinates="always")  # far to the right
    plane.draw_draggables(surface)
    assert ink(surface).max() == 0.0


def test_a_point_does_not_keep_its_plane_alive() -> None:
    plane, _ = make_plane()
    point = plane.draggable((1.0, 1.0))
    alive = weakref.ref(plane)
    del plane
    gc.collect()
    assert alive() is None
    point.remove()  # harmless once the plane is gone
