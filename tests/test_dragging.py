"""End-to-end tests of draggable points on a :class:`pygridplot.Grid`.

Mouse events are synthesised, as a user's drag would produce them, and fed
through :meth:`Grid.handle_event`; drawing goes to headless surfaces.
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


def make_grid(**kwargs: object) -> tuple[pgp.Grid, pygame.Surface]:
    """Return a 400x300 plane (instant easing) and a surface it has been drawn on."""
    kwargs.setdefault("interaction", pgp.InteractionConfig(smoothing=0))
    grid = pgp.Grid(pygame.Rect(0, 0, *SIZE), **kwargs)  # type: ignore[arg-type]
    surface = pygame.Surface(SIZE)
    grid.draw(surface)
    return grid, surface


def drag(
    grid: pgp.Grid, start: tuple[float, float], *moves: tuple[float, float]
) -> bool:
    """Press at ``start``, move through ``moves`` and release.

    Returns:
        Whether the press was used.
    """
    used = grid.handle_event(mouse(pygame.MOUSEBUTTONDOWN, start))
    for move in moves:
        grid.handle_event(mouse(pygame.MOUSEMOTION, move))
    grid.handle_event(mouse(pygame.MOUSEBUTTONUP, moves[-1] if moves else start))
    return used


# --- Grabbing and dragging -----------------------------------------------------


def test_pressing_on_a_point_drags_it_instead_of_panning() -> None:
    grid, _ = make_grid()
    point = grid.draggable((1.0, 1.0))
    sx, sy = point.screen_position
    assert drag(grid, (sx + 3, sy - 2), (sx + 53, sy - 52))
    assert point.position == pytest.approx((2.0, 2.0))
    assert grid.view.center == (0.0, 0.0)  # the plane did not pan
    assert grid.dragged is None


def test_pressing_elsewhere_pans() -> None:
    grid, _ = make_grid()
    point = grid.draggable((1.0, 1.0))
    drag(grid, (100, 100), (150, 100))
    assert grid.view.center == pytest.approx((-1.0, 0.0))
    assert point.position == (1.0, 1.0)


def test_grabbing_off_centre_does_not_jump() -> None:
    grid, _ = make_grid()
    point = grid.draggable((0.0, 0.0))
    sx, sy = point.screen_position
    grid.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx + 5, sy + 4)))
    grid.handle_event(mouse(pygame.MOUSEMOTION, (sx + 5, sy + 4)))
    assert point.position == (0.0, 0.0)


def test_the_nearest_point_wins_and_ties_go_to_the_newest() -> None:
    grid, _ = make_grid()
    first = grid.draggable((0.0, 0.0))
    second = grid.draggable((0.0, 0.0))
    near = grid.draggable((0.2, 0.0))  # 10 px to the right
    sx, sy = first.screen_position
    grid.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx + 2, sy)))
    assert grid.dragged is second
    grid.handle_event(mouse(pygame.MOUSEBUTTONUP, (sx + 2, sy)))
    grid.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx + 8, sy)))
    assert grid.dragged is near


def test_points_are_grabbed_within_the_margin_only() -> None:
    grid, _ = make_grid()
    point = grid.draggable((0.0, 0.0))  # radius 6 + outline 2 + margin 6
    sx, sy = point.screen_position
    grid.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx + 13, sy)))
    assert grid.dragged is point
    grid.handle_event(mouse(pygame.MOUSEBUTTONUP, (sx + 13, sy)))
    grid.interaction = pgp.InteractionConfig(smoothing=0, grab_margin_px=0)
    grid.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx + 13, sy)))
    assert grid.dragged is None  # this press pans instead


def test_disabled_points_are_not_grabbed() -> None:
    grid, _ = make_grid()
    point = grid.draggable((0.0, 0.0), enabled=False)
    sx, sy = point.screen_position
    drag(grid, (sx, sy), (sx + 50, sy))
    assert point.position == (0.0, 0.0)
    assert grid.view.center != (0.0, 0.0)  # it panned instead


def test_points_drag_on_a_grid_without_pan_and_zoom() -> None:
    grid, _ = make_grid(interactive=False)
    point = grid.draggable((0.0, 0.0), lock="y")
    sx, sy = point.screen_position
    drag(grid, (sx, sy), (sx + 100, sy + 80))
    assert point.position == pytest.approx((2.0, 0.0))
    assert not drag(grid, (20, 20), (60, 60))  # elsewhere: nothing happens
    assert grid.view.center == (0.0, 0.0)


def test_drags_stay_inside_the_plane() -> None:
    grid, _ = make_grid()
    point = grid.draggable((0.0, 0.0))
    sx, sy = point.screen_position
    drag(grid, (sx, sy), (5000, -5000))
    px, py = point.screen_position
    assert 0 <= px <= SIZE[0] - 1
    assert 0 <= py <= SIZE[1] - 1


def test_leaving_the_window_ends_a_drag() -> None:
    grid, _ = make_grid()
    point = grid.draggable((0.0, 0.0))
    sx, sy = point.screen_position
    grid.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx, sy)))
    assert point.dragging
    grid.handle_event(pygame.event.Event(pygame.WINDOWLEAVE))
    assert not point.dragging
    assert grid.dragged is None


def test_zooming_mid_drag_keeps_the_point_under_the_cursor() -> None:
    grid, surface = make_grid()
    point = grid.draggable((1.0, 1.0))
    sx, sy = point.screen_position
    grid.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx, sy)))
    grid.handle_event(mouse(pygame.MOUSEMOTION, (sx + 40, sy + 10)))
    grid.handle_event(wheel(3))
    grid.draw(surface)  # applies the zoom and re-drags the point
    assert grid.view.scale > 50
    assert point.screen_position == pytest.approx((round(sx + 40), round(sy + 10)))


def test_points_on_curves_follow_the_cursor_round_the_curve() -> None:
    grid, _ = make_grid()
    point = grid.draggable(curve=(np.cos, np.sin, (0, 2 * math.pi)))
    sx, sy = point.screen_position
    cx, cy = grid.to_screen(0.0, 0.0)
    moves = [
        (cx + 80 * math.cos(a), cy - 80 * math.sin(a)) for a in np.linspace(0, 6, 40)
    ]
    drag(grid, (sx, sy), *moves)
    assert math.hypot(point.x, point.y) == pytest.approx(1.0)
    assert point.t == pytest.approx(6.0, abs=0.02)


def test_removed_points_are_forgotten_even_mid_drag() -> None:
    grid, surface = make_grid()
    point = grid.draggable((0.0, 0.0))
    other = grid.draggable((2.0, 0.0))
    sx, sy = point.screen_position
    grid.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (sx, sy)))
    point.remove()
    assert grid.draggables == (other,)
    assert grid.dragged is None
    grid.handle_event(mouse(pygame.MOUSEMOTION, (sx + 30, sy)))
    grid.draw(surface)
    assert point.position == (0.0, 0.0)
    with pytest.raises(ValueError, match="not a draggable point"):
        grid.draw_controls(surface, point)


def test_hovering_sets_and_restores_the_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(pygame.mouse, "get_cursor", lambda: "arrow")
    monkeypatch.setattr(pygame.mouse, "set_cursor", calls.append)
    grid, _ = make_grid()
    point = grid.draggable((0.0, 0.0))
    sx, sy = point.screen_position
    grid.handle_event(mouse(pygame.MOUSEMOTION, (sx + 1, sy)))
    assert point.hovered
    assert calls == [pygame.SYSTEM_CURSOR_HAND]
    grid.handle_event(mouse(pygame.MOUSEMOTION, (sx + 60, sy)))
    assert not point.hovered
    assert calls == [pygame.SYSTEM_CURSOR_HAND, "arrow"]


def test_grids_share_the_hand_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
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
    left = pgp.Grid(pygame.Rect(0, 0, 200, 300))
    right = pgp.Grid(pygame.Rect(200, 0, 200, 300))
    surface = pygame.Surface(SIZE)
    left.draw(surface)
    right.draw(surface)
    a = left.draggable((0.0, 0.0))
    b = right.draggable((0.0, 0.0))
    for pos in (a.screen_position, b.screen_position, (10.0, 10.0)):
        event = mouse(pygame.MOUSEMOTION, pos)
        for grid in (right, left):  # the "wrong" order for the move to b
            grid.handle_event(event)
    assert calls == [pygame.SYSTEM_CURSOR_HAND, "arrow"]
    assert current[0] == "arrow"


def test_moved_is_stable_through_a_frame() -> None:
    grid, surface = make_grid()
    point = grid.draggable((0.0, 0.0))
    sx, sy = point.screen_position
    drag(grid, (sx, sy), (sx + 20, sy))
    grid.draw(surface)
    assert point.moved
    grid.draw(surface)
    assert not point.moved


# --- Drawing -------------------------------------------------------------------


def ink(surface: pygame.Surface) -> np.ndarray:
    """Return how far each pixel is from the background, shape (h, w)."""
    rgb = pygame.surfarray.array3d(surface).astype(float)
    return (255.0 - rgb.min(axis=2)).T / 255.0


def blank_grid() -> tuple[pgp.Grid, pygame.Surface]:
    """A plane with an empty white grid."""
    theme = pgp.GridTheme(
        show_minor=False, show_major=False, show_axes=False, show_labels=False
    )
    return make_grid(theme=theme)


def test_points_are_drawn_in_palette_colours_with_an_outline() -> None:
    grid, surface = blank_grid()
    first = grid.draggable((0.0, 0.0))
    second = grid.draggable((2.0, 0.0), coordinates="never")
    grid.draw_controls(surface)
    for index, point in enumerate((first, second)):
        sx, sy = (round(v) for v in point.screen_position)
        assert tuple(surface.get_at((sx, sy)))[:3] == grid.theme.palette[index]
        assert tuple(surface.get_at((sx + 7, sy)))[:3] == (255, 255, 255)  # outline
    assert ink(surface)[:, : SIZE[0] // 2 - 12].max() == 0.0  # nothing else


def test_hovered_points_get_a_ring_and_their_coordinates() -> None:
    grid, surface = blank_grid()
    point = grid.draggable((0.5, 0.5))
    sx, sy = point.screen_position
    grid.draw_controls(surface)
    quiet = ink(surface)
    grid.handle_event(mouse(pygame.MOUSEMOTION, (sx, sy)))
    grid.draw(surface)
    grid.draw_controls(surface)
    lit = ink(surface)
    x, y = round(sx), round(sy)
    assert quiet[y, x + 11] == 0.0
    assert 0.05 < lit[y, x + 11] < 0.6  # the translucent ring
    label = lit[y - 30 : y - 8, x + 10 : x + 80]  # above right: "(0.5, 0.5)"
    assert label.max() > 0.5
    assert quiet[y - 30 : y - 8, x + 10 : x + 80].max() == 0.0


def test_coordinates_can_always_show() -> None:
    grid, surface = blank_grid()
    point = grid.draggable((0.5, 0.5), coordinates="always")
    grid.draw_controls(surface)
    x, y = (round(v) for v in point.screen_position)
    assert ink(surface)[y - 30 : y - 8, x + 10 : x + 80].max() > 0.5


def test_off_screen_points_show_no_coordinates() -> None:
    grid, surface = blank_grid()
    grid.draggable((100.0, 0.0), coordinates="always")  # far to the right
    grid.draw_controls(surface)
    assert ink(surface).max() == 0.0


def test_a_point_does_not_keep_its_grid_alive() -> None:
    grid, _ = make_grid()
    point = grid.draggable((1.0, 1.0))
    alive = weakref.ref(grid)
    del grid
    gc.collect()
    assert alive() is None
    point.remove()  # harmless once the plane is gone
