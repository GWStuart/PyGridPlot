"""Tests for the public :class:`pygridplot.Plane` API."""

from __future__ import annotations

from dataclasses import replace

import pygame
import pytest

import pygridplot as pgp

pytestmark = pytest.mark.usefixtures("pygame_display")


def test_auto_rect_follows_the_surface() -> None:
    plane = pgp.Plane()
    small = pygame.Surface((200, 100))
    plane.draw_grid(small)
    assert plane.rect.size == (200, 100)
    big = pygame.Surface((640, 480))
    plane.draw_grid(big)
    assert plane.rect.size == (640, 480)
    assert plane.view.center == (0.0, 0.0)  # resizing keeps the centre...
    assert plane.view.scale == 50.0  # ...and the scale


def test_explicit_rect_clips_drawing() -> None:
    surface = pygame.Surface((300, 200))
    surface.fill((255, 0, 0))
    plane = pgp.Plane(pygame.Rect(50, 40, 100, 80))
    plane.draw_grid(surface)
    assert tuple(surface.get_at((10, 10)))[:3] == (255, 0, 0)
    assert tuple(surface.get_at((60, 50)))[:3] != (255, 0, 0)
    assert surface.get_clip() == surface.get_rect()  # clip area restored


def test_coordinate_helpers() -> None:
    plane = pgp.Plane(pygame.Rect(0, 0, 400, 300), center=(1.0, 2.0), scale=10.0)
    assert plane.to_screen(1.0, 2.0) == (200.0, 150.0)
    assert plane.to_world(210.0, 140.0) == pytest.approx((2.0, 3.0))


def test_set_view_reset_view_and_home() -> None:
    plane = pgp.Plane(pygame.Rect(0, 0, 400, 300))
    plane.set_view(x_range=(10.0, 20.0))
    assert plane.view.center_x == 15.0
    plane.reset_view()
    assert plane.view.center == (0.0, 0.0)
    assert plane.view.scale == 50.0
    plane.view.pan_world(3.0, 4.0)
    plane.set_home()
    plane.view.pan_world(10.0, 10.0)
    plane.reset_view()
    assert plane.view.center == (3.0, 4.0)


def test_interactive_flag_disables_event_handling() -> None:
    plane = pgp.Plane(pygame.Rect(0, 0, 400, 300))
    down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(100, 100), button=1)
    plane.interactive = False
    assert not plane.handle_event(down)
    plane.interactive = True
    assert plane.handle_event(down)


def test_theme_and_last_grid() -> None:
    plane = pgp.Plane(theme=pgp.DARK)
    assert plane.last_grid is None
    surface = pygame.Surface((320, 240))
    plane.draw_grid(surface)
    assert plane.last_grid is not None
    assert tuple(surface.get_at((1, 1)))[:3] == pgp.DARK.background
    plane.theme = replace(pgp.LIGHT, background=(1, 2, 3))
    plane.draw_grid(surface)
    assert tuple(surface.get_at((1, 1)))[:3] == (1, 2, 3)


def test_unequal_aspect_plane() -> None:
    plane = pgp.Plane(scale=(10.0, 100.0), equal_aspect=False)
    plane.draw_grid(pygame.Surface((320, 240)))
    grid = plane.last_grid
    assert grid is not None
    assert grid.x_spacing.dominant != grid.y_spacing.dominant


def test_background_none_draws_over_existing_content() -> None:
    surface = pygame.Surface((320, 240))
    surface.fill((10, 200, 10))
    plane = pgp.Plane(theme=replace(pgp.LIGHT, background=None))
    plane.draw_grid(surface)
    assert tuple(surface.get_at((3, 3)))[:3] == (10, 200, 10)


def test_show_runs_until_quit() -> None:
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    pgp.show(size=(320, 240), theme=pgp.DARK)  # returns once QUIT is seen
    assert pygame.get_init()  # pygame was initialised by the caller: left running


# ----------------------------------------------------------------------
# Initial ranges and logarithmic axes
# ----------------------------------------------------------------------
def visible(plane: pgp.Plane) -> tuple[float, float, float, float]:
    """Return the plane's visible world bounds."""
    b = plane.view.bounds
    return (b.x_min, b.x_max, b.y_min, b.y_max)


def test_ranges_fill_each_axis_exactly() -> None:
    plane = pgp.Plane(pygame.Rect(0, 0, 800, 600), x_range=(-2, 8), y_range=(-1, 3))
    assert not plane.view.equal_aspect  # both ranges given: each fills its axis
    assert plane.view.scale_x == pytest.approx(399.5 / 5)
    assert plane.view.scale_y == pytest.approx(299.5 / 2)
    x_min, x_max, y_min, y_max = visible(plane)
    assert x_min <= -2 < 8 <= x_max
    assert y_min <= -1 < 3 <= y_max
    plane.view.zoom(2.0)
    plane.reset_view()  # the fitted view is the home view
    assert plane.view.scale_x == pytest.approx(399.5 / 5)


def test_one_range_keeps_equal_aspect() -> None:
    plane = pgp.Plane(pygame.Rect(0, 0, 800, 600), x_range=(0, 16))
    assert plane.view.equal_aspect
    assert plane.view.center == (8.0, 0.0)
    assert plane.view.scale == pytest.approx(399.5 / 8)


def test_ranges_wait_for_the_real_size() -> None:
    plane = pgp.Plane(x_range=(0, 10), y_range=(0, 5))
    plane.draw_grid(pygame.Surface((1000, 500)))  # not the display's size
    assert plane.view.scale_x == pytest.approx(499.5 / 5)
    assert plane.view.scale_y == pytest.approx(249.5 / 2.5)
    plane.reset_view()
    assert plane.view.scale_x == pytest.approx(499.5 / 5)


def test_moving_the_view_before_the_first_frame_is_kept() -> None:
    plane = pgp.Plane(x_range=(0, 10), y_range=(0, 5))
    plane.view.center = (100.0, 100.0)
    plane.draw_grid(pygame.Surface((1000, 500)))
    assert plane.view.center == (100.0, 100.0)  # not refitted over the change


def test_invalid_ranges_raise_straight_away() -> None:
    with pytest.raises(ValueError, match="low < high"):
        pgp.Plane(x_range=(5, 1))
    with pytest.raises(ValueError, match="positive"):
        pgp.Plane(x_range=(0, 10), log_x=True)
    plane = pgp.Plane(log_y=True)
    with pytest.raises(ValueError, match="positive"):
        plane.set_view(y_range=(-1, 1))


def test_log_axes_from_the_constructor() -> None:
    plane = pgp.Plane(pygame.Rect(0, 0, 800, 600), log_x=True, log_y=True)
    assert plane.log_x
    assert plane.log_y
    assert plane.view.center == (1.0, 1.0)
    assert plane.view.equal_aspect  # same kind of axis: same scale per decade
    semi = pgp.Plane(pygame.Rect(0, 0, 800, 600), log_y=True)
    assert not semi.view.equal_aspect  # a unit and a decade are not comparable
    ranged = pgp.Plane(
        pygame.Rect(0, 0, 800, 600), x_range=(1, 1e4), y_range=(1e-3, 1), log_x=True
    )
    x_min, x_max, _, _ = visible(ranged)
    assert x_min <= 1.0 < 1e4 <= x_max * (1 + 1e-12)


def test_log_axes_can_be_switched_at_run_time() -> None:
    plane = pgp.Plane(pygame.Rect(0, 0, 800, 600), center=(5.0, 5.0))
    plane.log_x = True
    assert plane.log_x
    assert not plane.view.equal_aspect  # mixed axes
    assert visible(plane)[0] > 0
    plane.view.zoom(3.0)
    plane.reset_view()  # home is the view right after switching
    assert plane.view.scale_y == 50.0
    plane.log_x = False
    assert plane.view.equal_aspect  # back to the automatic choice
    plane.draw_grid(pygame.Surface((800, 600)))


def test_explicit_equal_aspect_is_respected_when_switching() -> None:
    plane = pgp.Plane(pygame.Rect(0, 0, 800, 600), equal_aspect=False)
    plane.log_y = True
    plane.log_y = False
    assert not plane.view.equal_aspect


def test_plots_follow_log_axes() -> None:
    surface = pygame.Surface((400, 300))
    plane = pgp.Plane(pygame.Rect(0, 0, 400, 300), log_x=True, log_y=True)
    plane.draw_grid(surface)
    plane.plot(surface, lambda x: x, color="red")
    plane.points(surface, (1.0, 1.0), color="blue", radius=5)
    assert tuple(surface.get_at((200, 150)))[:3] == (0, 0, 255)  # the point at 1, 1
    assert tuple(surface.get_at((275, 75)))[:3] == (255, 0, 0)  # y = x at 10^0.5
