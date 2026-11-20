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
