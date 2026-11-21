"""Helpers for the object tests: headless planes, synthetic drags, pixel checks."""

from __future__ import annotations

from collections.abc import Iterator

import pygame
import pytest

import pygridplot as pgp
from pygridplot.dragging import HAND_CURSOR

SIZE = (400, 300)


@pytest.fixture(autouse=True, scope="module")
def display() -> Iterator[None]:
    """Initialise pygame (dummy drivers) once for each module of object tests."""
    pygame.init()
    pygame.display.set_mode((64, 48))
    yield
    pygame.quit()


@pytest.fixture(autouse=True)
def fresh_cursor() -> None:
    """Start every test with no plane asking for the hand cursor."""
    HAND_CURSOR.reset()


def make_plane(**kwargs: object) -> tuple[pgp.Plane, pygame.Surface]:
    """Return a 400x300 plane (instant easing) and a surface it has drawn on."""
    kwargs.setdefault("interaction", pgp.InteractionConfig(smoothing=0))
    plane = pgp.Plane(pygame.Rect(0, 0, *SIZE), **kwargs)  # type: ignore[arg-type]
    surface = pygame.Surface(SIZE)
    plane.draw_grid(surface)
    return plane, surface


def mouse(kind: int, pos: tuple[float, float], button: int = 1) -> pygame.event.Event:
    """Build a mouse button or motion event at a (rounded) position."""
    at = (round(pos[0]), round(pos[1]))
    if kind == pygame.MOUSEMOTION:
        return pygame.event.Event(kind, pos=at, rel=(0, 0), buttons=(1, 0, 0))
    return pygame.event.Event(kind, pos=at, button=button)


def drag(
    plane: pgp.Plane, start: tuple[float, float], *moves: tuple[float, float]
) -> bool:
    """Press at ``start``, move through ``moves`` and release.

    Returns:
        Whether the press was used (by a point).
    """
    used = plane.handle_event(mouse(pygame.MOUSEBUTTONDOWN, start))
    for move in moves:
        plane.handle_event(mouse(pygame.MOUSEMOTION, move))
    plane.handle_event(mouse(pygame.MOUSEBUTTONUP, moves[-1] if moves else start))
    return used


def pixels(surface: pygame.Surface) -> bytes:
    """Return a surface's pixels, to compare drawings exactly."""
    return pygame.image.tobytes(surface, "RGB")
