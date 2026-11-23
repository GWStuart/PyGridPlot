"""End-to-end tests of sliders on a :class:`pygridplot.Grid`.

Mouse events are synthesised and fed through :meth:`Grid.handle_event`;
drawing goes to headless surfaces.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pygame
import pytest

import pygridplot as pgp
from pygridplot.core.handles.panel import SliderRow
from pygridplot.dragging import HAND_CURSOR
from tests.test_dragging import SIZE, drag, make_grid, mouse, wheel

pytestmark = pytest.mark.usefixtures("pygame_display")


@pytest.fixture(autouse=True)
def fresh_cursor() -> None:
    """Start every test with no plane asking for the hand cursor."""
    HAND_CURSOR.reset()


def frame(grid: pgp.Grid, surface: pygame.Surface, *items: Any) -> None:
    """Draw a frame: the grid and then the sliders (and points)."""
    grid.draw(surface)
    grid.draw_controls(surface, *items)


def row(grid: pgp.Grid, slider: pgp.Slider[Any]) -> SliderRow:
    """Where a slider was last drawn."""
    for panel in grid._panels:
        for found in panel.rows:
            if found.slider is slider:
                return found
    raise AssertionError("the slider was not drawn")


def at(found: SliderRow, fraction: float) -> tuple[float, float]:
    """The screen point a fraction of the way along a slider's track."""
    return (found.start + fraction * (found.end - found.start), found.y)


def test_pressing_the_track_moves_the_handle_there_and_drags() -> None:
    grid, surface = make_grid()
    a = grid.slider("a", (0, 10))
    frame(grid, surface)
    track = row(grid, a)
    center = grid.view.center
    assert drag(grid, at(track, 0.3), at(track, 0.6))
    assert a.value == pytest.approx(6.0, abs=0.05)
    assert grid.view.center == center  # no pan
    assert not a.dragging


def test_grabbing_the_handle_does_not_make_it_jump() -> None:
    grid, surface = make_grid()
    a = grid.slider("a", (0, 10), value=5)
    frame(grid, surface)
    track = row(grid, a)
    x, y = track.handle_x + 4, track.y  # on the handle, a little to the right
    grid.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (x, y)))
    assert a.value == 5
    assert a.dragging
    grid.handle_event(
        mouse(pygame.MOUSEMOTION, (x + (track.end - track.start) / 10, y))
    )
    assert a.value == pytest.approx(6.0, abs=0.05)
    grid.handle_event(mouse(pygame.MOUSEBUTTONUP, (x, y)))


def test_dragging_past_the_ends_stops_at_the_range() -> None:
    grid, surface = make_grid()
    n = grid.slider("n", (1, 20), value=5, step=1)
    frame(grid, surface)
    track = row(grid, n)
    drag(grid, at(track, 0.5), (track.end + 300, track.y))
    assert n.value == 20
    assert isinstance(n.value, int)
    drag(grid, at(track, 0.5), (-100, track.y - 50))  # anywhere, once grabbed
    assert n.value == 1


def test_presses_on_the_panel_never_pan() -> None:
    grid, surface = make_grid()
    a = grid.slider("a", (0, 1), value=0.5)
    frame(grid, surface)
    panel = grid._panels[0].rect
    center = grid.view.center
    corner = (panel.left + 3, panel.top + 3)  # the panel, away from the track
    assert drag(grid, corner, (corner[0] + 60, corner[1] - 60))
    assert grid.view.center == center
    assert a.value == 0.5
    outside = (SIZE[0] - 20, 20)
    drag(grid, outside, (outside[0] - 30, outside[1] + 10))  # elsewhere: pans
    assert grid.view.center != center


def test_the_wheel_moves_a_slider_instead_of_zooming() -> None:
    grid, surface = make_grid()
    n = grid.slider("n", (0, 10), value=5, step=1)
    frame(grid, surface)
    track = row(grid, n)
    scale = grid.view.scale_x
    grid.handle_event(mouse(pygame.MOUSEMOTION, at(track, 0.5)))
    assert grid.handle_event(wheel(2))
    assert n.value == 7
    assert grid.view.scale_x == scale
    panel = grid._panels[0].rect
    grid.handle_event(mouse(pygame.MOUSEMOTION, (panel.left + 2, panel.top + 2)))
    assert grid.handle_event(wheel(1))  # used by the panel, nothing moves
    assert n.value == 7
    assert grid.view.scale_x == scale
    grid.handle_event(mouse(pygame.MOUSEMOTION, (SIZE[0] - 20, 20)))
    grid.handle_event(wheel(1))
    grid.update()
    assert grid.view.scale_x != scale  # elsewhere the wheel zooms


def test_disabled_and_undrawn_sliders_cannot_be_dragged() -> None:
    grid, surface = make_grid()
    a = grid.slider("a", (0, 1), enabled=False)
    b = grid.slider("b", (0, 1), corner="top-left")
    frame(grid, surface)
    drag(grid, at(row(grid, a), 0.2), at(row(grid, a), 0.8))
    assert a.value == 0
    track = row(grid, b)
    frame(grid, surface, a)  # b is not drawn this frame
    center = grid.view.center
    drag(grid, at(track, 0.2), at(track, 0.8))
    assert b.value == 0
    assert grid.view.center != center  # where b was, the plane pans


def test_a_slider_stops_being_dragged_when_it_stops_being_drawn() -> None:
    grid, surface = make_grid()
    a = grid.slider("a", (0, 1))
    frame(grid, surface)
    track = row(grid, a)
    grid.handle_event(mouse(pygame.MOUSEBUTTONDOWN, at(track, 0.5)))
    assert a.dragging
    grid.draw(surface)  # a frame without the slider
    grid.draw(surface)
    assert not a.dragging


def test_sliders_are_above_draggable_points() -> None:
    grid, surface = make_grid()
    a = grid.slider("a", (0, 1))
    frame(grid, surface)
    track = row(grid, a)
    under = grid.to_world(*at(track, 0.5))
    point = grid.draggable(under)
    frame(grid, surface)
    drag(grid, at(track, 0.5), at(track, 0.9))
    assert point.position == pytest.approx(under)
    assert a.value == pytest.approx(0.9, abs=0.01)
    grid.handle_event(mouse(pygame.MOUSEMOTION, at(track, 0.5)))
    grid.update()
    assert not point.hovered  # the panel hides it from the cursor
    assert a.hovered


def test_a_dragged_point_does_not_grab_sliders_on_the_way() -> None:
    grid, surface = make_grid()
    a = grid.slider("a", (0, 1))
    point = grid.draggable((0.0, 0.0))
    frame(grid, surface)
    start = point.screen_position
    drag(grid, start, at(row(grid, a), 0.5), at(row(grid, a), 0.9))
    assert a.value == 0
    assert point.position != (0.0, 0.0)


def test_sliders_work_without_pan_and_zoom_and_report_moves() -> None:
    grid, surface = make_grid()
    grid.interactive = False
    a = grid.slider("a", (0, 1))
    frame(grid, surface)
    drag(grid, at(row(grid, a), 0.0), at(row(grid, a), 0.5))
    grid.draw(surface)
    assert a.moved
    grid.draw(surface)
    assert not a.moved


def test_hovering_a_slider_shows_the_hand_and_the_ring() -> None:
    grid, surface = make_grid()
    a = grid.slider("a", (0, 1), value=0.5)
    frame(grid, surface)
    grid.handle_event(mouse(pygame.MOUSEMOTION, at(row(grid, a), 0.5)))
    assert a.hovered
    assert len(HAND_CURSOR._wanting) == 1
    grid.handle_event(pygame.event.Event(pygame.WINDOWLEAVE))
    assert not a.hovered
    assert len(HAND_CURSOR._wanting) == 0


def test_removing_a_slider_forgets_it() -> None:
    grid, surface = make_grid()
    a = grid.slider("a", (0, 1))
    b = grid.slider("b", (0, 1))
    frame(grid, surface)
    grid.handle_event(mouse(pygame.MOUSEBUTTONDOWN, at(row(grid, a), 0.5)))
    a.remove()
    assert grid.sliders == (b,)
    assert not a.dragging
    with pytest.raises(ValueError, match="not a slider of this plane"):
        grid.draw_controls(surface, a)


def test_sliders_draw_their_handle_where_the_value_is() -> None:
    grid, surface = make_grid()
    a = grid.slider("a", (0, 10), value=7.5, color=(0, 0, 0))
    frame(grid, surface)
    track = row(grid, a)
    x, y = round(track.handle_x), round(track.y)
    pixels = pygame.surfarray.array3d(surface)
    assert tuple(pixels[x, y]) == (0, 0, 0)  # the handle's centre
    assert tuple(pixels[x - 30, y]) == (0, 0, 0)  # the filled part of the track
    assert np.all(pixels[x + 30, y] != 0)  # the rest of the track is grey


def test_integer_sliders_are_typed_and_validated_through_the_grid() -> None:
    grid, _ = make_grid()
    n = grid.slider("n", (0, 5), step=1)
    assert isinstance(n.value, int)
    with pytest.raises(ValueError, match="whole-number"):
        grid.slider("m", (0.5, 5), step=1)


def test_unchanged_sliders_are_drawn_from_the_same_primitives() -> None:
    """The renderer reuses coverage by identity: a still panel is not rasterised."""
    grid, surface = make_grid()
    a = grid.slider("a", (0, 1))
    b = grid.slider("b", (0, 1))
    frame(grid, surface)
    first = dict(grid._slider_drawings)
    frame(grid, surface)
    assert all(grid._slider_drawings[k][1] is first[k][1] for k in first)
    a.value = 0.5
    frame(grid, surface)
    assert grid._slider_drawings[a][1] is not first[a][1]
    assert grid._slider_drawings[b][1] is first[b][1]  # only a is rebuilt
    grid.theme = pgp.DARK
    frame(grid, surface)
    assert grid._slider_drawings[b][1] is not first[b][1]


# --- Sliders and the legend ------------------------------------------------------


def test_presses_on_the_legend_never_pan_or_reach_points_beneath() -> None:
    grid, surface = make_grid()
    grid.plot(surface, np.sin, label="sin x")
    grid.legend(surface)
    assert grid._legend_panel is not None
    rect = grid._legend_panel.rect
    inside = (rect.left + 5, rect.top + 5)
    point = grid.draggable(grid.to_world(*inside))
    frame(grid, surface)  # the next frame: the legend is drawn again
    grid.plot(surface, np.sin, label="sin x")
    grid.legend(surface)
    grid.draw_controls(surface, point)
    center = grid.view.center
    assert drag(grid, inside, (inside[0] - 40, inside[1] + 30))
    assert grid.view.center == center
    assert point.position == pytest.approx(grid.to_world(*inside))
    grid.handle_event(mouse(pygame.MOUSEMOTION, inside))
    grid.update()
    assert not point.hovered


def test_a_legend_in_a_corner_with_sliders_goes_beyond_their_panel() -> None:
    grid, surface = make_grid()
    grid.slider("a", (0, 1), corner="top-right")
    grid.slider("b", (0, 1), corner="top-right")
    # The first frame, with the legend before the sliders: they are expected.
    grid.plot(surface, np.sin, label="sin x")
    grid.legend(surface)
    grid.draw_controls(surface)
    sliders = grid._panels[0].rect
    assert grid._legend_panel is not None
    assert grid._legend_panel.rect.top > sliders.bottom
    # Later frames go by where the sliders were drawn.
    frame(grid, surface)
    grid.plot(surface, np.sin, label="sin x")
    grid.legend(surface)
    assert grid._legend_panel.rect.top == sliders.bottom + 10
    # Sliders no longer drawn leave the corner to the legend.
    for _ in range(2):
        grid.draw(surface)
        grid.plot(surface, np.sin, label="sin x")
        grid.legend(surface)
    assert grid._legend_panel.rect.top == 10
    # A legend in another corner is not moved.
    frame(grid, surface)
    grid.plot(surface, np.sin, label="sin x")
    grid.legend(surface, corner="top-left")
    assert grid._legend_panel.rect.top == 10
