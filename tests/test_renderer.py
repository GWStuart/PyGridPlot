"""Pixel-level tests for the pygame renderer (run headless)."""

from __future__ import annotations

import warnings

import numpy as np
import pygame
import pytest

from pygridplot.core.coords import ScreenRect
from pygridplot.core.grid import build_grid
from pygridplot.core.labels import LabelText, TextRun
from pygridplot.core.primitives import (
    FontSpec,
    LineBatch,
    Orientation,
    PaintLayer,
    TextItem,
)
from pygridplot.core.theme import LIGHT
from pygridplot.core.view import View
from pygridplot.render.colors import ColorResolver
from pygridplot.render.pygame_renderer import PygameRenderer

pytestmark = pytest.mark.usefixtures("pygame_display")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def blank(size: tuple[int, int] = (40, 30)) -> pygame.Surface:
    """Return a white surface."""
    surface = pygame.Surface(size)
    surface.fill(WHITE)
    return surface


def column(surface: pygame.Surface, x: int) -> list[tuple[int, int, int]]:
    """Return the RGB colours of one pixel column."""
    return [tuple(surface.get_at((x, y)))[:3] for y in range(surface.get_height())]


def vertical(position: float, width: int, layers: tuple[PaintLayer, ...]) -> LineBatch:
    """Build a single vertical line spanning a 30-pixel-tall surface."""
    return LineBatch(
        Orientation.VERTICAL, np.array([position]), (0.0, 30.0), layers, width
    )


def test_one_pixel_lines_are_crisp() -> None:
    surface = blank()
    PygameRenderer().draw_line_batch(surface, vertical(10.3, 1, (PaintLayer(BLACK),)))
    assert set(column(surface, 10)) == {BLACK}
    assert set(column(surface, 9)) == {WHITE}
    assert set(column(surface, 11)) == {WHITE}


def test_two_pixel_lines_cover_two_whole_columns() -> None:
    surface = blank()
    PygameRenderer().draw_line_batch(surface, vertical(10.0, 2, (PaintLayer(BLACK),)))
    assert set(column(surface, 10)) == {BLACK}
    assert set(column(surface, 11)) == {BLACK}
    assert set(column(surface, 9)) == {WHITE}
    assert set(column(surface, 12)) == {WHITE}


def test_translucent_paint_is_flattened_against_a_known_background() -> None:
    surface = blank()
    layers = (PaintLayer((0, 0, 0), 0.5),)
    PygameRenderer().draw_line_batch(surface, vertical(5.0, 1, layers), WHITE)
    (r, g, b) = column(surface, 5)[0]
    assert r == g == b
    assert 126 <= r <= 129


def test_translucent_paint_blends_onto_unknown_backgrounds() -> None:
    surface = pygame.Surface((40, 30))
    surface.fill((200, 0, 0))
    layers = (PaintLayer((0, 0, 200), 0.5),)
    PygameRenderer().draw_line_batch(surface, vertical(5.0, 1, layers), None)
    r, g, b = column(surface, 5)[0]
    assert 95 <= r <= 105
    assert g == 0
    assert 95 <= b <= 105


def test_stacked_layers_blend_linearly() -> None:
    resolver = ColorResolver()
    stack = (PaintLayer((100, 100, 100), 1.0), PaintLayer((200, 200, 200), 0.25))
    assert resolver.flatten(stack, WHITE)[:3] == (125, 125, 125)


def test_named_and_hex_colours_resolve() -> None:
    resolver = ColorResolver()
    assert resolver.rgba("white") == (255, 255, 255, 255)
    assert resolver.rgba("#ff8000") == (255, 128, 0, 255)
    assert resolver.rgba(pygame.Color(1, 2, 3)) == (1, 2, 3, 255)  # type: ignore[arg-type]


def test_clipping_is_applied_and_restored() -> None:
    surface = blank((60, 40))
    previous = pygame.Rect(0, 0, 50, 40)
    surface.set_clip(previous)
    renderer = PygameRenderer()
    with renderer.clipped(surface, ScreenRect(10, 10, 20, 20)) as visible:
        assert visible == pygame.Rect(10, 10, 20, 20)
        surface.fill(BLACK)
    assert surface.get_clip() == previous
    assert tuple(surface.get_at((15, 15)))[:3] == BLACK
    assert tuple(surface.get_at((5, 5)))[:3] == WHITE
    with renderer.clipped(surface, ScreenRect(55, 0, 5, 5)) as visible:
        assert visible is None  # entirely outside the existing clip area


def test_measured_digit_box_matches_the_drawn_ink() -> None:
    renderer = PygameRenderer()
    font = FontSpec(None, 14)
    text = LabelText((TextRun("808"),))
    size = renderer.measure_text(text, font)
    surface = blank((80, 60))
    renderer.draw_text(surface, TextItem(text, 20, 25, font, BLACK))
    pixels = pygame.surfarray.array3d(surface).min(axis=2)  # (w, h)
    rows = np.flatnonzero(pixels.min(axis=0) < 128)
    cols = np.flatnonzero(pixels.min(axis=1) < 128)
    assert rows.min() == 25  # the digits' top row is exactly the item's top
    assert rows.max() == 25 + size.height - 1  # ...and they end on the baseline
    assert cols.min() >= 20
    assert cols.max() < 20 + size.width


def test_superscripts_overflow_above_the_digit_box() -> None:
    renderer = PygameRenderer()
    font = FontSpec(None, 14)
    plain = renderer.measure_text(LabelText((TextRun("2×10"),)), font)
    sci = renderer.measure_text(
        LabelText((TextRun("2×10"), TextRun("6", "superscript"))), font
    )
    assert sci.height == plain.height
    assert sci.width > plain.width
    assert sci.overflow_top > 0


def test_sign_width_is_reported() -> None:
    renderer = PygameRenderer()
    size = renderer.measure_text(
        LabelText((TextRun("−", "sign"), TextRun("3"))), FontSpec(None, 12)
    )
    assert 0 < size.sign_width < size.width


def test_halo_surrounds_the_glyphs() -> None:
    renderer = PygameRenderer()
    surface = pygame.Surface((80, 60))
    surface.fill((0, 0, 255))
    item = TextItem(
        LabelText((TextRun("8"),)),
        30,
        20,
        FontSpec(None, 14),
        BLACK,
        halo_color=WHITE,
        halo_width=2,
    )
    renderer.draw_text(surface, item)
    colours = {tuple(surface.get_at((x, y)))[:3] for x in range(80) for y in range(60)}
    assert WHITE in colours  # the halo is drawn
    assert BLACK in colours  # with the text on top


def test_unknown_fonts_fall_back_with_a_warning() -> None:
    renderer = PygameRenderer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        size = renderer.measure_text(
            LabelText((TextRun("1"),)), FontSpec("no-such-font-xyz", 12)
        )
    assert size.width > 0
    assert any("no-such-font-xyz" in str(w.message) for w in caught)


def test_grid_lines_use_only_theme_colours() -> None:
    """Outside the labels, every pixel is exactly a theme colour (no blur)."""
    renderer = PygameRenderer()
    view = View(ScreenRect(0, 0, 320, 240), (0.37, -0.21), 41.0)
    scene = build_grid(view, LIGHT, renderer)
    assert scene.x_spacing.coarse is None  # not mid-fade: colours are exact
    surface = pygame.Surface((320, 240))
    renderer.draw_grid(surface, scene)
    mask = np.ones((320, 240), dtype=bool)
    for item in scene.labels:  # ignore antialiased label areas
        size = renderer.measure_text(item.text, item.font)
        x0, y0 = max(item.left - 4, 0), max(item.top - size.overflow_top - 4, 0)
        mask[x0 : item.left + size.width + 4, y0 : item.top + size.height + 4] = False
    pixels = pygame.surfarray.array3d(surface)[mask]
    allowed = {
        tuple(LIGHT.background),  # type: ignore[arg-type]
        tuple(LIGHT.minor_color),  # type: ignore[arg-type]
        tuple(LIGHT.major_color),  # type: ignore[arg-type]
        tuple(LIGHT.axis_color),  # type: ignore[arg-type]
    }
    found = {tuple(int(c) for c in p) for p in np.unique(pixels, axis=0)}
    assert found <= allowed


def test_clear_caches_drops_everything_and_drawing_still_works() -> None:
    renderer = PygameRenderer()
    surface = blank((80, 60))
    font = FontSpec(None, 12)
    text = LabelText((TextRun("12"),))
    renderer.draw_text(surface, TextItem(text, 10, 10, font, BLACK))
    renderer.clear_caches()
    assert not renderer._text._rendered
    renderer.draw_text(surface, TextItem(text, 10, 30, font, BLACK))  # re-renders
    label_area = pygame.surfarray.array3d(surface)[10:30, 30:40]
    assert label_area.min() < 128  # the label was drawn again
