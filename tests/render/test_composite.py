"""Tests for blending coverage onto pygame surfaces and drawing markers."""

from __future__ import annotations

import numpy as np
import pygame
import pytest

from pygridplot.core.primitives import Fill, Markers, Stroke
from pygridplot.render.colors import ColorResolver
from pygridplot.render.composite import blend_dense, blend_sparse
from pygridplot.render.markers import PHASES, MarkerPainter, _split_positions
from pygridplot.render.pygame_renderer import PygameRenderer
from pygridplot.render.raster import DenseCoverage, SparseCoverage

pytestmark = pytest.mark.usefixtures("pygame_display")

RED = (255, 0, 0, 255)


def half_coverage(xs: list[int], ys: list[int]) -> SparseCoverage:
    """Sparse coverage of 0.5 at the given pixels."""
    return SparseCoverage(
        np.array(xs, dtype=np.intp), np.array(ys, dtype=np.intp), np.full(len(xs), 0.5)
    )


def test_sparse_blend_on_an_opaque_surface() -> None:
    surface = pygame.Surface((10, 10))
    surface.fill((0, 0, 255))
    blend_sparse(surface, half_coverage([2, 3], [4, 4]), RED)
    assert tuple(surface.get_at((2, 4)))[:3] == (128, 0, 128)
    assert tuple(surface.get_at((5, 5)))[:3] == (0, 0, 255)


def test_sparse_blend_respects_the_clip_area() -> None:
    surface = pygame.Surface((10, 10))
    surface.fill((0, 0, 0))
    surface.set_clip(pygame.Rect(0, 0, 3, 10))
    blend_sparse(surface, half_coverage([1, 5], [1, 1]), RED)
    assert tuple(surface.get_at((1, 1)))[:3] != (0, 0, 0)
    assert tuple(surface.get_at((5, 1)))[:3] == (0, 0, 0)


def test_blending_onto_transparent_layers_has_no_dark_fringe() -> None:
    layer = pygame.Surface((10, 10), pygame.SRCALPHA)
    layer.fill((0, 0, 0, 0))
    blend_sparse(layer, half_coverage([4], [4]), RED)
    r, g, b, a = layer.get_at((4, 4))
    assert (r, g, b) == (255, 0, 0)  # full colour...
    assert a == 128  # ...at half opacity: compositing it later looks right
    blend_sparse(layer, half_coverage([4], [4]), RED)
    # 0.5 over the stored 128/255 is 0.751, which is 191.5 in 8 bits.
    assert layer.get_at((4, 4))[3] == 192


def test_dense_blend_both_paths() -> None:
    coverage = DenseCoverage(2, 3, np.array([[1.0, 0.5], [0.0, 0.25]]))
    opaque = pygame.Surface((10, 10))
    opaque.fill((255, 255, 255))
    blend_dense(opaque, coverage, (0, 0, 0, 255))
    assert tuple(opaque.get_at((2, 3)))[:3] == (0, 0, 0)
    assert tuple(opaque.get_at((3, 3)))[:3] == (127, 127, 127)
    assert tuple(opaque.get_at((2, 4)))[:3] == (255, 255, 255)
    layer = pygame.Surface((10, 10), pygame.SRCALPHA)
    layer.fill((0, 0, 0, 0))
    blend_dense(layer, coverage, (0, 0, 255, 255), opacity=0.5)
    assert tuple(layer.get_at((2, 3))) == (0, 0, 255, 128)


def test_marker_positions_split_into_quarter_pixel_phases() -> None:
    whole, phase = _split_positions(np.array([[10.0, 5.24], [3.9, 7.5], [1.13, -2.6]]))
    assert whole.tolist() == [[10, 5], [4, 7], [1, -3]]
    assert phase.tolist() == [[0, 1], [0, 2], [1, 2]]
    assert int(phase.max()) < PHASES


def test_markers_are_drawn_centred_and_stamps_are_cached() -> None:
    surface = pygame.Surface((40, 40))
    surface.fill((255, 255, 255))
    painter = MarkerPainter(ColorResolver())
    markers = Markers(np.array([[20.0, 20.0]]), "circle", 4.0, (0, 0, 0))
    painter.draw(surface, markers)
    assert tuple(surface.get_at((20, 20)))[:3] == (0, 0, 0)
    assert tuple(surface.get_at((20, 26)))[:3] == (255, 255, 255)
    assert tuple(surface.get_at((26, 20)))[:3] == (255, 255, 255)
    cached = len(painter._stamps)
    painter.draw(surface, markers)
    assert len(painter._stamps) == cached  # same style and phase: reused


def test_renderer_draws_every_primitive_type() -> None:
    surface = pygame.Surface((60, 60))
    surface.fill((255, 255, 255))
    renderer = PygameRenderer()
    renderer.draw_fill(
        surface,
        Fill((np.array([[5.0, 5.0], [25.0, 5.0], [25.0, 25.0], [5.0, 25.0]]),), "blue"),
    )
    renderer.draw_stroke(
        surface, Stroke((np.array([[30.0, 40.0], [55.0, 40.0]]),), "red", 3.0)
    )
    renderer.draw_markers(
        surface, Markers(np.array([[45.0, 15.0]]), "square", 5.0, "green")
    )
    assert tuple(surface.get_at((15, 15)))[:3] == (0, 0, 255)
    assert tuple(surface.get_at((40, 40)))[:3] == (255, 0, 0)
    assert tuple(surface.get_at((45, 15)))[:3] == (0, 255, 0)


def test_transparent_primitives_draw_nothing() -> None:
    surface = pygame.Surface((20, 20))
    surface.fill((1, 2, 3))
    renderer = PygameRenderer()
    renderer.draw_stroke(
        surface, Stroke((np.array([[0.0, 5.0], [20.0, 5.0]]),), "red", 3.0, opacity=0.0)
    )
    renderer.draw_markers(
        surface, Markers(np.array([[5.0, 5.0]]), "circle", 3.0, "red", opacity=0.0)
    )
    assert (
        pygame.surfarray.array3d(surface).reshape(-1, 3).tolist() == [[1, 2, 3]] * 400
    )


def big_coverage() -> SparseCoverage:
    """Return random coverage of every pixel of a 300x200 block."""
    rng = np.random.default_rng(1)
    xs, ys = np.meshgrid(np.arange(300), np.arange(200))
    return SparseCoverage(
        xs.ravel().astype(np.intp), ys.ravel().astype(np.intp), rng.random(60000)
    )


def blended(surface: pygame.Surface, draw: object) -> np.ndarray:
    """Fill a surface, apply ``draw`` to it and return its pixels."""
    surface.fill((10, 200, 30))
    draw(surface)  # type: ignore[operator]
    return pygame.surfarray.array3d(surface).astype(int)


def test_packed_blending_matches_per_channel_blending() -> None:
    from pygridplot.render import composite

    coverage = big_coverage()
    surface = pygame.Surface((300, 200))
    assert composite._packed_shifts(surface) is not None  # a 32-bit surface
    per_channel = blended(
        surface,
        lambda s: composite._over_opaque(
            s, coverage.x, coverage.y, coverage.alpha, RED
        ),
    )
    packed = blended(surface, lambda s: blend_sparse(s, coverage, RED))
    # float32 arithmetic may round a value lying exactly between two levels
    # the other way, which is invisible.
    difference = np.abs(per_channel - packed)
    assert difference.max() <= 1
    assert (difference > 0).mean() < 0.001


def test_surfaces_without_packed_access_blend_big_strokes_as_a_patch() -> None:
    from pygridplot.render import composite

    coverage = big_coverage()
    surface = pygame.Surface((300, 200), depth=24)
    assert composite._packed_shifts(surface) is None
    per_pixel = blended(
        surface,
        lambda s: composite._over_opaque(
            s, coverage.x, coverage.y, coverage.alpha, RED
        ),
    )
    patched = blended(surface, lambda s: blend_sparse(s, coverage, RED))  # > threshold
    assert np.abs(per_pixel - patched).max() <= 1  # identical up to 8-bit rounding
