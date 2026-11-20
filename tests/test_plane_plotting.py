"""End-to-end tests of the plotting API on :class:`pygridplot.Plane`.

These render into real (headless) pygame surfaces. The most important
checks are about panning and zooming: plotted objects must move rigidly
with the grid when panning, points must keep their pixel size when zooming,
and polygons must scale with the zoom.
"""

from __future__ import annotations

import math

import numpy as np
import pygame
import pytest

import pygridplot as pgp

pytestmark = pytest.mark.usefixtures("pygame_display")

SIZE = (320, 240)
BLACK = (0, 0, 0)


def blank_plane(**kwargs: object) -> tuple[pgp.Plane, pygame.Surface]:
    """Return a plane with an empty (grid-less) white theme, and a surface."""
    theme = pgp.GridTheme(
        show_minor=False, show_major=False, show_axes=False, show_labels=False
    )
    plane = pgp.Plane(theme=theme, **kwargs)  # type: ignore[arg-type]
    surface = pygame.Surface(SIZE)
    plane.draw_grid(surface)
    return plane, surface


def ink(surface: pygame.Surface) -> np.ndarray:
    """Return how far each pixel is from white, in [0, 1] (shape (h, w))."""
    rgb = pygame.surfarray.array3d(surface).astype(float)
    return (255.0 - rgb.min(axis=2)).T / 255.0


def test_plot_draws_a_curve_through_the_right_pixels() -> None:
    plane, surface = blank_plane()
    plane.plot(surface, lambda x: 0 * x + 1, color=BLACK)  # the line y = 1
    row = round(plane.to_screen(0.0, 1.0)[1])
    image = ink(surface)
    assert image[row, 20:300].min() > 0.9
    assert image[row - 4, :].max() == 0.0


def test_plot_survives_broken_functions() -> None:
    plane, surface = blank_plane()

    def broken(x: object) -> float:
        raise ZeroDivisionError

    plane.plot(surface, broken)
    plane.plot(surface, lambda x: "not a number")
    plane.plot(surface, math.sqrt)  # scalar-only, undefined for x < 0
    assert ink(surface)[:, : SIZE[0] // 2 - 2].max() == 0.0  # nothing left of 0
    assert ink(surface).max() > 0.5  # but sqrt is drawn for x >= 0


def test_drawing_is_clipped_to_the_plane_rect() -> None:
    surface = pygame.Surface(SIZE)
    surface.fill((255, 255, 255))
    plane = pgp.Plane(pygame.Rect(100, 50, 120, 100), scale=20)
    plane.plot(surface, lambda x: x, color=BLACK, width=5)
    plane.points(surface, [(-100, -100), (0, 0)], color=BLACK, radius=8)
    plane.polygon(surface, [(-50, -50), (50, -50), (0, 50)], color=BLACK)
    image = ink(surface)
    outside = np.ones_like(image, dtype=bool)
    outside[50:150, 100:220] = False
    assert image[outside].max() == 0.0
    assert image[~outside].max() > 0.9


def test_palette_colours_restart_every_frame() -> None:
    plane, surface = blank_plane()
    palette = plane.theme.palette
    seen = []
    for _ in range(2):
        plane.draw_grid(surface)
        for _ in range(2):
            seen.append(plane._colors.resolve(None))
    assert seen == [palette[0], palette[1], palette[0], palette[1]]


def test_explicit_colours_keep_the_palette_sequence() -> None:
    plane, surface = blank_plane()
    plane.plot(surface, np.sin, color="red")
    assert plane._colors.resolve(None) == plane.theme.palette[0]


def capture(plane: pgp.Plane, draw: object) -> np.ndarray:
    """Draw onto a fresh white surface and return its ink image."""
    surface = pygame.Surface(SIZE)
    plane.draw_grid(surface)
    draw(surface)  # type: ignore[operator]
    return ink(surface)


@pytest.mark.parametrize(("dx", "dy"), [(13, 0), (0, -9), (-21, 17)])
def test_plots_move_rigidly_when_panning(dx: int, dy: int) -> None:
    """Panning by whole pixels shifts every plotted pixel by exactly that much."""
    plane, _ = blank_plane(center=(0.31, -0.17), scale=37.0)

    def draw(surface: pygame.Surface) -> None:
        plane.plot(surface, lambda x: np.sin(2 * x) + 0.2 * x, color=BLACK)
        plane.points(surface, [(0.5, 0.5), (-1.2, 1.0)], color=BLACK)
        plane.polygon(surface, [(-3, -2), (-1, -2.5), (-2, -1)], color=BLACK)

    before = capture(plane, draw)
    plane.view.pan_pixels(dx, dy)
    after = capture(plane, draw)
    m = 30  # compare away from the edges, where content enters and leaves
    old = before[m - dy : SIZE[1] - m - dy, m - dx : SIZE[0] - m - dx]
    new = after[m : SIZE[1] - m, m : SIZE[0] - m]
    np.testing.assert_allclose(new, old, atol=2 / 255)


def test_points_keep_their_pixel_size_when_zooming() -> None:
    plane, _ = blank_plane()

    def draw(surface: pygame.Surface) -> None:
        plane.points(surface, [(0, 0)], color=BLACK, radius=6)

    small = capture(plane, draw).sum()
    plane.view.zoom(7.3)
    large = capture(plane, draw).sum()
    assert large == pytest.approx(small, rel=0.02)
    assert small == pytest.approx(math.pi * 36, rel=0.05)


def test_polygons_scale_with_the_zoom() -> None:
    plane, _ = blank_plane(scale=20.0)

    def draw(surface: pygame.Surface) -> None:
        plane.polygon(
            surface,
            [(-1, -1), (1, -1), (1, 1), (-1, 1)],
            color=BLACK,
            width=0,
            fill_opacity=1.0,
        )

    area_20 = capture(plane, draw).sum()
    plane.view.zoom(2.0)
    area_40 = capture(plane, draw).sum()
    assert area_20 == pytest.approx(40 * 40, rel=0.02)
    assert area_40 == pytest.approx(80 * 80, rel=0.02)


def test_curves_stay_correct_at_extreme_zoom() -> None:
    plane, surface = blank_plane(center=(1.0, 1.0))
    plane.view.zoom(1e9)
    plane.plot(surface, lambda x: x, color=BLACK)  # passes through the centre
    image = ink(surface)
    centre = image[
        SIZE[1] // 2 - 3 : SIZE[1] // 2 + 4, SIZE[0] // 2 - 3 : SIZE[0] // 2 + 4
    ]
    assert centre.max() > 0.9


def test_styles_and_keyword_overrides() -> None:
    plane, surface = blank_plane()
    thick = pgp.LineStyle(width=8, color=BLACK)
    plane.plot(surface, lambda x: 0 * x, style=thick)
    column = ink(surface)[:, 50]
    assert (column > 0.5).sum() == pytest.approx(8, abs=1)
    with pytest.raises(ValueError, match="width"):
        plane.plot(surface, np.sin, width=-1)


def test_show_plots_functions() -> None:
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    pgp.show(np.sin, lambda x: x**2, size=(200, 150))
