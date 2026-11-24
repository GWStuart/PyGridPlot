"""End-to-end tests of the plotting API on :class:`pygridplot.Grid`.

These render into real (headless) pygame surfaces. The most important
checks are about panning and zooming: plotted objects must move rigidly
with the grid when panning, points must keep their pixel size when zooming,
and polygons must scale with the zoom.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pygame
import pytest

import pygridplot as pgp

pytestmark = pytest.mark.usefixtures("pygame_display")

SIZE = (320, 240)
BLACK = (0, 0, 0)


def blank_grid(**kwargs: object) -> tuple[pgp.Grid, pygame.Surface]:
    """Return a plane with an empty (grid-less) white theme, and a surface."""
    theme = pgp.GridTheme(
        show_minor=False, show_major=False, show_axes=False, show_labels=False
    )
    grid = pgp.Grid(theme=theme, **kwargs)  # type: ignore[arg-type]
    surface = pygame.Surface(SIZE)
    grid.draw(surface)
    return grid, surface


def ink(surface: pygame.Surface) -> np.ndarray:
    """Return how far each pixel is from white, in [0, 1] (shape (h, w))."""
    rgb = pygame.surfarray.array3d(surface).astype(float)
    return (255.0 - rgb.min(axis=2)).T / 255.0


def test_plot_draws_a_curve_through_the_right_pixels() -> None:
    grid, surface = blank_grid()
    grid.plot(surface, lambda x: 0 * x + 1, color=BLACK)  # the line y = 1
    row = round(grid.to_screen(0.0, 1.0)[1])
    image = ink(surface)
    assert image[row, 20:300].min() > 0.9
    assert image[row - 4, :].max() == 0.0


def test_plot_survives_broken_functions() -> None:
    grid, surface = blank_grid()

    def broken(x: object) -> float:
        raise ZeroDivisionError

    grid.plot(surface, broken)
    grid.plot(surface, lambda x: "not a number")
    grid.plot(surface, math.sqrt)  # scalar-only, undefined for x < 0
    assert ink(surface)[:, : SIZE[0] // 2 - 2].max() == 0.0  # nothing left of 0
    assert ink(surface).max() > 0.5  # but sqrt is drawn for x >= 0


def test_drawing_is_clipped_to_the_grid_rect() -> None:
    surface = pygame.Surface(SIZE)
    surface.fill((255, 255, 255))
    grid = pgp.Grid(pygame.Rect(100, 50, 120, 100), scale=20)
    grid.plot(surface, lambda x: x, color=BLACK, width=5)
    grid.points(surface, [(-100, -100), (0, 0)], color=BLACK, radius=8)
    grid.polygon(surface, [(-50, -50), (50, -50), (0, 50)], color=BLACK)
    image = ink(surface)
    outside = np.ones_like(image, dtype=bool)
    outside[50:150, 100:220] = False
    assert image[outside].max() == 0.0
    assert image[~outside].max() > 0.9


def test_palette_colours_restart_every_frame() -> None:
    grid, surface = blank_grid()
    palette = grid.theme.palette
    seen = []
    for _ in range(2):
        grid.draw(surface)
        for _ in range(2):
            seen.append(grid._colors.resolve(None))
    assert seen == [palette[0], palette[1], palette[0], palette[1]]


def test_explicit_colours_keep_the_palette_sequence() -> None:
    grid, surface = blank_grid()
    grid.plot(surface, np.sin, color="red")
    assert grid._colors.resolve(None) == grid.theme.palette[0]


def capture(grid: pgp.Grid, draw: object) -> np.ndarray:
    """Draw onto a fresh white surface and return its ink image."""
    surface = pygame.Surface(SIZE)
    grid.draw(surface)
    draw(surface)  # type: ignore[operator]
    return ink(surface)


@pytest.mark.parametrize(("dx", "dy"), [(13, 0), (0, -9), (-21, 17)])
def test_plots_move_rigidly_when_panning(dx: int, dy: int) -> None:
    """Panning by whole pixels shifts every plotted pixel by exactly that much."""
    grid, _ = blank_grid(center=(0.31, -0.17), scale=37.0)

    def draw(surface: pygame.Surface) -> None:
        grid.plot(surface, lambda x: np.sin(2 * x) + 0.2 * x, color=BLACK)
        grid.points(surface, [(0.5, 0.5), (-1.2, 1.0)], color=BLACK)
        grid.polygon(surface, [(-3, -2), (-1, -2.5), (-2, -1)], color=BLACK)
        grid.text(surface, "Text\nlines", (-1.3, -0.4), anchor="bottom-left", gap=3)
        grid.points(surface, [(1.5, -1.0)], y_error=0.5, x_error=0.3, color=BLACK)
        grid.polyline(surface, [(-3, 1), (-2, 2), (-1, 1.5)], step="mid", color=BLACK)
        grid.bars(surface, [0.5, -0.4, 0.8], start=1.0, color=BLACK)
        grid.histogram(surface, [-0.6, -0.5, -0.1, 0.3], width=0.25, color=BLACK)

    before = capture(grid, draw)
    grid.view.pan_pixels(dx, dy)
    after = capture(grid, draw)
    m = 30  # compare away from the edges, where content enters and leaves
    old = before[m - dy : SIZE[1] - m - dy, m - dx : SIZE[0] - m - dx]
    new = after[m : SIZE[1] - m, m : SIZE[0] - m]
    np.testing.assert_allclose(new, old, atol=2 / 255)


def test_points_keep_their_pixel_size_when_zooming() -> None:
    grid, _ = blank_grid()

    def draw(surface: pygame.Surface) -> None:
        grid.points(surface, [(0, 0)], color=BLACK, radius=6)

    small = capture(grid, draw).sum()
    grid.view.zoom(7.3)
    large = capture(grid, draw).sum()
    assert large == pytest.approx(small, rel=0.02)
    assert small == pytest.approx(math.pi * 36, rel=0.05)


def test_polygons_scale_with_the_zoom() -> None:
    grid, _ = blank_grid(scale=20.0)

    def draw(surface: pygame.Surface) -> None:
        grid.polygon(
            surface,
            [(-1, -1), (1, -1), (1, 1), (-1, 1)],
            color=BLACK,
            width=0,
            fill_opacity=1.0,
        )

    area_20 = capture(grid, draw).sum()
    grid.view.zoom(2.0)
    area_40 = capture(grid, draw).sum()
    assert area_20 == pytest.approx(40 * 40, rel=0.02)
    assert area_40 == pytest.approx(80 * 80, rel=0.02)


def test_curves_stay_correct_at_extreme_zoom() -> None:
    grid, surface = blank_grid(center=(1.0, 1.0))
    grid.view.zoom(1e9)
    grid.plot(surface, lambda x: x, color=BLACK)  # passes through the centre
    image = ink(surface)
    centre = image[
        SIZE[1] // 2 - 3 : SIZE[1] // 2 + 4, SIZE[0] // 2 - 3 : SIZE[0] // 2 + 4
    ]
    assert centre.max() > 0.9


def test_styles_and_keyword_overrides() -> None:
    grid, surface = blank_grid()
    thick = pgp.LineStyle(width=8, color=BLACK)
    grid.plot(surface, lambda x: 0 * x, style=thick)
    column = ink(surface)[:, 50]
    assert (column > 0.5).sum() == pytest.approx(8, abs=1)
    with pytest.raises(ValueError, match="width"):
        grid.plot(surface, np.sin, width=-1)


def test_show_plots_functions() -> None:
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    pgp.show(np.sin, lambda x: x**2, size=(200, 150))
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    pgp.show(np.sin, size=(200, 150), x_title="x", y_title="sin x")


def test_parametric_and_polar_curves_draw_through_the_right_pixels() -> None:
    grid, surface = blank_grid()
    grid.parametric(surface, np.cos, np.sin, (0, 2 * np.pi), color=BLACK)
    grid.polar(surface, 2.0, color=BLACK)
    painted = ink(surface)
    cx, cy = (round(v) for v in grid.to_screen(0.0, 0.0))
    assert painted[cy, cx + 50] > 0.9  # the unit circle (50 px per unit)
    assert painted[cy - 100, cx] > 0.9  # the circle r = 2, at the top
    assert painted[cy, cx + 75] < 0.05  # nothing in between


def test_curves_take_palette_colours_and_styles() -> None:
    grid, surface = blank_grid()
    grid.parametric(surface, np.cos, np.sin, (0, 2 * np.pi))
    grid.polar(surface, 2.0, style=pgp.LineStyle(width=4.0))
    cx, cy = (round(v) for v in grid.to_screen(0.0, 0.0))
    first, second = grid.theme.palette[:2]
    assert tuple(surface.get_at((cx + 50, cy)))[:3] == first
    assert tuple(surface.get_at((cx + 100, cy)))[:3] == second
    assert tuple(surface.get_at((cx + 101, cy)))[:3] == second  # 4 px wide


@pytest.mark.parametrize(("dx", "dy"), [(7, -3), (-12, 5)])
def test_curves_move_rigidly_when_panning(dx: int, dy: int) -> None:
    grid, surface = blank_grid()
    rose = lambda t: 2 * np.cos(3 * t)  # noqa: E731
    grid.polar(surface, rose, color=BLACK)
    before = ink(surface)
    grid.view.pan_pixels(dx, dy)
    grid.draw(surface)
    grid.polar(surface, rose, color=BLACK)
    after = ink(surface)
    inner = (slice(40, 200), slice(40, 280))
    shifted = np.roll(np.roll(before, dy, axis=0), dx, axis=1)
    # As for function plots, float round-off may change a pixel by a level.
    np.testing.assert_allclose(after[inner], shifted[inner], atol=2 / 255)
    assert before.max() > 0.9  # the curve was drawn at all


def test_implicit_curves_draw_through_the_right_pixels() -> None:
    grid, surface = blank_grid()
    grid.implicit(surface, lambda x, y: x**2 + y**2, 4, color=BLACK)  # r = 2
    grid.implicit(surface, lambda x, y: y - np.tan(x), color=BLACK)
    painted = ink(surface)
    cx, cy = (round(v) for v in grid.to_screen(0.0, 0.0))
    assert painted[cy - 100, cx] > 0.9  # the circle, at the top
    assert painted[cy, cx - 100] > 0.9  # and at the left
    assert painted[cy, cx] > 0.9  # tan passes through the origin
    # No line is drawn down the asymptote of tan at x = pi / 2.
    only_tan = capture(
        grid, lambda s: grid.implicit(s, lambda x, y: y - np.tan(x), color=BLACK)
    )
    column = round(grid.to_screen(math.pi / 2, 0.0)[0])
    assert only_tan[:, column].max() < 0.05
    # The branch leaves the top (y = 2.4) at x = arctan(2.4), 20 px before.
    assert only_tan[:, column - 25].max() > 0.9


def test_implicit_levels_draw_contour_lines_in_one_colour() -> None:
    grid, surface = blank_grid()
    grid.implicit(surface, lambda x, y: x**2 + y**2, [1, 4])
    first = grid.theme.palette[0]
    cx, cy = (round(v) for v in grid.to_screen(0.0, 0.0))
    assert tuple(surface.get_at((cx + 50, cy)))[:3] == first  # r = 1
    assert tuple(surface.get_at((cx + 100, cy)))[:3] == first  # r = 2


def test_implicit_curves_take_palette_colours_and_styles() -> None:
    grid, surface = blank_grid()
    grid.implicit(surface, lambda x, y: x - 1)
    grid.implicit(surface, lambda x, y: x + 1, style=pgp.LineStyle(width=4.0))
    first, second = grid.theme.palette[:2]
    row = SIZE[1] // 3
    one, minus_one = (round(grid.to_screen(x, 0.0)[0]) for x in (1.0, -1.0))
    assert tuple(surface.get_at((one, row)))[:3] == first
    assert tuple(surface.get_at((minus_one, row)))[:3] == second
    assert tuple(surface.get_at((minus_one + 1, row)))[:3] == second  # 4 px wide


def test_implicit_curves_survive_broken_functions() -> None:
    grid, surface = blank_grid()

    def broken(x: object, y: object) -> float:
        raise ZeroDivisionError

    grid.implicit(surface, broken)
    grid.implicit(surface, lambda x, y: "not a number")
    grid.implicit(surface, lambda x, y: math.log(x) - y)  # scalar-only, x > 0
    assert ink(surface)[:, : SIZE[0] // 2 - 2].max() == 0.0  # nothing left of 0
    with pytest.raises(ValueError, match="level"):
        grid.implicit(surface, lambda x, y: x, math.inf)


@pytest.mark.parametrize(("dx", "dy"), [(7, -3), (-12, 5)])
def test_implicit_curves_move_rigidly_when_panning(dx: int, dy: int) -> None:
    grid, _ = blank_grid(center=(0.13, 0.29))
    heart = lambda x, y: (x**2 + y**2 - 1) ** 3 - x**2 * y**3  # noqa: E731
    before = capture(grid, lambda s: grid.implicit(s, heart, color=BLACK))
    grid.view.pan_pixels(dx, dy)
    after = capture(grid, lambda s: grid.implicit(s, heart, color=BLACK))
    inner = (slice(40, 200), slice(40, 280))
    shifted = np.roll(np.roll(before, dy, axis=0), dx, axis=1)
    np.testing.assert_allclose(after[inner], shifted[inner], atol=2 / 255)
    assert before.max() > 0.9  # the curve was drawn at all


# ----------------------------------------------------------------------
# Areas between graphs, and regions
# ----------------------------------------------------------------------
def test_fill_between_shades_between_the_graphs() -> None:
    grid, surface = blank_grid()
    grid.fill_between(surface, 1, -1, color=BLACK, opacity=0.5)
    image = ink(surface)
    top = round(grid.to_screen(0.0, 1.0)[1])
    bottom = round(grid.to_screen(0.0, -1.0)[1])
    assert np.allclose(image[top + 2 : bottom - 2, :], 0.5, atol=2 / 255)
    assert image[: top - 1, :].max() == 0.0
    assert image[bottom + 2 :, :].max() == 0.0


def test_fill_between_defaults_to_the_x_axis() -> None:
    grid, surface = blank_grid()
    grid.fill_between(surface, lambda x: 0 * x + 1, color=BLACK)
    image = ink(surface)
    zero = round(grid.to_screen(0.0, 0.0)[1])
    one = round(grid.to_screen(0.0, 1.0)[1])
    assert image[one + 2 : zero - 1, :].min() > 0.2
    assert image[zero + 2 :, :].max() == 0.0


def test_fill_between_shades_both_sides_of_a_crossing() -> None:
    grid, surface = blank_grid()
    grid.fill_between(surface, lambda x: x, lambda x: -x, color=BLACK)
    image = ink(surface)
    cx, cy = (round(v) for v in grid.to_screen(0.0, 0.0))
    assert image[cy, cx - 40] > 0.2  # left lobe
    assert image[cy, cx + 40] > 0.2  # right lobe
    assert image[cy - 40, cx] == 0.0  # not between the lobes
    assert image[cy + 40, cx] == 0.0


def test_fill_between_does_not_join_jumps() -> None:
    grid, surface = blank_grid()
    grid.fill_between(surface, np.tan, color=BLACK)
    image = ink(surface)
    pole = round(grid.to_screen(math.pi / 2, 0.0)[0])
    cy = round(grid.to_screen(0.0, 0.0)[1])
    # Left of the pole tan is large and positive; right of it, negative.
    assert image[cy - 60, pole - 4] > 0.2
    assert image[cy + 60, pole - 4] == 0.0
    assert image[cy + 60, pole + 4] > 0.2
    assert image[cy - 60, pole + 4] == 0.0


def test_fill_between_survives_broken_functions() -> None:
    grid, surface = blank_grid()

    def broken(x: object) -> float:
        raise ZeroDivisionError

    grid.fill_between(surface, broken, 0)
    grid.fill_between(surface, np.sin, lambda x: "not a number")
    assert ink(surface).max() == 0.0
    grid.fill_between(surface, math.sqrt, 0, color=BLACK)  # scalar-only, x >= 0
    image = ink(surface)
    assert image[:, : SIZE[0] // 2 - 1].max() == 0.0
    assert image.max() > 0.2
    with pytest.raises(TypeError, match="number"):
        grid.fill_between(surface, np.sin, "0")  # type: ignore[arg-type]


def test_fill_between_on_log_axes_reaches_the_bottom_for_zero() -> None:
    grid, surface = blank_grid(log_x=True, log_y=True)
    grid.fill_between(surface, lambda x: x, 0, domain=(1, 10), color=BLACK)
    image = ink(surface)
    column = round(grid.to_screen(3.0, 1.0)[0])
    row = round(grid.to_screen(3.0, 3.0)[1])
    assert image[row + 2 :, column].min() > 0.2  # down to the bottom edge
    assert image[: row - 2, column].max() == 0.0


def test_inequality_shades_where_the_condition_holds() -> None:
    grid, surface = blank_grid()
    grid.inequality(
        surface, lambda x, y: (x**2 + y**2 > 1) & (x**2 + y**2 < 4), color=BLACK
    )
    image = ink(surface)
    cx, cy = (round(v) for v in grid.to_screen(0.0, 0.0))
    assert image[cy, cx] == 0.0  # the hole
    assert image[cy, cx + 75] > 0.2  # the ring, r = 1.5
    assert image[cy - 75, cx] > 0.2
    assert image[cy, cx + 105] == 0.0  # outside, r = 2.1


def test_regions_have_no_seams() -> None:
    """The parts of a large region are filled as one: every pixel alike."""
    grid, surface = blank_grid()
    grid.inequality(surface, lambda x, y: x > 0.3 * np.sin(3 * y), color=BLACK)
    image = ink(surface)
    right = image[:, SIZE[0] // 2 + 20 :]
    assert np.ptp(right) == 0.0
    assert right.min() > 0.2


def test_inequalities_take_palette_colours_and_styles() -> None:
    grid, surface = blank_grid()
    grid.inequality(surface, lambda x, y: x < -1)
    grid.inequality(surface, lambda x, y: x > 1, style=pgp.FillStyle(opacity=1.0))
    first, second = grid.theme.palette[:2]
    cy = SIZE[1] // 2
    left = tuple(surface.get_at((5, cy)))[:3]
    assert left != (255, 255, 255)
    assert left != first  # translucent, by default
    assert tuple(surface.get_at((SIZE[0] - 5, cy)))[:3] == second
    grid.fill_between(surface, 1.5, 2, color="red", opacity=1.0)
    top = round(grid.to_screen(0.0, 1.75)[1])
    assert tuple(surface.get_at((SIZE[0] // 2, top)))[:3] == (255, 0, 0)


def test_inequalities_survive_broken_conditions() -> None:
    grid, surface = blank_grid()

    def broken(x: object, y: object) -> bool:
        raise ZeroDivisionError

    grid.inequality(surface, broken)
    grid.inequality(surface, lambda x, y: "not a number")
    assert ink(surface).max() == 0.0
    # Python's `and` works on single numbers (more slowly).
    grid.inequality(surface, lambda x, y: x > 0 and y > 0, color=BLACK)
    image = ink(surface)
    cx, cy = (round(v) for v in grid.to_screen(0.0, 0.0))
    assert image[cy - 20, cx + 20] > 0.2
    assert image[cy + 20, cx + 20] == 0.0
    assert image[cy - 20, cx - 20] == 0.0


@pytest.mark.parametrize(("dx", "dy"), [(7, -3), (-12, 5)])
def test_areas_move_rigidly_when_panning(dx: int, dy: int) -> None:
    grid, _ = blank_grid(center=(0.13, 0.29))

    def draw(surface: pygame.Surface) -> None:
        grid.inequality(surface, lambda x, y: x**2 + 2 * y**2 < 3, color=BLACK)
        grid.fill_between(surface, np.sin, lambda x: 0.5 * x - 1, color=BLACK)

    before = capture(grid, draw)
    grid.view.pan_pixels(dx, dy)
    after = capture(grid, draw)
    inner = (slice(40, 200), slice(40, 280))
    shifted = np.roll(np.roll(before, dy, axis=0), dx, axis=1)
    np.testing.assert_allclose(after[inner], shifted[inner], atol=2 / 255)
    assert before.max() > 0.2


# ----------------------------------------------------------------------
# Arrows and fields
# ----------------------------------------------------------------------
def test_arrows_draw_their_shaft_and_head() -> None:
    grid, surface = blank_grid()
    grid.arrow(surface, (-2, 0), (2, 0), color=BLACK, width=3, head_size=12)
    image = ink(surface)
    cx, cy = (round(v) for v in grid.to_screen(0.0, 0.0))
    tip = round(grid.to_screen(2.0, 0.0)[0])
    assert image[cy, cx - 90 : tip - 2].min() > 0.9  # the shaft, up to the tip
    assert image[cy - 3, tip - 10] > 0.9  # the head is wider than the shaft
    assert image[cy - 3, cx] == 0.0
    assert image[cy, tip + 2 :].max() == 0.0  # nothing beyond the tip


def test_translucent_arrows_have_no_darker_joins() -> None:
    grid, surface = blank_grid()
    grid.arrow(surface, (-2, 0), (2, 0), color=BLACK, width=5, opacity=0.5)
    image = ink(surface)
    cy = round(grid.to_screen(0.0, 0.0)[1])
    tip = round(grid.to_screen(2.0, 0.0)[0])
    inside = image[cy, 80 : tip - 4]  # along the shaft and into the head
    assert np.ptp(inside) < 2 / 255
    assert inside.min() == pytest.approx(0.5, abs=2 / 255)


def test_many_arrows_take_one_colour_and_style() -> None:
    grid, surface = blank_grid()
    grid.arrow(surface, (0, 0), [(2, 0), (-2, 0), (0, 2)])
    style = pgp.ArrowStyle(color="red", width=4.0)
    grid.arrow(surface, (0, -1), (2, -1), style=style)
    first = grid.theme.palette[0]
    cx, cy = (round(v) for v in grid.to_screen(0.0, 0.0))
    assert tuple(surface.get_at((cx + 50, cy)))[:3] == first
    assert tuple(surface.get_at((cx - 50, cy)))[:3] == first
    assert tuple(surface.get_at((cx, cy - 50)))[:3] == first
    row = round(grid.to_screen(0.0, -1.0)[1])
    assert tuple(surface.get_at((cx + 50, row)))[:3] == (255, 0, 0)
    assert tuple(surface.get_at((cx + 50, row + 1)))[:3] == (255, 0, 0)  # 4 px wide
    with pytest.raises(ValueError, match="starts and"):
        grid.arrow(surface, [(0, 0), (1, 1)], [(1, 0), (2, 0), (3, 0)])


def test_vector_fields_draw_a_lattice_of_arrows() -> None:
    grid, surface = blank_grid()  # 50 px per unit: arrows 1 unit apart
    grid.vector_field(surface, lambda x, y: (1 + 0 * x, 0 * y), color=BLACK)
    image = ink(surface)
    cx, cy = (round(v) for v in grid.to_screen(0.0, 0.0))
    for row in (cy - 50, cy, cy + 50):
        assert image[row, cx - 12 : cx + 12].min() > 0.9  # a horizontal arrow
    assert image[cy - 25, :].max() == 0.0  # and nothing between the rows
    assert image[:, cx + 25].max() == 0.0  # nor between the columns


def test_slope_fields_draw_short_segments() -> None:
    grid, surface = blank_grid()
    grid.slope_field(surface, lambda x, y: 0 * x + 0 * y, color=BLACK)
    image = ink(surface)
    cx, cy = (round(v) for v in grid.to_screen(0.0, 0.0))
    assert image[cy, cx - 10 : cx + 10].min() > 0.9  # a level segment at the origin
    assert image[cy, cx + 14 : cx + 36].max() == 0.0  # and a gap to the next


def test_fields_survive_broken_functions() -> None:
    grid, surface = blank_grid()

    def broken(x: object, y: object) -> object:
        raise ZeroDivisionError

    grid.vector_field(surface, broken)
    grid.vector_field(surface, lambda x, y: "not a vector")
    grid.slope_field(surface, broken)
    assert ink(surface).max() == 0.0
    grid.vector_field(surface, lambda x, y: (math.sqrt(x), 0.0), color=BLACK)
    assert ink(surface)[:, : SIZE[0] // 2 - 25].max() == 0.0  # sqrt needs x >= 0
    assert ink(surface).max() > 0.5
    with pytest.raises(ValueError, match="spacing"):
        grid.vector_field(surface, lambda x, y: (x, y), spacing=1)
    with pytest.raises(ValueError, match="scale"):
        grid.vector_field(surface, lambda x, y: (x, y), scale=-2)


@pytest.mark.parametrize(("dx", "dy"), [(7, -3), (-12, 5)])
def test_fields_move_rigidly_when_panning(dx: int, dy: int) -> None:
    grid, _ = blank_grid(center=(0.13, 0.29), scale=63.0)

    def draw(surface: pygame.Surface) -> None:
        grid.vector_field(surface, lambda x, y: (np.sin(y), np.cos(x)), color=BLACK)
        grid.slope_field(surface, lambda x, y: x * y, color=BLACK, spacing=55)

    before = capture(grid, draw)
    grid.view.pan_pixels(dx, dy)
    after = capture(grid, draw)
    inner = (slice(40, 200), slice(40, 280))
    shifted = np.roll(np.roll(before, dy, axis=0), dx, axis=1)
    np.testing.assert_allclose(after[inner], shifted[inner], atol=2 / 255)
    assert before.max() > 0.5


def test_calculus_results_plot_like_the_functions_they_equal() -> None:
    """An antiderivative, a Taylor polynomial and a Fourier sum are just functions."""
    cases = [
        (pgp.antiderivative(np.cos), np.sin),
        (pgp.antiderivative(lambda x: 2 * x, -1.0), lambda x: x**2 - 1),
        (pgp.taylor(lambda x: x**3 - x, 0.5, 3), lambda x: x**3 - x),
        (pgp.fourier(lambda x: np.cos(2 * x), 3), lambda x: np.cos(2 * x)),
    ]
    for computed, exact in cases:
        a, b = (blank_grid(scale=37.0) for _ in range(2))
        a[0].plot(a[1], computed, color=BLACK)
        b[0].plot(b[1], exact, color=BLACK)
        assert ink(a[1]).max() > 0.9
        np.testing.assert_allclose(ink(a[1]), ink(b[1]), atol=2 / 255)


def test_antiderivatives_end_where_their_integral_diverges() -> None:
    grid, surface = blank_grid()
    grid.plot(surface, pgp.antiderivative(lambda x: 1 / x, 1.0), color=BLACK)  # log x
    image = ink(surface)
    assert image[:, : SIZE[0] // 2 - 1].max() == 0.0  # nothing for x < 0
    assert image[:, SIZE[0] // 2 + 2 :].max() > 0.9


@pytest.mark.parametrize(("dx", "dy"), [(11, 0), (-6, 13)])
def test_antiderivatives_move_rigidly_when_panning(dx: int, dy: int) -> None:
    grid, _ = blank_grid(center=(0.21, -0.4), scale=45.0)

    def draw(surface: pygame.Surface) -> None:
        grid.plot(surface, pgp.antiderivative(np.sign, 0.3), color=BLACK)  # |x - 0.3|
        grid.plot(surface, pgp.fourier(np.sign, 7), color=BLACK)

    before = capture(grid, draw)
    grid.view.pan_pixels(dx, dy)
    after = capture(grid, draw)
    m = 30
    old = before[m - dy : SIZE[1] - m - dy, m - dx : SIZE[0] - m - dx]
    new = after[m : SIZE[1] - m, m : SIZE[0] - m]
    np.testing.assert_allclose(new, old, atol=2 / 255)


def test_sequences_draw_stems_and_markers_at_the_terms() -> None:
    grid, surface = blank_grid(scale=40.0)
    grid.sequence(surface, [2.0, -1.0, 1.5], start=1, color=BLACK)
    image = ink(surface)
    for n, value in [(1, 2.0), (2, -1.0), (3, 1.5)]:
        px, py = (round(v) for v in grid.to_screen(n, value))
        assert image[py, px] > 0.9  # the marker
        _, mid = (round(v) for v in grid.to_screen(n, value / 2))
        assert image[mid, px] > 0.9  # the stem, halfway down
    gap = round(grid.to_screen(1.5, 0.0)[0])
    _, above = (round(v) for v in grid.to_screen(1.5, 1.0))
    assert image[above, gap] == 0.0  # nothing between the stems


def test_sequences_of_functions_and_styles() -> None:
    grid, surface = blank_grid(scale=40.0)
    grid.sequence(surface, lambda n: math.factorial(n) / 3, start=0, width=0)  # dots
    assert ink(surface).max() > 0.5
    stems = pgp.SequenceStyle(radius=0, width=4)
    grid.sequence(surface, np.cos, style=stems, color=BLACK, opacity=0.5)
    with pytest.raises(ValueError, match="start"):
        grid.sequence(surface, [1, 2], start=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1-D"):
        grid.sequence(surface, [[1, 2]])
    with pytest.raises(ValueError, match="both"):
        grid.sequence(surface, [1, 2], width=0, radius=0)


@pytest.mark.parametrize(("dx", "dy"), [(9, 0), (-5, 11)])
def test_sequences_move_rigidly_when_panning(dx: int, dy: int) -> None:
    grid, _ = blank_grid(center=(0.37, 0.21), scale=31.0)

    def draw(surface: pygame.Surface) -> None:
        grid.sequence(surface, lambda n: np.sin(n) * 2, color=BLACK)
        grid.sequence(surface, [1.0, 2.0, 3.0], start=-4, color=BLACK)

    before = capture(grid, draw)
    grid.view.pan_pixels(dx, dy)
    after = capture(grid, draw)
    m = 30
    old = before[m - dy : SIZE[1] - m - dy, m - dx : SIZE[0] - m - dx]
    new = after[m : SIZE[1] - m, m : SIZE[0] - m]
    np.testing.assert_allclose(new, old, atol=2 / 255)
    assert before.max() > 0.9


# --- Axis titles and arrowheads -------------------------------------------------


def test_axis_titles_and_arrowheads_are_drawn() -> None:
    theme = pgp.GridTheme(show_minor=False, show_major=False, show_labels=False)
    grid = pgp.Grid(theme=theme)
    plain = capture(grid, lambda surface: None)
    surface = pygame.Surface(SIZE)
    grid.draw(surface, x_title="time", y_title="speed")
    titled = ink(surface)
    ax, ay = (round(v) for v in grid.to_screen(0.0, 0.0))
    # The x title, at the right end just above the x axis.
    assert (titled[ay - 20 : ay - 3, -60:] > plain[ay - 20 : ay - 3, -60:]).any()
    # The y title, at the top just right of the y axis.
    assert (titled[:20, ax + 3 : ax + 60] > plain[:20, ax + 3 : ax + 60]).any()
    assert grid.last_scene is not None
    assert [i.text.plain for i in grid.last_scene.labels] == ["time", "speed"]
    grid.theme = replace(theme, show_arrows=True)
    arrows = capture(grid, lambda surface: None)
    # The heads are wider than the axes near their ends.
    assert arrows[ay - 3, -13] > 0.8
    assert plain[ay - 3, -13] == 0.0
    assert arrows[12, ax + 2] > 0.8
    assert plain[12, ax + 2] == 0.0
    with pytest.raises(TypeError, match="x_title"):
        grid.draw(surface, x_title=3)  # type: ignore[arg-type]


def test_arrowheads_that_do_not_move_are_drawn_from_the_same_primitives() -> None:
    grid = pgp.Grid(theme=replace(pgp.LIGHT, show_arrows=True))
    surface = pygame.Surface(SIZE)
    grid.draw(surface)
    assert grid.last_scene is not None
    x_head, y_head = grid.last_scene.arrows
    grid.view.pan_pixels(7, 0)  # sideways: only the y axis moves
    grid.draw(surface)
    assert grid.last_scene.arrows[0] is x_head
    assert grid.last_scene.arrows[1] is not y_head
    assert not np.array_equal(grid.last_scene.arrows[1].rings[0], y_head.rings[0])


def test_changing_a_title_redraws_the_grid() -> None:
    grid = pgp.Grid()
    surface = pygame.Surface(SIZE)
    grid.draw(surface, x_title="a")
    first = grid.last_scene
    grid.draw(surface, x_title="a")
    assert grid.last_scene is first  # unchanged: reused
    grid.draw(surface, x_title="b")
    assert grid.last_scene is not first
    assert grid.last_scene is not None
    assert grid.last_scene.axis_labels.x_title[0].text.plain == "b"


# --- Text -------------------------------------------------------------------------


def test_text_is_written_at_its_point() -> None:
    grid, surface = blank_grid()
    grid.text(surface, "Hello", (1.0, 1.0), color=BLACK, halo=False)
    image = ink(surface)
    sx, sy = grid.to_screen(1.0, 1.0)
    rows, cols = np.nonzero(image > 0.5)
    assert abs((rows.min() + rows.max()) / 2 - sy) <= 3
    assert abs((cols.min() + cols.max()) / 2 - sx) <= 2
    grid2, surface2 = blank_grid()
    grid2.text(surface2, "Hello", (1.0, 1.0), anchor="left", gap=10, color=BLACK)
    cols2 = np.nonzero(ink(surface2) > 0.5)[1]
    assert sx + 10 <= cols2.min() <= sx + 13


def test_text_takes_no_palette_colour() -> None:
    grid, surface = blank_grid()
    grid.text(surface, "label", (0.0, 0.0))
    grid.plot(surface, lambda x: 0 * x + 2, width=4)
    row = round(grid.to_screen(0.0, 2.0)[1])
    color = pygame.surfarray.array3d(surface)[SIZE[0] // 2, row]
    assert tuple(color) == grid.theme.palette[0]  # the plot still gets the first


def test_text_sizes_styles_and_bad_arguments() -> None:
    grid, surface = blank_grid()
    small = capture(grid, lambda s: grid.text(s, "Mg", (0, 0), color=BLACK))
    big = capture(grid, lambda s: grid.text(s, "Mg", (0, 0), color=BLACK, size=24))
    assert (big > 0.5).sum() > 2 * (small > 0.5).sum()
    faint = capture(
        grid,
        lambda s: grid.text(
            s, "Mg", (0, 0), style=pgp.TextStyle(color=BLACK, opacity=0.3)
        ),
    )
    assert 0.1 < faint.max() < 0.5
    blank = capture(grid, lambda s: grid.text(s, "Mg", (np.nan, 0)))
    assert blank.max() == 0.0
    with pytest.raises(TypeError, match="string"):
        grid.text(surface, 3, (0, 0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one point"):
        grid.text(surface, "a", [(0, 0), (1, 1)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="anchor"):
        grid.text(surface, "a", (0, 0), anchor="middle")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gap"):
        grid.text(surface, "a", (0, 0), gap=-2)


# --- The legend ---------------------------------------------------------------


def legend_image(grid: pgp.Grid, surface: pygame.Surface) -> np.ndarray:
    """The pixels of the legend's panel, as last drawn."""
    panel = grid._legend_panel
    assert panel is not None
    rect = panel.rect
    pixels = pygame.surfarray.array3d(surface)
    return pixels[rect.left : rect.right, rect.top : rect.bottom]


def test_the_legend_shows_the_labelled_plots() -> None:
    grid, surface = blank_grid()
    grid.plot(surface, np.sin, color=(255, 0, 0), label="sin x")
    grid.plot(surface, np.cos, color=(0, 0, 255))  # no label: not listed
    grid.points(surface, [(1, 1)], color=(0, 160, 0), label="a point")
    grid.legend(surface)
    panel = grid._legend_panel
    assert panel is not None
    assert [row.entry.label for row in panel.rows] == ["sin x", "a point"]
    assert panel.rect.right == SIZE[0] - 10  # top right by default
    image = legend_image(grid, surface)
    assert ((image == (255, 0, 0)).all(axis=2)).any()  # the line's swatch
    assert ((image == (0, 160, 0)).all(axis=2)).any()  # the point's marker
    assert not ((image == (0, 0, 255)).all(axis=2)).any()


def test_legend_entries_last_one_frame_and_repeat_once() -> None:
    grid, surface = blank_grid()
    for _ in range(3):  # a plot drawn in pieces, each labelled
        grid.plot(surface, np.sin, color=(255, 0, 0), label="pieces")
    grid.legend(surface)
    assert grid._legend_panel is not None
    assert len(grid._legend_panel.rows) == 1
    grid.draw(surface)  # the next frame: nothing labelled yet
    grid.legend(surface)
    assert grid._legend_panel is None


def test_the_legend_is_drawn_from_the_same_primitives_while_unchanged() -> None:
    """The renderer reuses coverage by identity: a still legend is not rasterised."""
    grid, surface = blank_grid()

    def frame(label: str, color: tuple[int, int, int] = (0, 0, 200)) -> dict:
        grid.draw(surface)
        grid.plot(surface, np.sin, color=color, label=label)
        grid.points(surface, (0, 0), label="origin")
        grid.legend(surface, corner="top-left")
        return {name: shapes for name, (_, shapes) in grid._legend_drawings.items()}

    first = frame("sin x")
    assert set(first) == {"panel", 0, 1}
    again = frame("sin x")
    assert all(again[name] is first[name] for name in first)
    grid.view.pan_pixels(30, 10)  # the legend stays put while panning
    panned = frame("sin x")
    assert all(panned[name] is first[name] for name in first)
    # A longer label widens the panel; the swatches, at the left, stay.
    renamed = frame("the sine of x")
    assert renamed["panel"] is not first["panel"]
    assert renamed[0] is first[0]
    assert renamed[1] is first[1]
    recoloured = frame("the sine of x", (200, 0, 0))
    assert recoloured[0] is not first[0]
    assert recoloured[1] is first[1]


def test_labels_and_corners_are_validated() -> None:
    grid, surface = blank_grid()
    with pytest.raises(TypeError, match="label"):
        grid.plot(surface, np.sin, label=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one line"):
        grid.points(surface, [(0, 0)], label="a\nb")
    with pytest.raises(ValueError, match="corner"):
        grid.legend(surface, corner="middle")  # type: ignore[arg-type]


def test_every_plotting_method_takes_a_label() -> None:
    grid, surface = blank_grid()
    grid.plot(surface, np.sin, label="plot")
    grid.parametric(surface, np.cos, np.sin, (0, 1), label="parametric")
    grid.polar(surface, lambda t: 1 + 0 * t, label="polar")
    grid.implicit(surface, lambda x, y: x * y, 1, label="implicit")
    grid.fill_between(surface, np.sin, 0, label="fill_between")
    grid.inequality(surface, lambda x, y: y > x, label="inequality")
    grid.arrow(surface, (0, 0), (1, 1), label="arrow")
    grid.vector_field(surface, lambda x, y: (y, -x), label="vector_field")
    grid.slope_field(surface, lambda x, y: x, label="slope_field")
    grid.sequence(surface, [1, 2], label="sequence")
    grid.points(surface, (0, 0), label="points")
    grid.polyline(surface, [(0, 0), (1, 1)], label="polyline")
    grid.polygon(surface, [(0, 0), (1, 0), (0, 1)], label="polygon")
    grid.legend(surface)
    assert grid._legend_panel is not None
    names = [row.entry.label for row in grid._legend_panel.rows]
    assert names == [
        "plot",
        "parametric",
        "polar",
        "implicit",
        "fill_between",
        "inequality",
        "arrow",
        "vector_field",
        "slope_field",
        "sequence",
        "points",
        "polyline",
        "polygon",
    ]


# --- Data plots ----------------------------------------------------------------


def color_at(surface: pygame.Surface, grid: pgp.Grid, x: float, y: float) -> tuple:
    """The colour of the pixel at a world point."""
    px, py = grid.to_screen(x, y)
    return tuple(surface.get_at((round(px), round(py))))[:3]


def test_points_take_a_colour_and_size_each() -> None:
    grid, surface = blank_grid()
    colors = [(255, 0, 0), (0, 0, 255), (0, 160, 0)]
    grid.points(surface, [(-2, 0), (0, 0), (2, 0)], color=colors, radius=[3, 6, 0])
    assert color_at(surface, grid, -2, 0) == (255, 0, 0)
    assert color_at(surface, grid, 0, 0) == (0, 0, 255)
    assert color_at(surface, grid, 2, 0) == (255, 255, 255)  # size 0
    px, py = grid.to_screen(0, 0)
    assert tuple(surface.get_at((round(px) + 5, round(py))))[:3] == (0, 0, 255)
    px, py = grid.to_screen(-2, 0)
    assert tuple(surface.get_at((round(px) + 5, round(py))))[:3] == (255, 255, 255)
    # Colours per point use no palette colour: the next plot gets the first.
    grid.plot(surface, lambda x: 0 * x + 2, width=4)
    assert color_at(surface, grid, 0, 2) == grid.theme.palette[0]


def test_error_bars_are_drawn_under_the_markers() -> None:
    grid, surface = blank_grid()
    grid.points(surface, (1, 1), color=BLACK, y_error=[[1.0], [0.5]], x_error=0.6)
    image = ink(surface)
    px, py = (round(v) for v in grid.to_screen(1, 1))
    assert image[py + 40, px] > 0.5  # 0.8 below: inside the bar's 1 (50 px)
    assert image[py - 20, px] > 0.5  # 0.4 above: inside its 0.5
    assert image[py - 30, px] == 0.0  # 0.6 above: beyond it
    assert image[py, px + 25] > 0.5  # across x
    low_cap = round(grid.to_screen(1, 0)[1])
    assert image[low_cap, px + 3] > 0.5  # the cap, 8 px wide
    assert image[low_cap, px + 6] == 0.0
    grid2, surface2 = blank_grid()
    grid2.points(surface2, (1, 1), y_error=0.5, radius=0, color=BLACK)
    assert ink(surface2)[py, px] > 0.5  # the bar alone, through the point
    assert ink(surface2)[py, px + 3] == 0.0  # no marker


def test_steps_bars_and_histograms_are_drawn() -> None:
    grid, surface = blank_grid()
    grid.polyline(surface, [(-3, 0), (-1, 2)], step="post", color=BLACK, width=3)
    assert color_at(surface, grid, -2, 0) == BLACK  # holds at 0 until x = -1
    assert color_at(surface, grid, -1, 1) == BLACK  # then up
    assert color_at(surface, grid, -2, 1) == (255, 255, 255)
    grid.bars(surface, [(1, 2), (2, -1)], color=(0, 0, 200), opacity=1.0)
    assert color_at(surface, grid, 1, 1) == (0, 0, 200)
    assert color_at(surface, grid, 1.5, 1) == (255, 255, 255)  # a gap
    assert color_at(surface, grid, 2, -0.5) == (0, 0, 200)  # hangs down
    grid.histogram(surface, [0.1, 0.3], width=1.0, color=(0, 150, 0), opacity=1)
    assert color_at(surface, grid, 0.5, 1.5) == (0, 150, 0)  # two in [0, 1)
    assert color_at(surface, grid, 0.5, 2.2) == (255, 255, 255)


def test_data_plots_validate_their_arguments() -> None:
    grid, surface = blank_grid()
    with pytest.raises(ValueError, match="colours for"):
        grid.points(surface, [(0, 0), (1, 1)], color=["red"])
    with pytest.raises(ValueError, match="sizes"):
        grid.points(surface, [(0, 0)], radius=[1, 2])
    with pytest.raises(ValueError, match="y_error"):
        grid.points(surface, [(0, 0)], y_error=[1, 2, 3])
    with pytest.raises(ValueError, match="step"):
        grid.polyline(surface, [(0, 0), (1, 1)], step="later")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="closed"):
        grid.polyline(surface, [(0, 0), (1, 1)], step="post", closed=True)
    with pytest.raises(ValueError, match="width"):
        grid.bars(surface, [1, 2], width=0)
    with pytest.raises(ValueError, match="baseline"):
        grid.bars(surface, [1, 2], baseline=math.nan)
    with pytest.raises(ValueError, match="not both"):
        grid.histogram(surface, [1, 2], width=1, edges=[0, 3])
    with pytest.raises(ValueError, match="edges"):
        grid.histogram(surface, [1, 2], edges=[3, 0])


def test_data_plots_have_legend_entries() -> None:
    grid, surface = blank_grid()
    grid.points(surface, [(0, 0)], y_error=0.5, label="measured")
    grid.bars(surface, [1, 2], label="bars")
    grid.histogram(surface, [0.5, 1.5], label="histogram")
    grid.polyline(surface, [(0, 0), (1, 1)], step="post", label="steps")
    grid.legend(surface)
    assert grid._legend_panel is not None
    entries = [row.entry for row in grid._legend_panel.rows]
    assert [e.label for e in entries] == ["measured", "bars", "histogram", "steps"]
    assert entries[0].errors
    assert isinstance(entries[1].style, pgp.BarStyle)


def test_colour_bars_show_the_map_and_take_presses() -> None:
    grid, surface = blank_grid()
    grid.color_bar(surface, "gray", (0, 10), title="value")
    panel = grid._color_bar_panel
    assert panel is not None
    bar = panel.bar
    top = tuple(surface.get_at((bar.left + 3, bar.top + 1)))[:3]
    bottom = tuple(surface.get_at((bar.left + 3, bar.bottom - 2)))[:3]
    assert top[0] > 240  # white, the high end, at the top
    assert bottom[0] < 15
    assert panel.rect.right == SIZE[0] - 10  # top right by default
    first = grid._color_bar_drawing
    grid.draw(surface)
    grid.color_bar(surface, "gray", (0, 10), title="value")
    assert grid._color_bar_drawing is first  # unchanged: the same primitives
    assert Panel_rect_in_overlays(grid, panel.rect)
    with pytest.raises(ValueError, match="range"):
        grid.color_bar(surface, "gray", (1, 1))
    with pytest.raises(ValueError, match="colour map"):
        grid.color_bar(surface, "rainbow", (0, 1))
    with pytest.raises(ValueError, match="corner"):
        grid.color_bar(surface, "gray", (0, 1), corner="middle")  # type: ignore[arg-type]


def Panel_rect_in_overlays(grid: pgp.Grid, rect: object) -> bool:  # noqa: N802
    """Whether a panel is among those presses do not go through."""
    return any(panel.rect == rect for panel in grid._overlays())


def test_a_colour_bar_goes_beyond_the_legend_in_its_corner() -> None:
    grid, surface = blank_grid()
    grid.points(surface, (0, 0), label="points")
    grid.legend(surface)
    grid.color_bar(surface, "viridis", (0, 1))
    legend, bar = grid._legend_panel, grid._color_bar_panel
    assert legend is not None
    assert bar is not None
    assert bar.rect.top == legend.rect.bottom + 10
    grid.draw(surface)  # next frame, colour bar first: the legend as before
    grid.color_bar(surface, "viridis", (0, 1))
    assert grid._color_bar_panel is not None
    assert grid._color_bar_panel.rect.top == legend.rect.bottom + 10


def test_points_coloured_by_value() -> None:
    grid, surface = blank_grid()
    colors = pgp.color_map([0, 1], "gray", (0, 1))
    grid.points(surface, [(-1, 0), (1, 0)], color=colors, radius=5)
    assert color_at(surface, grid, -1, 0) == (0, 0, 0)
    assert color_at(surface, grid, 1, 0) == (255, 255, 255)


def test_horizontal_stacked_bars_and_cumulative_histograms() -> None:
    grid, surface = blank_grid()
    grid.bars(surface, [2.0], start=1, horizontal=True, color=BLACK, opacity=1)
    assert color_at(surface, grid, 1.5, 1) == BLACK  # along x at y = 1
    assert color_at(surface, grid, 1.5, 0.3) == (255, 255, 255)
    grid.bars(surface, [2.0], start=-2, baseline=[1.0], color=(0, 0, 200), opacity=1)
    assert color_at(surface, grid, -2, 1.5) == (0, 0, 200)  # from 1 to 2
    assert color_at(surface, grid, -2, 0.5) == (255, 255, 255)
    with pytest.raises(ValueError, match="values for"):
        grid.bars(surface, [1, 2], baseline=[0])
    grid2, surface2 = blank_grid()
    grid2.histogram(
        surface2, [0.1, 1.1], width=1, cumulative=True, color=BLACK, opacity=1
    )
    assert color_at(surface2, grid2, 1.5, 1.5) == BLACK  # 2 by the second bin
    assert color_at(surface2, grid2, 0.5, 1.5) == (255, 255, 255)


def test_box_and_violin_plots_are_drawn() -> None:
    grid, surface = blank_grid()
    groups = [np.linspace(0, 2, 41), np.linspace(-1, 1, 41)]
    grid.box_plot(surface, groups, start=-1, color=BLACK, opacity=1)
    assert color_at(surface, grid, -1, 1.0) == BLACK  # inside the first box
    assert color_at(surface, grid, -1.4, 1.0) == (255, 255, 255)  # beside it
    grid2, surface2 = blank_grid()
    grid2.violin_plot(surface2, groups, start=-1, color=BLACK, opacity=1, width=0.8)
    assert color_at(surface2, grid2, 0, 0) == BLACK  # the middle of the second
    with pytest.raises(ValueError, match="positions"):
        grid2.box_plot(surface2, groups, positions=[1])
    grid2.legend(surface2)


def test_axes_name_categories() -> None:
    grid = pgp.Grid(x_range=(-1, 4))
    surface = pygame.Surface(SIZE)
    grid.draw(surface, x_labels=["a", "b", "c"])
    assert grid.last_scene is not None
    assert [i.text.plain for i in grid.last_scene.axis_labels.x] == ["a", "b", "c"]
    first = grid.last_scene
    grid.draw(surface, x_labels=["a", "b", "c"])
    assert grid.last_scene is first  # unchanged: reused
    grid.draw(surface, x_labels={0: "a", 2: "z"})
    assert [i.text.plain for i in grid.last_scene.axis_labels.x] == ["a", "z"]
    with pytest.raises(TypeError, match="strings"):
        grid.draw(surface, y_labels=[1, 2])  # type: ignore[list-item]


# ----------------------------------------------------------------------
# Dashed and dotted lines
# ----------------------------------------------------------------------
def draw_dashed(grid: pgp.Grid, surface: pygame.Surface, dash: object) -> None:
    """Draw one of every kind of line, dashed."""
    grid.plot(surface, lambda x: np.sin(2 * x) + 0.2 * x, color=BLACK, dash=dash)
    grid.parametric(
        surface, np.cos, lambda t: np.sin(2 * t), (0, 2 * np.pi), color=BLACK, dash=dash
    )
    grid.polar(surface, 1.5, color=BLACK, dash=dash)
    grid.implicit(surface, lambda x, y: x * y, 1, color=BLACK, dash=dash)
    grid.polyline(surface, [(-3, 1), (-2, 2), (-1, 1.5)], color=BLACK, dash=dash)
    grid.polygon(surface, [(1, -2), (3, -2.5), (2, -1)], color=BLACK, dash=dash)


@pytest.mark.parametrize(("dx", "dy"), [(13, 0), (0, -9), (-21, 17)])
@pytest.mark.parametrize("dash", ["dashed", "dotted", (9, 3, 0, 3)])
def test_dashes_move_rigidly_when_panning(dx: int, dy: int, dash: object) -> None:
    """Panning by whole pixels moves every dash with its line."""
    grid, _ = blank_grid(center=(0.31, -0.17), scale=37.0)
    before = capture(grid, lambda s: draw_dashed(grid, s, dash))
    grid.view.pan_pixels(dx, dy)
    after = capture(grid, lambda s: draw_dashed(grid, s, dash))
    m = 30
    old = before[m - dy : SIZE[1] - m - dy, m - dx : SIZE[0] - m - dx]
    new = after[m : SIZE[1] - m, m : SIZE[0] - m]
    np.testing.assert_allclose(new, old, atol=2 / 255)


def test_dashed_lines_have_gaps() -> None:
    grid, _ = blank_grid()
    for kind in ("plot", "polyline", "polygon"):

        def draw(surface: pygame.Surface, dash: str, kind: str = kind) -> None:
            if kind == "plot":
                grid.plot(surface, lambda x: 0 * x + 1, color=BLACK, dash=dash)
            elif kind == "polyline":
                grid.polyline(surface, [(-2, 1), (2, 1)], color=BLACK, dash=dash)
            else:
                square = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
                grid.polygon(surface, square, color=BLACK, fill=False, dash=dash)

        solid = capture(grid, lambda s: draw(s, "solid")).sum()
        dashed = capture(grid, lambda s: draw(s, "dashed")).sum()
        dotted = capture(grid, lambda s: draw(s, "dotted")).sum()
        assert 0.45 * solid < dashed < 0.75 * solid, kind
        assert 0.2 * solid < dotted < 0.5 * solid, kind


def test_a_dashed_style_can_be_drawn_solid_per_call() -> None:
    grid, _ = blank_grid()
    style = pgp.LineStyle(dash="dashed", color=BLACK)
    solid = capture(grid, lambda s: grid.plot(s, np.sin, color=BLACK)).sum()
    reused = capture(grid, lambda s: grid.plot(s, np.sin, style=style, dash="solid"))
    assert reused.sum() == pytest.approx(solid)


def test_slope_fields_cannot_be_dashed() -> None:
    grid, surface = blank_grid()
    with pytest.raises(ValueError, match="solid"):
        grid.slope_field(surface, lambda x, y: x, style=pgp.LineStyle(dash="dashed"))


def test_dashed_plots_have_dashed_legend_swatches() -> None:
    grid, _ = blank_grid()

    def legend(dash: str) -> np.ndarray:
        def draw(surface: pygame.Surface) -> None:
            grid.plot(surface, np.sin, color=BLACK, dash=dash, label="sine")
            grid.legend(surface)

        return capture(grid, draw)

    corner = (slice(0, 40), slice(SIZE[0] - 120, SIZE[0]))
    assert legend("dashed")[corner].sum() < legend("solid")[corner].sum()
