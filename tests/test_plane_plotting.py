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


def test_parametric_and_polar_curves_draw_through_the_right_pixels() -> None:
    plane, surface = blank_plane()
    plane.parametric(surface, np.cos, np.sin, (0, 2 * np.pi), color=BLACK)
    plane.polar(surface, 2.0, color=BLACK)
    painted = ink(surface)
    cx, cy = (round(v) for v in plane.to_screen(0.0, 0.0))
    assert painted[cy, cx + 50] > 0.9  # the unit circle (50 px per unit)
    assert painted[cy - 100, cx] > 0.9  # the circle r = 2, at the top
    assert painted[cy, cx + 75] < 0.05  # nothing in between


def test_curves_take_palette_colours_and_styles() -> None:
    plane, surface = blank_plane()
    plane.parametric(surface, np.cos, np.sin, (0, 2 * np.pi))
    plane.polar(surface, 2.0, style=pgp.LineStyle(width=4.0))
    cx, cy = (round(v) for v in plane.to_screen(0.0, 0.0))
    first, second = plane.theme.palette[:2]
    assert tuple(surface.get_at((cx + 50, cy)))[:3] == first
    assert tuple(surface.get_at((cx + 100, cy)))[:3] == second
    assert tuple(surface.get_at((cx + 101, cy)))[:3] == second  # 4 px wide


@pytest.mark.parametrize(("dx", "dy"), [(7, -3), (-12, 5)])
def test_curves_move_rigidly_when_panning(dx: int, dy: int) -> None:
    plane, surface = blank_plane()
    rose = lambda t: 2 * np.cos(3 * t)  # noqa: E731
    plane.polar(surface, rose, color=BLACK)
    before = ink(surface)
    plane.view.pan_pixels(dx, dy)
    plane.draw_grid(surface)
    plane.polar(surface, rose, color=BLACK)
    after = ink(surface)
    inner = (slice(40, 200), slice(40, 280))
    shifted = np.roll(np.roll(before, dy, axis=0), dx, axis=1)
    # As for function plots, float round-off may change a pixel by a level.
    np.testing.assert_allclose(after[inner], shifted[inner], atol=2 / 255)
    assert before.max() > 0.9  # the curve was drawn at all


def test_implicit_curves_draw_through_the_right_pixels() -> None:
    plane, surface = blank_plane()
    plane.implicit(surface, lambda x, y: x**2 + y**2, 4, color=BLACK)  # r = 2
    plane.implicit(surface, lambda x, y: y - np.tan(x), color=BLACK)
    painted = ink(surface)
    cx, cy = (round(v) for v in plane.to_screen(0.0, 0.0))
    assert painted[cy - 100, cx] > 0.9  # the circle, at the top
    assert painted[cy, cx - 100] > 0.9  # and at the left
    assert painted[cy, cx] > 0.9  # tan passes through the origin
    # No line is drawn down the asymptote of tan at x = pi / 2.
    only_tan = capture(
        plane, lambda s: plane.implicit(s, lambda x, y: y - np.tan(x), color=BLACK)
    )
    column = round(plane.to_screen(math.pi / 2, 0.0)[0])
    assert only_tan[:, column].max() < 0.05
    # The branch leaves the top (y = 2.4) at x = arctan(2.4), 20 px before.
    assert only_tan[:, column - 25].max() > 0.9


def test_implicit_levels_draw_contour_lines_in_one_colour() -> None:
    plane, surface = blank_plane()
    plane.implicit(surface, lambda x, y: x**2 + y**2, [1, 4])
    first = plane.theme.palette[0]
    cx, cy = (round(v) for v in plane.to_screen(0.0, 0.0))
    assert tuple(surface.get_at((cx + 50, cy)))[:3] == first  # r = 1
    assert tuple(surface.get_at((cx + 100, cy)))[:3] == first  # r = 2


def test_implicit_curves_take_palette_colours_and_styles() -> None:
    plane, surface = blank_plane()
    plane.implicit(surface, lambda x, y: x - 1)
    plane.implicit(surface, lambda x, y: x + 1, style=pgp.LineStyle(width=4.0))
    first, second = plane.theme.palette[:2]
    row = SIZE[1] // 3
    one, minus_one = (round(plane.to_screen(x, 0.0)[0]) for x in (1.0, -1.0))
    assert tuple(surface.get_at((one, row)))[:3] == first
    assert tuple(surface.get_at((minus_one, row)))[:3] == second
    assert tuple(surface.get_at((minus_one + 1, row)))[:3] == second  # 4 px wide


def test_implicit_curves_survive_broken_functions() -> None:
    plane, surface = blank_plane()

    def broken(x: object, y: object) -> float:
        raise ZeroDivisionError

    plane.implicit(surface, broken)
    plane.implicit(surface, lambda x, y: "not a number")
    plane.implicit(surface, lambda x, y: math.log(x) - y)  # scalar-only, x > 0
    assert ink(surface)[:, : SIZE[0] // 2 - 2].max() == 0.0  # nothing left of 0
    with pytest.raises(ValueError, match="level"):
        plane.implicit(surface, lambda x, y: x, math.inf)


@pytest.mark.parametrize(("dx", "dy"), [(7, -3), (-12, 5)])
def test_implicit_curves_move_rigidly_when_panning(dx: int, dy: int) -> None:
    plane, _ = blank_plane(center=(0.13, 0.29))
    heart = lambda x, y: (x**2 + y**2 - 1) ** 3 - x**2 * y**3  # noqa: E731
    before = capture(plane, lambda s: plane.implicit(s, heart, color=BLACK))
    plane.view.pan_pixels(dx, dy)
    after = capture(plane, lambda s: plane.implicit(s, heart, color=BLACK))
    inner = (slice(40, 200), slice(40, 280))
    shifted = np.roll(np.roll(before, dy, axis=0), dx, axis=1)
    np.testing.assert_allclose(after[inner], shifted[inner], atol=2 / 255)
    assert before.max() > 0.9  # the curve was drawn at all


# ----------------------------------------------------------------------
# Areas between graphs, and regions
# ----------------------------------------------------------------------
def test_fill_between_shades_between_the_graphs() -> None:
    plane, surface = blank_plane()
    plane.fill_between(surface, 1, -1, color=BLACK, opacity=0.5)
    image = ink(surface)
    top = round(plane.to_screen(0.0, 1.0)[1])
    bottom = round(plane.to_screen(0.0, -1.0)[1])
    assert np.allclose(image[top + 2 : bottom - 2, :], 0.5, atol=2 / 255)
    assert image[: top - 1, :].max() == 0.0
    assert image[bottom + 2 :, :].max() == 0.0


def test_fill_between_defaults_to_the_x_axis() -> None:
    plane, surface = blank_plane()
    plane.fill_between(surface, lambda x: 0 * x + 1, color=BLACK)
    image = ink(surface)
    zero = round(plane.to_screen(0.0, 0.0)[1])
    one = round(plane.to_screen(0.0, 1.0)[1])
    assert image[one + 2 : zero - 1, :].min() > 0.2
    assert image[zero + 2 :, :].max() == 0.0


def test_fill_between_shades_both_sides_of_a_crossing() -> None:
    plane, surface = blank_plane()
    plane.fill_between(surface, lambda x: x, lambda x: -x, color=BLACK)
    image = ink(surface)
    cx, cy = (round(v) for v in plane.to_screen(0.0, 0.0))
    assert image[cy, cx - 40] > 0.2  # left lobe
    assert image[cy, cx + 40] > 0.2  # right lobe
    assert image[cy - 40, cx] == 0.0  # not between the lobes
    assert image[cy + 40, cx] == 0.0


def test_fill_between_does_not_join_jumps() -> None:
    plane, surface = blank_plane()
    plane.fill_between(surface, np.tan, color=BLACK)
    image = ink(surface)
    pole = round(plane.to_screen(math.pi / 2, 0.0)[0])
    cy = round(plane.to_screen(0.0, 0.0)[1])
    # Left of the pole tan is large and positive; right of it, negative.
    assert image[cy - 60, pole - 4] > 0.2
    assert image[cy + 60, pole - 4] == 0.0
    assert image[cy + 60, pole + 4] > 0.2
    assert image[cy - 60, pole + 4] == 0.0


def test_fill_between_survives_broken_functions() -> None:
    plane, surface = blank_plane()

    def broken(x: object) -> float:
        raise ZeroDivisionError

    plane.fill_between(surface, broken, 0)
    plane.fill_between(surface, np.sin, lambda x: "not a number")
    assert ink(surface).max() == 0.0
    plane.fill_between(surface, math.sqrt, 0, color=BLACK)  # scalar-only, x >= 0
    image = ink(surface)
    assert image[:, : SIZE[0] // 2 - 1].max() == 0.0
    assert image.max() > 0.2
    with pytest.raises(TypeError, match="number"):
        plane.fill_between(surface, np.sin, "0")  # type: ignore[arg-type]


def test_fill_between_on_log_axes_reaches_the_bottom_for_zero() -> None:
    plane, surface = blank_plane(log_x=True, log_y=True)
    plane.fill_between(surface, lambda x: x, 0, domain=(1, 10), color=BLACK)
    image = ink(surface)
    column = round(plane.to_screen(3.0, 1.0)[0])
    row = round(plane.to_screen(3.0, 3.0)[1])
    assert image[row + 2 :, column].min() > 0.2  # down to the bottom edge
    assert image[: row - 2, column].max() == 0.0


def test_inequality_shades_where_the_condition_holds() -> None:
    plane, surface = blank_plane()
    plane.inequality(
        surface, lambda x, y: (x**2 + y**2 > 1) & (x**2 + y**2 < 4), color=BLACK
    )
    image = ink(surface)
    cx, cy = (round(v) for v in plane.to_screen(0.0, 0.0))
    assert image[cy, cx] == 0.0  # the hole
    assert image[cy, cx + 75] > 0.2  # the ring, r = 1.5
    assert image[cy - 75, cx] > 0.2
    assert image[cy, cx + 105] == 0.0  # outside, r = 2.1


def test_regions_have_no_seams() -> None:
    """The parts of a large region are filled as one: every pixel alike."""
    plane, surface = blank_plane()
    plane.inequality(surface, lambda x, y: x > 0.3 * np.sin(3 * y), color=BLACK)
    image = ink(surface)
    right = image[:, SIZE[0] // 2 + 20 :]
    assert np.ptp(right) == 0.0
    assert right.min() > 0.2


def test_inequalities_take_palette_colours_and_styles() -> None:
    plane, surface = blank_plane()
    plane.inequality(surface, lambda x, y: x < -1)
    plane.inequality(surface, lambda x, y: x > 1, style=pgp.FillStyle(opacity=1.0))
    first, second = plane.theme.palette[:2]
    cy = SIZE[1] // 2
    left = tuple(surface.get_at((5, cy)))[:3]
    assert left != (255, 255, 255)
    assert left != first  # translucent, by default
    assert tuple(surface.get_at((SIZE[0] - 5, cy)))[:3] == second
    plane.fill_between(surface, 1.5, 2, color="red", opacity=1.0)
    top = round(plane.to_screen(0.0, 1.75)[1])
    assert tuple(surface.get_at((SIZE[0] // 2, top)))[:3] == (255, 0, 0)


def test_inequalities_survive_broken_conditions() -> None:
    plane, surface = blank_plane()

    def broken(x: object, y: object) -> bool:
        raise ZeroDivisionError

    plane.inequality(surface, broken)
    plane.inequality(surface, lambda x, y: "not a number")
    assert ink(surface).max() == 0.0
    # Python's `and` works on single numbers (more slowly).
    plane.inequality(surface, lambda x, y: x > 0 and y > 0, color=BLACK)
    image = ink(surface)
    cx, cy = (round(v) for v in plane.to_screen(0.0, 0.0))
    assert image[cy - 20, cx + 20] > 0.2
    assert image[cy + 20, cx + 20] == 0.0
    assert image[cy - 20, cx - 20] == 0.0


@pytest.mark.parametrize(("dx", "dy"), [(7, -3), (-12, 5)])
def test_areas_move_rigidly_when_panning(dx: int, dy: int) -> None:
    plane, _ = blank_plane(center=(0.13, 0.29))

    def draw(surface: pygame.Surface) -> None:
        plane.inequality(surface, lambda x, y: x**2 + 2 * y**2 < 3, color=BLACK)
        plane.fill_between(surface, np.sin, lambda x: 0.5 * x - 1, color=BLACK)

    before = capture(plane, draw)
    plane.view.pan_pixels(dx, dy)
    after = capture(plane, draw)
    inner = (slice(40, 200), slice(40, 280))
    shifted = np.roll(np.roll(before, dy, axis=0), dx, axis=1)
    np.testing.assert_allclose(after[inner], shifted[inner], atol=2 / 255)
    assert before.max() > 0.2


# ----------------------------------------------------------------------
# Arrows and fields
# ----------------------------------------------------------------------
def test_arrows_draw_their_shaft_and_head() -> None:
    plane, surface = blank_plane()
    plane.arrow(surface, (-2, 0), (2, 0), color=BLACK, width=3, head_size=12)
    image = ink(surface)
    cx, cy = (round(v) for v in plane.to_screen(0.0, 0.0))
    tip = round(plane.to_screen(2.0, 0.0)[0])
    assert image[cy, cx - 90 : tip - 2].min() > 0.9  # the shaft, up to the tip
    assert image[cy - 3, tip - 10] > 0.9  # the head is wider than the shaft
    assert image[cy - 3, cx] == 0.0
    assert image[cy, tip + 2 :].max() == 0.0  # nothing beyond the tip


def test_translucent_arrows_have_no_darker_joins() -> None:
    plane, surface = blank_plane()
    plane.arrow(surface, (-2, 0), (2, 0), color=BLACK, width=5, opacity=0.5)
    image = ink(surface)
    cy = round(plane.to_screen(0.0, 0.0)[1])
    tip = round(plane.to_screen(2.0, 0.0)[0])
    inside = image[cy, 80 : tip - 4]  # along the shaft and into the head
    assert np.ptp(inside) < 2 / 255
    assert inside.min() == pytest.approx(0.5, abs=2 / 255)


def test_many_arrows_take_one_colour_and_style() -> None:
    plane, surface = blank_plane()
    plane.arrow(surface, (0, 0), [(2, 0), (-2, 0), (0, 2)])
    style = pgp.ArrowStyle(color="red", width=4.0)
    plane.arrow(surface, (0, -1), (2, -1), style=style)
    first = plane.theme.palette[0]
    cx, cy = (round(v) for v in plane.to_screen(0.0, 0.0))
    assert tuple(surface.get_at((cx + 50, cy)))[:3] == first
    assert tuple(surface.get_at((cx - 50, cy)))[:3] == first
    assert tuple(surface.get_at((cx, cy - 50)))[:3] == first
    row = round(plane.to_screen(0.0, -1.0)[1])
    assert tuple(surface.get_at((cx + 50, row)))[:3] == (255, 0, 0)
    assert tuple(surface.get_at((cx + 50, row + 1)))[:3] == (255, 0, 0)  # 4 px wide
    with pytest.raises(ValueError, match="starts and"):
        plane.arrow(surface, [(0, 0), (1, 1)], [(1, 0), (2, 0), (3, 0)])


def test_vector_fields_draw_a_lattice_of_arrows() -> None:
    plane, surface = blank_plane()  # 50 px per unit: arrows 1 unit apart
    plane.vector_field(surface, lambda x, y: (1 + 0 * x, 0 * y), color=BLACK)
    image = ink(surface)
    cx, cy = (round(v) for v in plane.to_screen(0.0, 0.0))
    for row in (cy - 50, cy, cy + 50):
        assert image[row, cx - 12 : cx + 12].min() > 0.9  # a horizontal arrow
    assert image[cy - 25, :].max() == 0.0  # and nothing between the rows
    assert image[:, cx + 25].max() == 0.0  # nor between the columns


def test_slope_fields_draw_short_segments() -> None:
    plane, surface = blank_plane()
    plane.slope_field(surface, lambda x, y: 0 * x + 0 * y, color=BLACK)
    image = ink(surface)
    cx, cy = (round(v) for v in plane.to_screen(0.0, 0.0))
    assert image[cy, cx - 10 : cx + 10].min() > 0.9  # a level segment at the origin
    assert image[cy, cx + 14 : cx + 36].max() == 0.0  # and a gap to the next


def test_fields_survive_broken_functions() -> None:
    plane, surface = blank_plane()

    def broken(x: object, y: object) -> object:
        raise ZeroDivisionError

    plane.vector_field(surface, broken)
    plane.vector_field(surface, lambda x, y: "not a vector")
    plane.slope_field(surface, broken)
    assert ink(surface).max() == 0.0
    plane.vector_field(surface, lambda x, y: (math.sqrt(x), 0.0), color=BLACK)
    assert ink(surface)[:, : SIZE[0] // 2 - 25].max() == 0.0  # sqrt needs x >= 0
    assert ink(surface).max() > 0.5
    with pytest.raises(ValueError, match="spacing"):
        plane.vector_field(surface, lambda x, y: (x, y), spacing=1)
    with pytest.raises(ValueError, match="scale"):
        plane.vector_field(surface, lambda x, y: (x, y), scale=-2)


@pytest.mark.parametrize(("dx", "dy"), [(7, -3), (-12, 5)])
def test_fields_move_rigidly_when_panning(dx: int, dy: int) -> None:
    plane, _ = blank_plane(center=(0.13, 0.29), scale=63.0)

    def draw(surface: pygame.Surface) -> None:
        plane.vector_field(surface, lambda x, y: (np.sin(y), np.cos(x)), color=BLACK)
        plane.slope_field(surface, lambda x, y: x * y, color=BLACK, spacing=55)

    before = capture(plane, draw)
    plane.view.pan_pixels(dx, dy)
    after = capture(plane, draw)
    inner = (slice(40, 200), slice(40, 280))
    shifted = np.roll(np.roll(before, dy, axis=0), dx, axis=1)
    np.testing.assert_allclose(after[inner], shifted[inner], atol=2 / 255)
    assert before.max() > 0.5
