"""Render a gallery of tricky grid and plot views to PNG files for visual review.

This is a development tool. It runs headless (no window), renders each view
in :data:`VIEWS` to ``<out>/<name>.png``, and also writes a contact sheet
``<out>/_sheet.png`` with every view at reduced size.

Compare galleries from before and after a change to catch regressions by
eye::

    python scripts/render_gallery.py --out renders/before
    # ... change something ...
    python scripts/render_gallery.py --out renders/after
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

import pygridplot as pgp

Scene = Callable[[pgp.Plane, pygame.Surface], None]
"""Draws plots on top of a view's grid."""


def scene_test_set(plane: pgp.Plane, surface: pygame.Surface) -> None:
    """Plot the shared function test set from the project plan.

    Args:
        plane: The plane.
        surface: The surface to draw on.
    """
    for func in (lambda x: 1 / x, np.tan, np.sqrt, np.log, np.floor, lambda x: x**2):
        plane.plot(surface, func)


def scene_hard(plane: pgp.Plane, surface: pygame.Surface) -> None:
    """Plot functions that are hard to sample well.

    Args:
        plane: The plane.
        surface: The surface to draw on.
    """
    plane.plot(surface, math.gamma)
    plane.plot(surface, lambda x: np.sin(x) / x)
    plane.plot(surface, lambda x: np.sin(1 / x))
    plane.plot(surface, lambda x: np.sqrt(np.sin(x)))


def scene_shapes(plane: pgp.Plane, surface: pygame.Surface) -> None:
    """Draw every marker shape, polylines and polygons (both fill rules).

    Args:
        plane: The plane.
        surface: The surface to draw on.
    """
    for i, marker in enumerate(("circle", "square", "diamond", "cross", "plus")):
        x = -6 + 1.3 * i
        plane.points(surface, (x, 4.0), marker=marker, radius=6)  # type: ignore[arg-type]
        plane.points(
            surface,
            (x, 3.0),
            marker=marker,
            radius=6,
            edge_color="white",  # type: ignore[arg-type]
        )
    angles = np.pi / 2 + np.arange(5) * 4 * np.pi / 5
    star = np.column_stack((2 * np.cos(angles), 2 * np.sin(angles)))
    plane.polygon(surface, star + np.array([3.5, 2.5]), fill_rule="nonzero")
    plane.polygon(surface, star + np.array([7.0, 2.5]), fill_rule="evenodd")
    plane.polygon(surface, [(-6, -1), (-3, -1), (-4.5, 1.5)])
    theta = np.linspace(0, 6 * np.pi, 300)
    spiral = np.column_stack(
        (0.12 * theta * np.cos(theta), 0.12 * theta * np.sin(theta))
    )
    plane.polyline(surface, spiral + np.array([2.0, -2.5]), width=2)
    xs = np.linspace(-7, 7, 50)
    plane.points(surface, np.column_stack((xs, -4.5 + 0.4 * np.sin(xs))), radius=3)


SCENES: dict[str, Scene] = {
    "test_set": scene_test_set,
    "hard": scene_hard,
    "shapes": scene_shapes,
}


@dataclass(frozen=True)
class GalleryView:
    """One view to render.

    Attributes:
        name: File name (without extension).
        center: World point at the middle of the plane.
        scale: Pixels per world unit (one number, or ``(x, y)``).
        theme: Theme name: ``"light"`` or ``"dark"``.
        overrides: Theme fields to override.
        embedded: Draw the plane in a sub-rectangle of the surface.
        scene: Name of the plots to draw on top (see :data:`SCENES`), if any.
    """

    name: str
    center: tuple[float, float] = (0.0, 0.0)
    scale: float | tuple[float, float] = 50.0
    theme: str = "light"
    overrides: dict[str, object] = field(default_factory=dict)
    embedded: bool = False
    scene: str | None = None


VIEWS: tuple[GalleryView, ...] = (
    GalleryView("01_default"),
    GalleryView("02_zoom_in", center=(0.3, 0.2), scale=5e4),
    GalleryView("03_zoom_in_sci", center=(0.0, 0.0), scale=2e7),
    GalleryView("04_zoom_out", scale=0.5),
    GalleryView("05_zoom_out_sci", scale=2e-5),
    GalleryView("06_axis_near_bottom", center=(0.0, 5.6)),
    GalleryView("07_axis_below_screen", center=(3.0, 40.0)),
    GalleryView("08_axis_near_left", center=(8.7, 0.0)),
    GalleryView("09_axes_off_corner", center=(-40.0, -30.0)),
    GalleryView("10_far_offset_deep", center=(12345.678, -98.7654), scale=1e7),
    GalleryView("11_fade_1_to_2", scale=100 / math.sqrt(2)),
    GalleryView("12_fade_2_to_5", scale=100 / math.sqrt(10)),
    GalleryView("13_dark", theme="dark"),
    GalleryView("14_embedded", embedded=True, center=(2.0, 1.0)),
    GalleryView("15_unequal_aspect", scale=(50.0, 5000.0), center=(0.0, 0.03)),
    GalleryView("16_snap_mode", overrides={"transition": "snap"}),
    GalleryView("17_big_negative", center=(-2.5e6, -1.5e6), scale=2e-4),
    GalleryView("18_axis_near_top_right", center=(-8.8, -5.9)),
    GalleryView("19_functions", scale=60.0, scene="test_set"),
    GalleryView("20_functions_dark", scale=60.0, theme="dark", scene="test_set"),
    GalleryView("21_hard_functions", scale=60.0, scene="hard"),
    GalleryView("22_zoomed_function", center=(1.0, 0.8415), scale=4e5, scene="hard"),
    GalleryView("23_shapes", scale=55.0, scene="shapes"),
    GalleryView("24_shapes_embedded", scale=40.0, embedded=True, scene="shapes"),
)


def theme_for(view: GalleryView) -> pgp.GridTheme:
    """Return the theme for a gallery view, with its overrides applied.

    Args:
        view: The gallery view.

    Returns:
        The theme to render with.
    """
    base = pgp.DARK if view.theme == "dark" else pgp.LIGHT
    return replace(base, **view.overrides)  # type: ignore[arg-type]


def render_view(view: GalleryView, size: tuple[int, int]) -> pygame.Surface:
    """Render one gallery view to a new surface.

    Args:
        view: The view to render.
        size: Surface size in pixels.

    Returns:
        The rendered surface.
    """
    surface = pygame.Surface(size)
    surface.fill((90, 96, 110))
    rect = None
    if view.embedded:
        rect = pygame.Rect(220, 40, size[0] - 260, size[1] - 80)
    plane = pgp.Plane(
        rect,
        center=view.center,
        scale=view.scale,
        equal_aspect=not isinstance(view.scale, tuple),
        theme=theme_for(view),
    )
    plane.draw_grid(surface)
    if view.scene is not None:
        SCENES[view.scene](plane, surface)
    return surface


def contact_sheet(
    images: list[tuple[str, pygame.Surface]], columns: int
) -> pygame.Surface:
    """Arrange half-size thumbnails of the images in a grid.

    Args:
        images: ``(name, surface)`` pairs.
        columns: Thumbnails per row.

    Returns:
        The contact sheet.
    """
    thumb_w, thumb_h = (s // 2 for s in images[0][1].get_size())
    rows = math.ceil(len(images) / columns)
    sheet = pygame.Surface((columns * (thumb_w + 8) + 8, rows * (thumb_h + 24) + 8))
    sheet.fill((60, 60, 60))
    font = pygame.font.Font(None, 20)
    for index, (name, image) in enumerate(images):
        col, row = index % columns, index // columns
        x = 8 + col * (thumb_w + 8)
        y = 8 + row * (thumb_h + 24)
        sheet.blit(font.render(name, True, (230, 230, 230)), (x, y))
        sheet.blit(pygame.transform.smoothscale(image, (thumb_w, thumb_h)), (x, y + 16))
    return sheet


def main() -> None:
    """Render the gallery as configured on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("renders/gallery"))
    parser.add_argument("--size", default="900x600", help="surface size, WxH")
    parser.add_argument("--only", nargs="*", help="render only these view names")
    args = parser.parse_args()
    size = tuple(int(v) for v in args.size.lower().split("x"))
    args.out.mkdir(parents=True, exist_ok=True)

    pygame.init()
    pygame.display.set_mode((1, 1))
    rendered = []
    for view in VIEWS:
        if args.only and view.name not in args.only:
            continue
        image = render_view(view, size)  # type: ignore[arg-type]
        pygame.image.save(image, args.out / f"{view.name}.png")
        rendered.append((view.name, image))
    if rendered:
        pygame.image.save(contact_sheet(rendered, 3), args.out / "_sheet.png")
    print(f"rendered {len(rendered)} views to {args.out}")
    pygame.quit()


if __name__ == "__main__":
    main()
