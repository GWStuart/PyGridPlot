"""Scaling primitives onto a picture with more (or fewer) pixels: for export.

A frame of a plane is drawn as backend-neutral primitives in screen pixels
(see :mod:`pygridplot.core.primitives`). Saving it at twice the
resolution is drawing the same primitives again onto a surface twice as
wide and twice as tall, with every position, width, radius and font size
doubled: the same picture, sharper. :class:`Scaling` maps the plane's
rectangle on screen onto such a picture, and the ``scale_*`` functions
return scaled copies of each kind of primitive.

Positions are continuous pixel coordinates (integers are pixel centres,
see :mod:`pygridplot.core.pixels`), so pixel *edges* are what scale: the
plane's left edge, at ``left - 0.5``, goes to ``-0.5``, the left edge of
the picture, and a line one pixel wide at pixel ``p`` becomes one two
pixels wide covering pixels ``2p`` and ``2p + 1``, as crisp as before.

Text is drawn with the font at the scaled size, which is not exactly the
scaled width of the original (fonts are hinted for each size), so each
label is placed by the centre of its box, which keeps tick labels centred
on their grid lines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.primitives import (
    Fill,
    FontSpec,
    Image,
    LineBatch,
    Markers,
    Orientation,
    Stroke,
    TextItem,
    TextMeasurer,
)

MAX_SIDE_PX: Final = 16384
"""The largest picture side, in pixels, a plane can be exported at."""


@dataclass(frozen=True, slots=True)
class Scaling:
    """Maps a plane's rectangle on screen onto a picture ``factor`` times as big.

    Attributes:
        factor: How many picture pixels one screen pixel becomes (along each
            side). Positive; below 1 makes a smaller picture.
        left: Screen x of the plane's leftmost column of pixels.
        top: Screen y of its top row.
    """

    factor: float
    left: float = 0.0
    top: float = 0.0

    def x(self, px: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """Map continuous screen x coordinates to the picture."""
        return (px - self.left) * self.factor + 0.5 * (self.factor - 1.0)

    def y(self, py: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """Map continuous screen y coordinates to the picture."""
        return (py - self.top) * self.factor + 0.5 * (self.factor - 1.0)

    def points(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map points to the picture: any array whose last axis is ``(x, y)``."""
        out = np.empty_like(points, dtype=np.float64)
        out[..., 0] = self.x(points[..., 0])
        out[..., 1] = self.y(points[..., 1])
        return out

    def size(self, width: int, height: int) -> tuple[int, int]:
        """Return the picture's size for a plane ``width`` by ``height`` pixels.

        Raises:
            ValueError: If a side would be larger than :data:`MAX_SIDE_PX`.
        """
        w = max(1, round(width * self.factor))
        h = max(1, round(height * self.factor))
        if max(w, h) > MAX_SIDE_PX:
            msg = (
                f"a picture {w} × {h} pixels is too large: the most is {MAX_SIDE_PX} "
                "pixels on a side"
            )
            raise ValueError(msg)
        return w, h


def check_scale(scale: object) -> float:
    """Validate an export scale.

    Args:
        scale: How many times as many pixels along each side.

    Returns:
        The scale as a float.

    Raises:
        TypeError: If it is not a number.
        ValueError: If it is not positive and finite.
    """
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        msg = f"scale must be a number, got {scale!r}"
        raise TypeError(msg)
    value = float(scale)
    if not (math.isfinite(value) and value > 0):
        msg = f"scale must be positive and finite, got {scale!r}"
        raise ValueError(msg)
    return value


def scale_stroke(stroke: Stroke, scaling: Scaling) -> Stroke:
    """Return a stroke on the scaled picture: its paths moved, its pen widened."""
    paths: tuple[NDArray[np.float64], ...] | NDArray[np.float64]
    if isinstance(stroke.paths, np.ndarray):
        paths = scaling.points(stroke.paths)
    else:
        paths = tuple(scaling.points(path) for path in stroke.paths)
    return replace(stroke, paths=paths, width=stroke.width * scaling.factor)


def scale_fill(fill: Fill, scaling: Scaling) -> Fill:
    """Return a fill on the scaled picture."""
    return replace(fill, rings=tuple(scaling.points(ring) for ring in fill.rings))


def scale_markers(markers: Markers, scaling: Scaling) -> Markers:
    """Return markers on the scaled picture, their sizes and outlines scaled."""
    k = scaling.factor
    return replace(
        markers,
        positions=scaling.points(markers.positions),
        radius=markers.radius * k,
        edge_width=markers.edge_width * k,
        radii=None if markers.radii is None else markers.radii * k,
    )


def scale_image(image: Image, scaling: Scaling) -> Image:
    """Return an image on the scaled picture, resampled to its new size.

    Each new pixel takes the colour of the old pixel under its centre
    (images are smooth gradients, as a colour bar's strip).
    """
    height, width = image.pixels.shape[:2]
    k = scaling.factor
    left = round((image.left - scaling.left) * k)
    top = round((image.top - scaling.top) * k)
    new_w, new_h = max(1, round(width * k)), max(1, round(height * k))
    cols = np.minimum(((np.arange(new_w) + 0.5) * width / new_w).astype(int), width - 1)
    rows = np.minimum(
        ((np.arange(new_h) + 0.5) * height / new_h).astype(int), height - 1
    )
    pixels = image.pixels[rows[:, None], cols[None, :]]
    return replace(image, pixels=np.ascontiguousarray(pixels), left=left, top=top)


def scale_line_batch(batch: LineBatch, scaling: Scaling) -> LineBatch:
    """Return grid lines on the scaled picture, a whole number of pixels wide."""
    vertical = batch.orientation is Orientation.VERTICAL
    across = scaling.x if vertical else scaling.y
    along = scaling.y if vertical else scaling.x
    positions = np.asarray(across(batch.positions), dtype=np.float64)
    span = (float(along(batch.span[0])), float(along(batch.span[1])))
    width = max(1, round(batch.width * scaling.factor))
    return LineBatch(batch.orientation, positions, span, batch.layers, width)


def scale_font(font: FontSpec, factor: float) -> FontSpec:
    """Return a font at a scaled size (a whole number of points, at least 1)."""
    return FontSpec(font.name, max(1, round(font.size * factor)))


def scale_text(
    item: TextItem,
    scaling: Scaling,
    before: TextMeasurer,
    after: TextMeasurer,
) -> TextItem:
    """Return text on the scaled picture, in its font at the scaled size.

    It is placed by the centre of its box (see the module's docs).

    Args:
        item: The text as drawn on screen.
        scaling: The mapping onto the picture.
        before: Measures the text as drawn on screen.
        after: Measures it in the scaled font.

    Returns:
        The scaled text.
    """
    font = scale_font(item.font, scaling.factor)
    old = before.measure_text(item.text, item.font)
    new = after.measure_text(item.text, font)
    # The box's centre, in continuous coordinates, then the new box around it.
    cx = float(scaling.x(item.left - 0.5 + old.width / 2))
    cy = float(scaling.y(item.top - 0.5 + old.height / 2))
    halo = item.halo_width
    if halo > 0:
        halo = max(1, round(halo * scaling.factor))
    return replace(
        item,
        left=round(cx - new.width / 2 + 0.5),
        top=round(cy - new.height / 2 + 0.5),
        font=font,
        halo_width=halo,
    )
