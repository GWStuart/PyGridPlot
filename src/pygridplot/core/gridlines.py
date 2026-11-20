"""Turning classified grid lines into painted line batches.

Given the candidate lines of an axis (see :mod:`pygridplot.core.ticks`),
this module decides how each line is painted: minor, major, or a blend of
the two while the grid cross-fades between spacings. It groups lines that
share a paint into :class:`~pygridplot.core.primitives.LineBatch`
primitives, and adds the two axes.
"""

from __future__ import annotations

import numpy as np

from pygridplot.core.coords import Axis, ScreenRect
from pygridplot.core.pixels import line_start
from pygridplot.core.primitives import (
    FULL_OPACITY,
    LineBatch,
    Orientation,
    PaintLayer,
)
from pygridplot.core.theme import GridTheme
from pygridplot.core.ticks import AxisTicks, LineClass
from pygridplot.core.view import View


def grid_line_batches(
    ticks: AxisTicks, rect: ScreenRect, theme: GridTheme
) -> list[tuple[int, LineBatch]]:
    """Group an axis' candidate lines into batches that share a paint.

    Args:
        ticks: The classified candidate lines of one axis.
        rect: The plane's screen rectangle.
        theme: Supplies colours, widths and visibility switches.

    Returns:
        ``(rank, batch)`` pairs, where ``rank`` orders batches in painter's
        order (minor-style before major-style).
    """
    if len(ticks) == 0 or not (theme.show_minor or theme.show_major):
        return []
    classes = np.stack(ticks.classes)  # shape: (levels, lines)
    if not theme.show_major:
        classes[classes == LineClass.MAJOR] = LineClass.MINOR
    keep = _lines_inside(ticks, rect, max(theme.minor_width, theme.major_width))
    if theme.show_axes:
        keep &= ticks.indices != 0  # the axis is drawn separately, on top
    weights = tuple(level.weight for level in ticks.levels)
    orientation, span = _orientation_and_span(ticks.axis, rect)

    # Encode each line's classification under every level as one integer key.
    keys = np.zeros(len(ticks), dtype=np.int64)
    for row in classes:
        keys = keys * 3 + row
    batches = []
    for key in np.unique(keys[keep]):
        combo = _decode_key(int(key), len(weights))
        if all(c == LineClass.NONE for c in combo):
            continue
        layers, width, rank = _paint_for(combo, weights, theme)
        positions = ticks.positions[keep & (keys == key)]
        batches.append((rank, LineBatch(orientation, positions, span, layers, width)))
    return batches


def _lines_inside(ticks: AxisTicks, rect: ScreenRect, width: int) -> np.ndarray:
    """Return a mask of candidate lines that touch the plane's rectangle.

    Args:
        ticks: The candidate lines.
        rect: The plane's screen rectangle.
        width: The thickest line width in use.

    Returns:
        A boolean mask over the candidate lines.
    """
    low, high = (
        (rect.left, rect.right) if ticks.axis == "x" else (rect.top, rect.bottom)
    )
    return (ticks.positions > low - width) & (ticks.positions < high + width)


def _orientation_and_span(
    axis: Axis, rect: ScreenRect
) -> tuple[Orientation, tuple[float, float]]:
    """Return the line direction and extent for an axis' grid lines.

    Args:
        axis: ``"x"`` (vertical lines) or ``"y"`` (horizontal lines).
        rect: The plane's screen rectangle.

    Returns:
        ``(orientation, span)``. The span covers the whole rectangle.
    """
    if axis == "x":
        return Orientation.VERTICAL, (float(rect.top), float(rect.bottom))
    return Orientation.HORIZONTAL, (float(rect.left), float(rect.right))


def _decode_key(key: int, levels: int) -> tuple[LineClass, ...]:
    """Invert the base-3 encoding used to group lines by classification.

    Args:
        key: The encoded classification.
        levels: How many levels were encoded.

    Returns:
        The :class:`LineClass` under each level, finest level first.
    """
    digits = []
    for _ in range(levels):
        key, digit = divmod(key, 3)
        digits.append(LineClass(digit))
    return tuple(reversed(digits))


def _paint_for(
    combo: tuple[LineClass, ...], weights: tuple[float, ...], theme: GridTheme
) -> tuple[tuple[PaintLayer, ...], int, int]:
    """Work out the paint, width and draw rank of lines with one classification.

    While two levels are cross-faded, a line can be (say) major in one level
    and minor in the other. Such a line is painted with the minor colour and
    the major colour composited on top at the major level's weight. That is
    exactly a linear blend between the two looks.

    Args:
        combo: The line's class under each active level.
        weights: The opacity of each active level.
        theme: Supplies colours and widths.

    Returns:
        ``(layers, width, rank)``.
    """
    present = [(cls, w) for cls, w in zip(combo, weights, strict=True) if cls]
    distinct = sorted({cls for cls, _ in present})
    colors = {LineClass.MINOR: theme.minor_color, LineClass.MAJOR: theme.major_color}
    widths = {LineClass.MINOR: theme.minor_width, LineClass.MAJOR: theme.major_width}
    if len(distinct) == 1:
        cls = distinct[0]
        opacity = sum(w for _, w in present)
        opacity = 1.0 if opacity > FULL_OPACITY else opacity
        return (PaintLayer(colors[cls], opacity),), widths[cls], int(cls)
    major_weight = sum(w for cls, w in present if cls == LineClass.MAJOR)
    layers = (
        PaintLayer(theme.minor_color, 1.0),
        PaintLayer(theme.major_color, major_weight),
    )
    dominant = LineClass.MAJOR if major_weight >= 0.5 else LineClass.MINOR
    return layers, widths[dominant], int(LineClass.MAJOR)


def axis_batches(view: View, theme: GridTheme) -> list[LineBatch]:
    """Return line batches for whichever axes are on screen.

    Args:
        view: The view being drawn.
        theme: Supplies the axis colour, width and visibility.

    Returns:
        Zero, one or two single-line batches.
    """
    if not theme.show_axes:
        return []
    rect = view.rect
    width = theme.axis_width
    paint = (PaintLayer(theme.axis_color),)
    batches = []
    axis_y = view.to_screen_y(0.0)
    if _line_visible(axis_y, width, rect.top, rect.bottom):
        span = (float(rect.left), float(rect.right))
        positions = np.array([axis_y])
        batches.append(LineBatch(Orientation.HORIZONTAL, positions, span, paint, width))
    axis_x = view.to_screen_x(0.0)
    if _line_visible(axis_x, width, rect.left, rect.right):
        span = (float(rect.top), float(rect.bottom))
        positions = np.array([axis_x])
        batches.append(LineBatch(Orientation.VERTICAL, positions, span, paint, width))
    return batches


def _line_visible(position: float, width: int, low: int, high: int) -> bool:
    """Return whether a ``width``-pixel line at ``position`` touches ``[low, high)``.

    Args:
        position: Continuous screen coordinate of the line's centre.
        width: Line thickness in pixels.
        low: First pixel of the visible range.
        high: One past the last pixel of the visible range.

    Returns:
        ``True`` if any of the line's pixels are visible.
    """
    if not low - width - 1 < position < high + width + 1:
        return False  # also guards line_start against huge off-screen values
    first = line_start(position, width)
    return first < high and first + width > low
