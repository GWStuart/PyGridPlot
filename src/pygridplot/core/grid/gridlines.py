"""Turning classified grid lines into painted line batches.

Every axis describes its lines as :class:`AxisLines`: a position and two
opacities per line (how visible it is, and how much of that is in the major
style). Linear axes derive them from their classified candidate lines (see
:mod:`pygridplot.core.grid.ticks` and :func:`linear_axis_lines`); logarithmic axes
compute them line by line (see :mod:`pygridplot.core.grid.logticks`). This
module then groups lines that share a paint into
:class:`~pygridplot.core.primitives.LineBatch` primitives, and adds the two
axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.coords import Axis, ScreenRect
from pygridplot.core.grid.theme import GridTheme
from pygridplot.core.grid.ticks import AxisTicks, LineClass
from pygridplot.core.pixels import line_start
from pygridplot.core.primitives import (
    FULL_OPACITY,
    LineBatch,
    Orientation,
    PaintLayer,
)
from pygridplot.core.view import View


@dataclass(frozen=True, slots=True, eq=False)
class AxisLines:
    """The grid lines of one axis, each with its own opacity.

    Both linear and logarithmic axes describe their lines this way, so they
    share one painting step (:func:`line_batches`).

    Attributes:
        axis: ``"x"`` (vertical lines) or ``"y"`` (horizontal lines).
        positions: Continuous screen coordinate of each line.
        total: Overall opacity of each line in ``[0, 1]``.
        major: How much of that opacity is painted in the major style, in
            ``[0, total]``. The rest is painted in the minor style, so
            ``major == total`` is a major line and ``major == 0`` a minor one.
    """

    axis: Axis
    positions: NDArray[np.float64]
    total: NDArray[np.float64]
    major: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.positions.shape[0])


def level_visibility(
    classes: tuple[NDArray[np.int8], ...], weights: tuple[float, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Combine a line's class under each cross-faded level into opacities.

    Args:
        classes: The :class:`~pygridplot.core.grid.ticks.LineClass` of every line
            under each level.
        weights: The opacity of each level.

    Returns:
        ``(total, major)`` per line, as in :class:`AxisLines`: the summed
        weight of the levels that draw the line at all, and of those that draw
        it as a major line.
    """
    count = classes[0].shape[0] if classes else 0
    total = np.zeros(count)
    major = np.zeros(count)
    for row, weight in zip(classes, weights, strict=True):
        total += np.where(row != LineClass.NONE, weight, 0.0)
        major += np.where(row == LineClass.MAJOR, weight, 0.0)
    total[total > FULL_OPACITY] = 1.0
    major[major > FULL_OPACITY] = 1.0
    return total, major


def linear_axis_lines(ticks: AxisTicks, *, skip_axis: bool) -> AxisLines:
    """Describe the classified candidate lines of an axis as :class:`AxisLines`.

    Args:
        ticks: The axis' classified candidate lines.
        skip_axis: Leave out the line at zero (the axis, drawn separately on
            top).

    Returns:
        The lines with their opacities. Lines no level draws are dropped.
    """
    weights = tuple(level.weight for level in ticks.levels)
    total, major = level_visibility(ticks.classes, weights)
    keep = total > 0
    if skip_axis:
        keep &= ticks.indices != 0
    return AxisLines(ticks.axis, ticks.positions[keep], total[keep], major[keep])


def line_batches(
    lines: AxisLines, rect: ScreenRect, theme: GridTheme
) -> list[tuple[int, LineBatch]]:
    """Group an axis' lines into batches that share a paint.

    Args:
        lines: The lines of one axis.
        rect: The plane's screen rectangle.
        theme: Supplies colours, widths and visibility switches.

    Returns:
        ``(rank, batch)`` pairs, where ``rank`` orders batches in painter's
        order (minor-style before major-style).
    """
    if len(lines) == 0 or not (theme.show_minor or theme.show_major):
        return []
    keep = _lines_inside(lines, rect, max(theme.minor_width, theme.major_width))
    total, major = lines.total[keep], lines.major[keep]
    if not theme.show_major:
        major = np.zeros_like(major)
    positions = lines.positions[keep]
    orientation, span = _orientation_and_span(lines.axis, rect)
    pairs = np.column_stack((total, major))
    batches = []
    for pair in np.unique(pairs, axis=0):
        line_total, line_major = float(pair[0]), float(pair[1])
        if line_total <= 0.0:
            continue
        layers, width, rank = _paint_for(line_total, line_major, theme)
        chosen = positions[(total == line_total) & (major == line_major)]
        batches.append((rank, LineBatch(orientation, chosen, span, layers, width)))
    return batches


def _lines_inside(lines: AxisLines, rect: ScreenRect, width: int) -> np.ndarray:
    """Return a mask of the lines that touch the plane's rectangle.

    Args:
        lines: The lines of one axis.
        rect: The plane's screen rectangle.
        width: The thickest line width in use.

    Returns:
        A boolean mask over the lines.
    """
    low, high = (
        (rect.left, rect.right) if lines.axis == "x" else (rect.top, rect.bottom)
    )
    return (lines.positions > low - width) & (lines.positions < high + width)


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


def _paint_for(
    total: float, major: float, theme: GridTheme
) -> tuple[tuple[PaintLayer, ...], int, int]:
    """Work out the paint, width and draw rank of lines with given opacities.

    A line that is partly minor and partly major (while the grid cross-fades
    between spacings) is painted with the minor colour and the major colour
    composited on top. The layer opacities are chosen so the result is
    exactly ``total`` opaque, of which ``major`` is in the major colour: a
    linear blend between the two looks.

    Args:
        total: The line's overall opacity.
        major: The part of it in the major style.
        theme: Supplies colours and widths.

    Returns:
        ``(layers, width, rank)``.
    """
    if major <= 0.0:
        return (PaintLayer(theme.minor_color, total),), theme.minor_width, 1
    if major >= total:
        return (PaintLayer(theme.major_color, major),), theme.major_width, 2
    under = 0.0 if major >= 1.0 else (total - major) / (1.0 - major)
    layers = (
        PaintLayer(theme.minor_color, under),
        PaintLayer(theme.major_color, major),
    )
    width = theme.major_width if major >= 0.5 * total else theme.minor_width
    return layers, width, 2


def axis_batches(view: View, theme: GridTheme) -> list[LineBatch]:
    """Return line batches for whichever axes are on screen.

    Args:
        view: The view being drawn.
        theme: Supplies the axis colour, width and visibility.

    Returns:
        Zero, one or two single-line batches. A log axis has no zero, so the
        axis line perpendicular to it is never drawn.
    """
    if not theme.show_axes:
        return []
    rect = view.rect
    width = theme.axis_width
    paint = (PaintLayer(theme.axis_color),)
    batches = []
    axis_y = view.to_screen_y(0.0)
    if not view.log_y and _line_visible(axis_y, width, rect.top, rect.bottom):
        span = (float(rect.left), float(rect.right))
        positions = np.array([axis_y])
        batches.append(LineBatch(Orientation.HORIZONTAL, positions, span, paint, width))
    axis_x = view.to_screen_x(0.0)
    if not view.log_x and _line_visible(axis_x, width, rect.left, rect.right):
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
