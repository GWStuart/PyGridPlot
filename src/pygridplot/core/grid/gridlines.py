"""Turning classified grid lines into painted line batches.

Every axis describes its lines as :class:`AxisLines`: a position and two
opacities per line (how visible it is, and how much of that is in the major
style). Linear axes derive them from their classified candidate lines (see
:mod:`pygridplot.core.grid.ticks` and :func:`linear_axis_lines`); logarithmic axes
compute them line by line (see :mod:`pygridplot.core.grid.logticks`). This
module then groups lines that share a paint into
:class:`~pygridplot.core.primitives.LineBatch` primitives, and adds the two
axes, with an arrowhead at each one's positive end if the theme asks for it
(:func:`axis_arrows`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.coords import Axis, ScreenRect
from pygridplot.core.grid.theme import GridTheme
from pygridplot.core.grid.ticks import AxisTicks, LineClass
from pygridplot.core.pixels import line_start
from pygridplot.core.primitives import (
    FULL_OPACITY,
    Fill,
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


ARROW_LENGTH_PER_WIDTH: Final = 3.0
"""An axis arrowhead is this many axis widths long, plus :data:`ARROW_LENGTH_BASE`."""

ARROW_LENGTH_BASE: Final = 6.0
"""Pixels added to the length of every axis arrowhead."""

ARROW_ASPECT: Final = 0.8
"""The width of an axis arrowhead across its barbs, as a fraction of its length.

The same proportions as plotted arrows (see :mod:`pygridplot.core.plot.arrows`).
"""

ARROW_INSET_PX: Final = 1.0
"""How far inside the plane's edge an axis arrowhead's tip is."""

ARROW_OVERLAP_PX: Final = 2.0
"""How far an axis line runs on under its arrowhead, so they join without a seam.

The head is wider than the line for this far (by much more), so the line
never shows beside it.
"""


def arrow_head(theme: GridTheme) -> tuple[float, float]:
    """Return the length and width of the axes' arrowheads, in pixels.

    Args:
        theme: Supplies the axis width.

    Returns:
        ``(length, width)``, the width measured across the barbs.
    """
    length = ARROW_LENGTH_PER_WIDTH * theme.axis_width + ARROW_LENGTH_BASE
    return length, ARROW_ASPECT * length


def arrow_overhang(theme: GridTheme) -> float:
    """Return how far the axes' arrowheads reach beyond the sides of the axis lines.

    Axis titles keep this much further from the axes, so they clear the heads.

    Args:
        theme: Supplies the axis settings.

    Returns:
        The overhang in pixels (0 without arrowheads).
    """
    if not (theme.show_axes and theme.show_arrows):
        return 0.0
    _, width = arrow_head(theme)
    return max(0.0, (width - theme.axis_width) / 2.0)


def axis_batches(view: View, theme: GridTheme) -> list[LineBatch]:
    """Return line batches for whichever axes are on screen.

    With arrowheads (:attr:`GridTheme.show_arrows`) each axis stops just
    under its head, at its positive end, instead of running to the edge.

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
    shorten = 0.0
    if theme.show_arrows:
        length, _ = arrow_head(theme)
        shorten = ARROW_INSET_PX + length - ARROW_OVERLAP_PX
    batches = []
    axis_y = _visible_axis(view, "x", width)
    if axis_y is not None:
        span = (float(rect.left), rect.right - shorten)
        positions = np.array([axis_y])
        batches.append(LineBatch(Orientation.HORIZONTAL, positions, span, paint, width))
    axis_x = _visible_axis(view, "y", width)
    if axis_x is not None:
        span = (rect.top + shorten, float(rect.bottom))
        positions = np.array([axis_x])
        batches.append(LineBatch(Orientation.VERTICAL, positions, span, paint, width))
    return batches


def axis_arrows(view: View, theme: GridTheme) -> list[Fill]:
    """Return the arrowheads at the positive ends of the axes on screen.

    Each head is a triangle centred on its axis line's pixels (so on the
    line exactly, whatever its width), with its tip just inside the plane's
    edge. The two heads are separate fills, so neither is rasterised over
    the whole plane.

    Args:
        view: The view being drawn.
        theme: Supplies the axis colour, width and the switch.

    Returns:
        Zero, one or two fills (none unless :attr:`GridTheme.show_arrows`).
    """
    if not (theme.show_axes and theme.show_arrows):
        return []
    rect = view.rect
    width = theme.axis_width
    length, head_width = arrow_head(theme)
    half = head_width / 2.0
    heads = []
    axis_y = _visible_axis(view, "x", width)
    if axis_y is not None:
        center = line_start(axis_y, width) + (width - 1) / 2.0
        tip = rect.right - 0.5 - ARROW_INSET_PX
        base = tip - length
        ring = np.array([[tip, center], [base, center + half], [base, center - half]])
        heads.append(Fill((ring,), theme.axis_color))
    axis_x = _visible_axis(view, "y", width)
    if axis_x is not None:
        center = line_start(axis_x, width) + (width - 1) / 2.0
        tip = rect.top - 0.5 + ARROW_INSET_PX
        base = tip + length
        ring = np.array([[center, tip], [center - half, base], [center + half, base]])
        heads.append(Fill((ring,), theme.axis_color))
    return heads


def _visible_axis(view: View, axis: Axis, width: int) -> float | None:
    """Return the screen position of an axis line, if it is drawn.

    Args:
        view: The view being drawn.
        axis: ``"x"`` for the x axis (the line ``y = 0``, horizontal) or
            ``"y"`` for the y axis (the line ``x = 0``, vertical).
        width: The axis line's thickness in pixels.

    Returns:
        The line's continuous screen coordinate across it, or ``None`` if
        it is off screen, or does not exist because the other axis is
        logarithmic (which has no zero).
    """
    rect = view.rect
    if axis == "x":
        position = view.to_screen_y(0.0)
        drawn = not view.log_y and _line_visible(position, width, rect.top, rect.bottom)
    else:
        position = view.to_screen_x(0.0)
        drawn = not view.log_x and _line_visible(position, width, rect.left, rect.right)
    return position if drawn else None


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
