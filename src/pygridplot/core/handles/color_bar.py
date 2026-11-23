"""The colour bar: which colour is which value, in a corner of the plane.

Points coloured by a value (see :func:`pygridplot.color_map`) need a key:
:meth:`pygridplot.Grid.color_bar` draws a panel in a corner holding the
colour map as an upright strip, low at the bottom, with ticks and labels at
round values and an optional title above. It is a panel like the legend's
and the sliders' (see :mod:`pygridplot.core.handles.panel`), with their look,
and it stacks with them in a corner: beyond the sliders and the legend.

The ticks are exact, like the grid's: on a linear scale, the multiples of
the 1-2-5 step that gives about :data:`TICKS` of them over the range,
written as exact decimals in one notation; on a log scale, the powers of
ten in the range (and their 2 and 5 multiples, when the range spans less
than a decade or two). The strip itself is an
:class:`~pygridplot.core.primitives.Image`, one row of pixels per colour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

import numpy as np

from pygridplot.core.coords import ScreenRect
from pygridplot.core.grid.formatting import (
    choose_notation,
    format_log_value,
    format_value,
)
from pygridplot.core.grid.nice import NiceStep, select_level
from pygridplot.core.grid.theme import GridTheme
from pygridplot.core.handles.panel import (
    PADDING_PX,
    background_primitives,
    place_in_corner,
)
from pygridplot.core.handles.slider import Corner
from pygridplot.core.plot.color_maps import ColorStops, map_values
from pygridplot.core.primitives import (
    Fill,
    Image,
    LabelText,
    Markers,
    Stroke,
    TextItem,
    TextMeasurer,
    TextRun,
    TextSize,
)

BAR_WIDTH_PX: Final = 14
"""Width of the colour strip."""

BAR_LENGTH_PX: Final = 160
"""Length of the colour strip, from the low end to the high end."""

TICK_PX: Final = 4
"""Length of a tick, beside the strip."""

LABEL_GAP_PX: Final = 4
"""Space between a tick and its label."""

TITLE_GAP_PX: Final = 8
"""Space between the title and the strip."""

TICKS: Final = 5
"""About how many ticks a linear colour bar has."""


@dataclass(frozen=True, slots=True, eq=False)
class ColorBarPanel:
    """A colour bar as laid out on screen.

    Attributes:
        rect: The panel's area (presses there never reach the plane).
        bar: The pixels of the colour strip.
        ticks: Each tick's screen y (continuous) and its label's text, size
            and top-left pixel.
        title: The title and its top-left pixel, or ``None``.
    """

    rect: ScreenRect
    bar: ScreenRect
    ticks: tuple[tuple[float, LabelText, int, int], ...]
    title: tuple[LabelText, int, int] | None


def color_bar_ticks(
    low: float, high: float, *, log: bool, theme: GridTheme
) -> list[tuple[float, LabelText]]:
    """Return a colour bar's ticks: round values in its range, with labels.

    Args:
        low: The range's low end.
        high: Its high end (greater than ``low``).
        log: Whether the bar is on a log scale (then both are positive).
        theme: Supplies the number format.

    Returns:
        ``(value, label)`` pairs, low to high. Always at least the range's
        ends when nothing round lies inside it.
    """
    fmt = theme.number_format
    if log:
        first, last = (
            math.ceil(math.log10(low) - 1e-9),
            math.floor(math.log10(high) + 1e-9),
        )
        mantissas = (1,) if last - first >= 2 else (1, 2, 5)
        values = [
            Decimal(m).scaleb(power)
            for power in range(first - 1, last + 1)
            for m in mantissas
            if low * (1 - 1e-9) <= m * 10.0**power <= high * (1 + 1e-9)
        ]
        if len(values) >= 2:
            return [(float(v), format_log_value(v, fmt)) for v in values]
        return [(low, _plain(low, theme)), (high, _plain(high, theme))]
    step = NiceStep(select_level((high - low) / (TICKS - 1)))
    size = step.decimal
    first = math.ceil(low / step.value - 1e-9)
    last = math.floor(high / step.value + 1e-9)
    values = [size * k for k in range(first, last + 1)]
    if len(values) < 2:
        return [(low, _plain(low, theme)), (high, _plain(high, theme))]
    largest = max(abs(values[0]), abs(values[-1]))
    notation = choose_notation(largest, step, fmt)
    return [(float(v), format_value(v, fmt, notation)) for v in values]


def _plain(value: float, theme: GridTheme) -> LabelText:
    """A value written to four significant digits (for ranges with no round values)."""
    return format_value(Decimal(f"{value:.4g}"), theme.number_format)


def layout_color_bar(
    low: float,
    high: float,
    *,
    log: bool,
    title: str | None,
    corner: Corner,
    rect: ScreenRect,
    theme: GridTheme,
    measurer: TextMeasurer,
    occupied: ScreenRect | None = None,
) -> ColorBarPanel | None:
    """Place a colour bar's strip, ticks, labels and title in a corner panel.

    Args:
        low: The value at the strip's bottom.
        high: The value at its top.
        log: Whether the strip is on a log scale.
        title: A title above the strip, or ``None``.
        corner: The corner.
        rect: The plane's rectangle.
        theme: Supplies the font and the number format.
        measurer: Measures the labels.
        occupied: The panels already in the corner, which the colour bar
            goes beyond.

    Returns:
        The panel, or ``None`` on an empty plane.
    """
    if rect.is_empty:
        return None
    font = theme.label_font_spec
    ticks = color_bar_ticks(low, high, log=log, theme=theme)
    sizes = [measurer.measure_text(text, font) for _, text in ticks]
    label_width = max(size.width for size in sizes)
    heading: tuple[LabelText, TextSize] | None = None
    if title:
        text = LabelText((TextRun(title),))
        heading = (text, measurer.measure_text(text, font))
    digit = sizes[0].height
    # Labels at the ends reach half a digit beyond the strip.
    above = (digit + 1) // 2
    title_height = 0 if heading is None else heading[1].height + heading[1].descent
    title_height += 0 if heading is None else TITLE_GAP_PX
    width = 2 * PADDING_PX + BAR_WIDTH_PX + TICK_PX + LABEL_GAP_PX + label_width
    if heading is not None:
        width = max(width, 2 * PADDING_PX + heading[1].width)
    height = 2 * PADDING_PX + title_height + above + BAR_LENGTH_PX + above
    left, top = place_in_corner(width, height, corner, rect, occupied)
    bar_top = top + PADDING_PX + title_height + above
    bar = ScreenRect(left + PADDING_PX, bar_top, BAR_WIDTH_PX, BAR_LENGTH_PX)
    label_left = bar.right + TICK_PX + LABEL_GAP_PX
    placed = []
    for (value, text), size in zip(ticks, sizes, strict=True):
        y = bar.top - 0.5 + (1.0 - _fraction(value, low, high, log)) * BAR_LENGTH_PX
        label_top = round(y - (size.height - 1) / 2)
        placed.append((y, text, label_left, label_top))
    named = (
        None if heading is None else (heading[0], left + PADDING_PX, top + PADDING_PX)
    )
    return ColorBarPanel(
        ScreenRect(left, top, width, height), bar, tuple(placed), named
    )


def _fraction(value: float, low: float, high: float, log: bool) -> float:
    """Where a value lies along the strip: 0 at the bottom, 1 at the top."""
    if log:
        return (math.log10(value) - math.log10(low)) / (
            math.log10(high) - math.log10(low)
        )
    return (value - low) / (high - low)


def color_bar_primitives(
    panel: ColorBarPanel, stops: ColorStops, theme: GridTheme
) -> tuple[list[Fill | Stroke | Markers | Image], list[TextItem]]:
    """Return what draws a colour bar: its shapes back to front, then its texts.

    Args:
        panel: The laid out colour bar.
        stops: The colour map's colours.
        theme: Supplies the panel's colours and the labels' font and colour.

    Returns:
        The shapes (the panel, the strip, its border and the ticks) and the
        texts (the labels and the title).
    """
    shapes: list[Fill | Stroke | Markers | Image] = list(
        background_primitives(panel.rect, theme)
    )
    bar = panel.bar
    # One row per pixel, the top row at the high end; each row's colour is
    # the map's at the middle of the row.
    fractions = 1.0 - (np.arange(bar.height) + 0.5) / bar.height
    rows = map_values(fractions, stops, (0.0, 1.0))
    pixels = np.repeat(rows[:, None, :], bar.width, axis=1)
    shapes.append(Image(np.ascontiguousarray(pixels), bar.left, bar.top))
    left, top = bar.left - 0.5, bar.top - 0.5
    right, bottom = bar.right - 0.5, bar.bottom - 0.5
    border = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom], [left, top]]
    )
    shapes.append(Stroke((border,), theme.axis_color, 1.0))
    marks = tuple(np.array([[right, y], [right + TICK_PX, y]]) for y, *_ in panel.ticks)
    shapes.append(Stroke(marks, theme.axis_color, 1.0))
    font = theme.label_font_spec
    texts = [
        TextItem(text, x, y, font, theme.label_color) for _, text, x, y in panel.ticks
    ]
    if panel.title is not None:
        text, x, y = panel.title
        texts.append(TextItem(text, x, y, font, theme.label_color))
    return shapes, texts
