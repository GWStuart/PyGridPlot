"""The legend: a panel in a corner of the plane naming the labelled plots.

Every plotting method takes a ``label=``. A labelled call adds a
:class:`LegendEntry` to the plane's list for the frame (its label, its
resolved style and colour), and :meth:`pygridplot.Grid.legend` draws the
list as a panel in a corner: a row per entry, a small sample of the plot
(its **swatch**) beside its label. In immediate mode the list starts empty
every frame, so the legend always shows what was drawn in it, and an
entry repeated exactly (the same label, style and colour, from drawing a
plot in pieces, say) is shown once.

The swatch is drawn from the call's style, so it looks like the plot:

* a line (:class:`~pygridplot.core.plot.style.LineStyle`) for curves,
  polylines and slope fields;
* a filled box (:class:`~pygridplot.core.plot.style.FillStyle`) for areas
  and regions, at their opacity;
* a filled box with its outline
  (:class:`~pygridplot.core.plot.style.PolygonStyle`) for polygons, and
  likewise (:class:`~pygridplot.core.plot.style.BarStyle`) for bars and
  histograms;
* a marker (:class:`~pygridplot.core.plot.style.PointStyle`) for points,
  with an error bar through it if they had error bars;
* a stem with its marker (:class:`~pygridplot.core.plot.style.SequenceStyle`)
  for sequences;
* an arrow (:class:`~pygridplot.core.plot.style.ArrowStyle`) for arrows and
  vector fields.

Sizes are capped (:data:`MAX_LINE_PX`, :data:`MAX_RADIUS_PX`,
:data:`MAX_HEAD_PX`) so that a very thick line or big marker still fits its
row.

The panel looks like the sliders' panels (see
:mod:`pygridplot.core.handles.panel`), and shares their corners: sliders
keep their places, and a legend in the same corner sits beyond their panel,
so that a legend that grows never moves a slider from under the cursor.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np

from pygridplot.core.coords import ColorLike, ScreenRect
from pygridplot.core.grid.theme import GridTheme
from pygridplot.core.handles.panel import (
    PADDING_PX,
    background_primitives,
    panel_color,
    place_in_corner,
)
from pygridplot.core.handles.slider import Corner
from pygridplot.core.plot.arrows import arrow_rings, head_shape
from pygridplot.core.plot.dashes import dash_pattern, swatch_dashes
from pygridplot.core.plot.style import (
    ArrowStyle,
    BarStyle,
    FillStyle,
    LineStyle,
    PointStyle,
    PolygonStyle,
    SequenceStyle,
)
from pygridplot.core.primitives import (
    Fill,
    LabelText,
    Markers,
    Stroke,
    TextItem,
    TextMeasurer,
    TextRun,
)

SwatchStyle: TypeAlias = (
    LineStyle
    | PointStyle
    | PolygonStyle
    | FillStyle
    | ArrowStyle
    | SequenceStyle
    | BarStyle
)
"""The styles a legend entry can show: one per kind of plot."""

SWATCH_WIDTH_PX: Final = 24
"""Width of a swatch, in pixels."""

SWATCH_HEIGHT_PX: Final = 10
"""Height of a filled swatch (areas, regions and polygons), in pixels."""

SWATCH_GAP_PX: Final = 8
"""Space between a swatch and its label."""

ROW_HEIGHT_PX: Final = 12
"""The least height of a row; a row is otherwise as tall as its label."""

ROW_GAP_PX: Final = 6
"""Space between the rows of a legend."""

MAX_LINE_PX: Final = 6.0
"""The thickest line, stem or shaft drawn in a swatch."""

MAX_OUTLINE_PX: Final = 3.0
"""The thickest polygon outline drawn in a swatch."""

MAX_RADIUS_PX: Final = 6.0
"""The biggest marker drawn in a swatch."""

MAX_HEAD_PX: Final = 12.0
"""The longest arrowhead drawn in a swatch."""


@dataclass(frozen=True, slots=True)
class LegendEntry:
    """One labelled plot: what the legend shows for it.

    Entries compare equal when their label, style and colour are equal, so
    a plot drawn in several labelled calls is shown once.

    Attributes:
        label: The name shown.
        style: The call's resolved style (it decides the swatch).
        color: The call's resolved colour.
        errors: Whether the call drew error bars (points only).
    """

    label: str
    style: SwatchStyle
    color: ColorLike
    errors: bool = False


@dataclass(frozen=True, slots=True, eq=False)
class LegendRow:
    """Where one entry is drawn.

    Attributes:
        entry: The entry.
        text: Its label as label text.
        text_left: Screen x of the left of the label's digit box.
        text_top: Screen y of the top of the label's digit box.
        swatch_left: Continuous screen x of the swatch's left edge.
        center_y: Continuous screen y of the swatch's centre line (level
            with the middle of the label's digits).
    """

    entry: LegendEntry
    text: LabelText
    text_left: int
    text_top: int
    swatch_left: float
    center_y: float


@dataclass(frozen=True, slots=True, eq=False)
class LegendPanel:
    """A legend as laid out on screen.

    Attributes:
        rect: The panel's area (presses there never reach the plane).
        rows: Its entries, top to bottom.
    """

    rect: ScreenRect
    rows: tuple[LegendRow, ...]


def check_label(label: object) -> str | None:
    """Check a plotting call's ``label=``.

    Args:
        label: The label given.

    Returns:
        The label, or ``None`` for none (an empty label is none).

    Raises:
        TypeError: If it is neither a string nor ``None``.
        ValueError: If it has more than one line.
    """
    if label is None or label == "":
        return None
    if not isinstance(label, str):
        msg = f"label must be a string or None, got {type(label).__name__}"
        raise TypeError(msg)
    if "\n" in label:
        msg = f"a legend label must be one line, got {label!r}"
        raise ValueError(msg)
    return label


def layout_legend(
    entries: Sequence[LegendEntry],
    corner: Corner,
    rect: ScreenRect,
    theme: GridTheme,
    measurer: TextMeasurer,
    occupied: ScreenRect | None = None,
) -> LegendPanel | None:
    """Place a legend's rows in a panel in a corner of the plane.

    Args:
        entries: The entries, top to bottom.
        corner: The corner.
        rect: The plane's rectangle.
        theme: Supplies the font.
        measurer: Measures the labels.
        occupied: A panel already in that corner (the sliders'), which the
            legend goes beyond.

    Returns:
        The panel, or ``None`` with no entries or an empty plane.
    """
    if not entries or rect.is_empty:
        return None
    font = theme.label_font_spec
    texts = [LabelText((TextRun(entry.label),)) for entry in entries]
    sizes = [measurer.measure_text(text, font) for text in texts]
    heights = [max(ROW_HEIGHT_PX, size.height + size.descent) for size in sizes]
    width = (
        2 * PADDING_PX
        + SWATCH_WIDTH_PX
        + SWATCH_GAP_PX
        + max(size.width for size in sizes)
    )
    height = 2 * PADDING_PX + sum(heights) + ROW_GAP_PX * (len(entries) - 1)
    left, top = place_in_corner(width, height, corner, rect, occupied)
    rows = []
    y = top + PADDING_PX
    for entry, text, size, row_height in zip(
        entries, texts, sizes, heights, strict=True
    ):
        text_top = y + (row_height - size.height - size.descent) // 2
        rows.append(
            LegendRow(
                entry,
                text,
                left + PADDING_PX + SWATCH_WIDTH_PX + SWATCH_GAP_PX,
                text_top,
                left + PADDING_PX - 0.5,
                text_top + (size.height - 1) / 2,
            )
        )
        y += row_height + ROW_GAP_PX
    return LegendPanel(ScreenRect(left, top, width, height), tuple(rows))


def legend_primitives(
    panel: LegendPanel, theme: GridTheme
) -> tuple[list[Fill | Stroke | Markers], list[TextItem]]:
    """Return what draws a legend: its shapes back to front, then its labels.

    Args:
        panel: The laid out legend.
        theme: Supplies the panel's colours and the labels' font and colour.

    Returns:
        The shapes (the panel, then each row's swatch) and the labels.
    """
    shapes: list[Fill | Stroke | Markers] = list(
        background_primitives(panel.rect, theme)
    )
    background = panel_color(theme)
    for row in panel.rows:
        shapes.extend(
            swatch_primitives(row.entry, row.swatch_left, row.center_y, background)
        )
    return shapes, legend_texts(panel, theme)


def legend_texts(panel: LegendPanel, theme: GridTheme) -> list[TextItem]:
    """Return a legend's labels.

    Args:
        panel: The laid out legend.
        theme: Supplies the labels' font and colour.

    Returns:
        A text item per row.
    """
    font = theme.label_font_spec
    return [
        TextItem(row.text, row.text_left, row.text_top, font, theme.label_color)
        for row in panel.rows
    ]


def swatch_primitives(
    entry: LegendEntry, left: float, center_y: float, background: ColorLike | None
) -> list[Fill | Stroke | Markers]:
    """Return a small sample of a plot, drawn in its style.

    Args:
        entry: The entry.
        left: Continuous screen x of the swatch's left edge.
        center_y: Continuous screen y of its centre line.
        background: The panel's colour (for a sequence marker's edge), or
            ``None``.

    Returns:
        The swatch's shapes, back to front, within :data:`SWATCH_WIDTH_PX`
        of ``left`` (markers and arrowheads may reach a little above and
        below the row).
    """
    style, color = entry.style, entry.color
    right = left + SWATCH_WIDTH_PX
    middle = (left + right) / 2
    if isinstance(style, LineStyle):
        width = min(style.width, MAX_LINE_PX)
        start, end = (left + width / 2, center_y), (right - width / 2, center_y)
        pattern = dash_pattern(style.dash, width)
        if pattern is not None:
            dashes = swatch_dashes(start, end, pattern)
            return [Stroke(dashes, color, width, style.opacity)]
        return [Stroke((np.array([start, end]),), color, width, style.opacity)]
    if isinstance(style, FillStyle):
        return [Fill((_box(left, right, center_y),), color, style.opacity)]
    if isinstance(style, PolygonStyle):
        shapes: list[Fill | Stroke | Markers] = []
        if style.fill:
            fill = style.fill_color if style.fill_color is not None else color
            shapes.append(
                Fill((_box(left, right, center_y),), fill, style.fill_opacity)
            )
        if style.width > 0:
            width = min(style.width, MAX_OUTLINE_PX)
            ring = _box(left + width / 2, right - width / 2, center_y, width / 2)
            outline = np.vstack([ring, ring[:1]])
            pattern = dash_pattern(style.dash, width)
            if pattern is not None:
                # Each side starts and ends with a dash, so the corners are drawn.
                sides = [
                    swatch_dashes(a, b, pattern) for a, b in itertools.pairwise(outline)
                ]
                stroke = Stroke(np.concatenate(sides), color, width, style.opacity)
            else:
                stroke = Stroke((outline,), color, width, style.opacity)
            shapes.append(stroke)
        return shapes
    if isinstance(style, BarStyle):
        shapes = []
        if style.opacity > 0:
            shapes.append(Fill((_box(left, right, center_y),), color, style.opacity))
        if style.edge_width > 0:
            width = min(style.edge_width, MAX_OUTLINE_PX)
            ring = _box(left + width / 2, right - width / 2, center_y, width / 2)
            outline = np.vstack([ring, ring[:1]])
            shapes.append(Stroke((outline,), color, width, 1.0))
        return shapes
    if isinstance(style, PointStyle):
        shapes = []
        if entry.errors:
            width = min(style.error_width, MAX_OUTLINE_PX)
            reach = ROW_HEIGHT_PX / 2 + 1
            cap = min(style.error_cap, SWATCH_WIDTH_PX) / 2
            paths = [np.array([[middle, center_y - reach], [middle, center_y + reach]])]
            if cap > 0:
                for y in (center_y - reach, center_y + reach):
                    paths.append(np.array([[middle - cap, y], [middle + cap, y]]))
            shapes.append(Stroke(tuple(paths), color, width, style.opacity))
        if style.radius > 0:
            edge = style.edge_width if style.edge_color is not None else 0.0
            shapes.append(
                Markers(
                    np.array([[middle, center_y]]),
                    style.marker,
                    min(style.radius, MAX_RADIUS_PX),
                    color,
                    style.edge_color,
                    edge,
                    style.opacity,
                )
            )
        return shapes
    if isinstance(style, SequenceStyle):
        shapes = []
        if style.width > 0:
            width = min(style.width, MAX_LINE_PX)
            stem = np.array(
                [[middle, center_y + SWATCH_HEIGHT_PX / 2 + 1], [middle, center_y]]
            )
            shapes.append(Stroke((stem,), color, width, style.opacity))
        if style.radius > 0:
            radius = min(style.radius, MAX_RADIUS_PX)
            shapes.append(
                Markers(
                    np.array([[middle, center_y]]),
                    style.marker,
                    radius,
                    color,
                    opacity=style.opacity,
                )
            )
        return shapes
    # An arrow (and a vector field's arrows) points right across the swatch.
    width = min(style.width, MAX_LINE_PX)
    length, _ = head_shape(width, style.head_size)
    head = head_shape(width, min(length, MAX_HEAD_PX))
    rings = arrow_rings(
        np.array([[left, center_y]]), np.array([[right, center_y]]), width, head
    )
    return [Fill(tuple(rings), color, style.opacity)]


def _box(left: float, right: float, center_y: float, inset: float = 0.0) -> np.ndarray:
    """Return a filled swatch's rectangle, as a ring.

    Args:
        left: Continuous screen x of its left edge.
        right: Continuous screen x of its right edge.
        center_y: Continuous screen y of its centre line.
        inset: How far inside :data:`SWATCH_HEIGHT_PX` its top and bottom are.

    Returns:
        The four corners, clockwise on screen.
    """
    half = SWATCH_HEIGHT_PX / 2 - inset
    top, bottom = center_y - half, center_y + half
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
