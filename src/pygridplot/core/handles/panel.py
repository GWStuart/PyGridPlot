"""The panels of sliders and checkboxes in the corners of a plane.

A plane's sliders and checkboxes (its *controls*) are drawn in a panel in a
corner of the plane, one panel per corner, stacked in the order they are
drawn (creation order by default). Each control takes a row. A slider's
row has its text (``label = value``) above and its track below, with the
handle on the track, and a play button left of the track if it can play;
in a panel with play buttons, the tracks of the sliders without one start
further left, so that all the tracks end together. A checkbox's row is its
box with its label beside it. :func:`layout_panels` places the rows in
screen pixels, and :func:`panel_primitives` returns what to draw, as the
same backend-neutral primitives as every other plot (fills for the panel,
the boxes and the play icons, strokes for the tracks and ticks, markers for
the handles and buttons, and text items), so the renderer needs nothing
new.

The layout is kept by the plane until the next frame: it is what the user
sees, so presses and the wheel are matched against it (:class:`SliderRow`
and :class:`CheckboxRow` give the bands that a press grabs, and a slider's
row maps the cursor to a point of the track). A control not drawn in the
last frame cannot be used.

The panel has the theme's background colour, nearly opaque, and a thin
border in its major grid line colour, so it reads as a control on top of
the plot in light and dark themes alike; the text uses the theme's label
font and colour.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeAlias

import numpy as np

from pygridplot.core.coords import ColorLike, ScreenRect
from pygridplot.core.grid.theme import GridTheme
from pygridplot.core.handles.checkbox import Checkbox
from pygridplot.core.handles.slider import CORNERS, Slider
from pygridplot.core.primitives import (
    Fill,
    LabelText,
    Markers,
    Stroke,
    TextItem,
    TextMeasurer,
    TextRun,
)

Control: TypeAlias = Slider[Any] | Checkbox
"""A control drawn in a corner panel."""

MARGIN_PX: Final = 10
"""Distance from a panel to the plane's edges, in pixels."""

PADDING_PX: Final = 10
"""Space inside a panel around its rows, in pixels."""

TRACK_PX: Final = 200.0
"""Length of a track, between the handle's positions at its two ends."""

MIN_TRACK_PX: Final = 40.0
"""Shortest track drawn; a plane too narrow for it shows no sliders."""

HANDLE_RADIUS_PX: Final = 7.0
"""Radius of a slider's handle."""

HANDLE_EDGE_PX: Final = 2.0
"""Width of the outline (in the background colour) around a handle."""

TRACK_WIDTH_PX: Final = 4.0
"""Thickness of a track."""

TEXT_GAP_PX: Final = 6
"""Space between a slider's text and its track."""

ROW_GAP_PX: Final = 10
"""Space between the rows of a panel."""

GRAB_PX: Final = 4.0
"""How far beyond the handle (above, below and past the ends) a press grabs."""

RING_PX: Final = 5.0
"""How far the hover ring reaches beyond the handle."""

RING_OPACITY: Final = 0.3
"""Opacity of the hover ring."""

PANEL_OPACITY: Final = 0.88
"""Opacity of the panel's background."""

DISABLED_OPACITY: Final = 0.4
"""Opacity of a disabled slider."""

CORNER_RADIUS_PX: Final = 6.0
"""Radius of a panel's rounded corners."""

PLAY_RADIUS_PX: Final = 9.0
"""Radius of a slider's play button."""

PLAY_GAP_PX: Final = 6.0
"""Space between a play button and the handle at the low end of the track."""

BOX_PX: Final = 14.0
"""Size of a checkbox's box."""

BOX_GAP_PX: Final = 8
"""Space between a checkbox's box and its label."""

BOX_RADIUS_PX: Final = 3.0
"""Radius of a checkbox's rounded corners."""

BOX_EDGE_PX: Final = 1.5
"""Width of an unticked box's outline."""

BOX_EDGE_OPACITY: Final = 0.6
"""Opacity of an unticked box's outline (in the label colour)."""

TICK_WIDTH_PX: Final = 2.0
"""Width of the tick in a ticked box."""


@dataclass(frozen=True, slots=True, eq=False)
class SliderRow:
    """Where one slider is drawn, and how to tell what a press on it means.

    Attributes:
        slider: The slider.
        text: Its text, ``label = value``.
        text_left: Screen x of the left of the text's digit box.
        text_top: Screen y of the top of the text's digit box.
        start: Screen x of the handle's centre at the low end of the track.
        end: Screen x of the handle's centre at the high end.
        y: Screen y of the track's centre line.
        band: ``(left, top, right, bottom)`` in continuous pixels: a press
            inside grabs the slider.
        button: The centre of the play button, or ``None`` for none.
    """

    slider: Slider[Any]
    text: LabelText
    text_left: int
    text_top: int
    start: float
    end: float
    y: float
    band: tuple[float, float, float, float]
    button: tuple[float, float] | None = None

    @property
    def control(self) -> Slider[Any]:
        """The slider (every row has a ``control``)."""
        return self.slider

    def on_button(self, px: float, py: float) -> bool:
        """Return whether a screen point is on the play button (with its grab margin).

        Args:
            px: Screen x.
            py: Screen y.

        Returns:
            Whether a press there starts or stops playing.
        """
        if self.button is None:
            return False
        bx, by = self.button
        return math.hypot(px - bx, py - by) <= PLAY_RADIUS_PX + GRAB_PX

    @property
    def handle_x(self) -> float:
        """Screen x of the handle's centre."""
        return self.start + self.slider.fraction * (self.end - self.start)

    def fraction_at(self, px: float) -> float:
        """Return how far along the track a screen x is (clamped to ``[0, 1]``).

        Args:
            px: Screen x.

        Returns:
            The fraction of the track.
        """
        return min(max((px - self.start) / (self.end - self.start), 0.0), 1.0)

    def contains(self, px: float, py: float) -> bool:
        """Return whether a screen point is in the band that grabs the slider.

        Args:
            px: Screen x.
            py: Screen y.

        Returns:
            Whether a press there grabs it.
        """
        left, top, right, bottom = self.band
        return left <= px <= right and top <= py <= bottom

    def on_handle(self, px: float, py: float) -> bool:
        """Return whether a screen point is on the handle (with its grab margin).

        Args:
            px: Screen x.
            py: Screen y.

        Returns:
            Whether a press there grabs the handle where it is, rather than
            moving it to the press.
        """
        reach = HANDLE_RADIUS_PX + HANDLE_EDGE_PX + GRAB_PX
        return math.hypot(px - self.handle_x, py - self.y) <= reach


@dataclass(frozen=True, slots=True, eq=False)
class CheckboxRow:
    """Where one checkbox is drawn, and the band a press toggles it in.

    Attributes:
        checkbox: The checkbox.
        text: Its label.
        text_left: Screen x of the left of the label's digit box.
        text_top: Screen y of the top of the label's digit box.
        box: ``(left, top, right, bottom)`` of the box, in continuous pixels.
        band: ``(left, top, right, bottom)`` in continuous pixels: a press
            inside (on the box or the label) toggles the checkbox.
    """

    checkbox: Checkbox
    text: LabelText
    text_left: int
    text_top: int
    box: tuple[float, float, float, float]
    band: tuple[float, float, float, float]

    @property
    def control(self) -> Checkbox:
        """The checkbox (every row has a ``control``)."""
        return self.checkbox

    def contains(self, px: float, py: float) -> bool:
        """Return whether a screen point is in the band that toggles the checkbox.

        Args:
            px: Screen x.
            py: Screen y.

        Returns:
            Whether a press there toggles it.
        """
        left, top, right, bottom = self.band
        return left <= px <= right and top <= py <= bottom


Row: TypeAlias = SliderRow | CheckboxRow
"""A row of a panel: a slider's or a checkbox's."""


@dataclass(frozen=True, slots=True, eq=False)
class Panel:
    """A corner's panel of sliders and checkboxes.

    Attributes:
        rect: The panel's area on screen (presses there never reach the
            plane).
        rows: Its controls, top to bottom.
    """

    rect: ScreenRect
    rows: tuple[Row, ...]


def layout_panels(
    controls: Sequence[Control],
    rect: ScreenRect,
    theme: GridTheme,
    measurer: TextMeasurer,
) -> tuple[Panel, ...]:
    """Place sliders and checkboxes in panels, one per corner of the plane.

    Each slider's :attr:`~Slider.track_px` is set to the length of its track
    (which its text's precision depends on).

    Args:
        controls: The controls to draw, in order (top to bottom in each panel).
        rect: The plane's rectangle.
        theme: Supplies the font and number format.
        measurer: Measures the texts.

    Returns:
        The panels (none if the plane is too narrow for a track).
    """
    available = rect.width - 2 * (MARGIN_PX + PADDING_PX + HANDLE_RADIUS_PX)
    track = min(TRACK_PX, available)
    if track < MIN_TRACK_PX or rect.height <= 2 * MARGIN_PX:
        return ()
    font = theme.label_font_spec
    panels = []
    for corner in CORNERS:
        chosen = [control for control in controls if control.corner == corner]
        if not chosen:
            continue
        # With play buttons, the tracks without one are as much longer, so
        # every track ends at the same place (and the buttons' tracks shorter
        # where the plane is too narrow for both).
        playing = any(_plays(control) for control in chosen)
        button = 2 * PLAY_RADIUS_PX + PLAY_GAP_PX if playing else 0.0
        full = min(track + button, available)
        if full - button < MIN_TRACK_PX:
            continue
        texts, sizes = [], []
        for control in chosen:
            if isinstance(control, Slider):
                control.track_px = full - button if _plays(control) else full
                text = control.text(theme.number_format)
            else:
                text = LabelText((TextRun(control.label),))
            texts.append(text)
            sizes.append(measurer.measure_text(text, font))
        widths = [
            full + 2 * HANDLE_RADIUS_PX
            if isinstance(control, Slider)
            else BOX_PX + BOX_GAP_PX + size.width
            for control, size in zip(chosen, sizes, strict=True)
        ]
        inner = max(*widths, *(size.width for size in sizes))
        width = math.ceil(inner) + 2 * PADDING_PX
        heights = [
            _row_height(control, size.overflow_top + size.height)
            for control, size in zip(chosen, sizes, strict=True)
        ]
        height = (
            math.ceil(sum(heights) + ROW_GAP_PX * (len(chosen) - 1)) + 2 * PADDING_PX
        )
        left = (
            rect.left + MARGIN_PX
            if corner.endswith("left")
            else rect.right - MARGIN_PX - width
        )
        top = (
            rect.top + MARGIN_PX
            if corner.startswith("top")
            else rect.bottom - MARGIN_PX - height
        )
        rows: list[Row] = []
        y = float(top + PADDING_PX)
        for control, text, size, row_height in zip(
            chosen, texts, sizes, heights, strict=True
        ):
            if isinstance(control, Slider):
                rows.append(
                    _slider_row(control, text, size.overflow_top, size.height, left, y)
                )
            else:
                rows.append(_checkbox_row(control, text, size, left, y, row_height))
            y += row_height + ROW_GAP_PX
        panels.append(Panel(ScreenRect(left, top, width, height), tuple(rows)))
    return tuple(panels)


def _plays(control: Control) -> bool:
    """Whether a control is a slider with a play button."""
    return isinstance(control, Slider) and control.play is not None


def _row_height(control: Control, text_height: float) -> float:
    """The height of a control's row, given its text's (superscripts included)."""
    if isinstance(control, Slider):
        reach = PLAY_RADIUS_PX if _plays(control) else HANDLE_RADIUS_PX
        return text_height + TEXT_GAP_PX + 2 * reach
    return max(BOX_PX, text_height)


def _slider_row(
    slider: Slider[Any],
    text: LabelText,
    overflow: int,
    text_height: int,
    left: int,
    y: float,
) -> SliderRow:
    """Place a slider's row, its top at ``y`` in a panel whose left is ``left``."""
    text_top = round(y + overflow)
    plays = _plays(slider)
    reach = PLAY_RADIUS_PX if plays else HANDLE_RADIUS_PX
    track_y = float(text_top + text_height + TEXT_GAP_PX + reach)
    inset = 2 * PLAY_RADIUS_PX + PLAY_GAP_PX if plays else 0.0
    start = float(left + PADDING_PX + inset + HANDLE_RADIUS_PX)
    end = start + slider.track_px
    grab = HANDLE_RADIUS_PX + GRAB_PX
    button = (left + PADDING_PX + PLAY_RADIUS_PX, track_y) if plays else None
    return SliderRow(
        slider,
        text,
        left + PADDING_PX,
        text_top,
        start,
        end,
        track_y,
        (start - grab, track_y - grab, end + grab, track_y + grab),
        button,
    )


def _checkbox_row(
    checkbox: Checkbox,
    text: LabelText,
    size: Any,
    left: int,
    y: float,
    row_height: float,
) -> CheckboxRow:
    """Place a checkbox's row, its top at ``y`` in a panel whose left is ``left``."""
    box_left = float(left + PADDING_PX) - 0.5
    box_top = float(round(y + (row_height - BOX_PX) / 2)) - 0.5
    box = (box_left, box_top, box_left + BOX_PX, box_top + BOX_PX)
    text_block = size.overflow_top + size.height
    text_top = round(y + (row_height - text_block) / 2 + size.overflow_top)
    text_left = left + PADDING_PX + round(BOX_PX) + BOX_GAP_PX
    band = (
        box[0] - GRAB_PX,
        min(box[1], text_top) - GRAB_PX,
        text_left + size.width + GRAB_PX,
        max(box[3], text_top + size.height) + GRAB_PX,
    )
    return CheckboxRow(checkbox, text, text_left, text_top, box, band)


def panel_primitives(
    panel: Panel, theme: GridTheme, colors: Mapping[Control, ColorLike]
) -> tuple[list[Fill | Stroke | Markers], list[TextItem]]:
    """Return what draws a panel: its shapes back to front, then its texts.

    Args:
        panel: The panel.
        theme: Supplies the colours, the background and the font.
        colors: The resolved colour of each control.

    Returns:
        The shapes (the panel, then each control's: see
        :func:`row_primitives`) and the texts.
    """
    shapes: list[Fill | Stroke | Markers] = list(
        background_primitives(panel.rect, theme)
    )
    texts = []
    for row in panel.rows:
        row_shapes, text = row_primitives(row, theme, colors[row.control])
        shapes.extend(row_shapes)
        texts.append(text)
    return shapes, texts


def background_primitives(rect: ScreenRect, theme: GridTheme) -> list[Fill | Stroke]:
    """Return a panel's background and border (for sliders and the legend alike).

    Args:
        rect: The panel's area on screen.
        theme: Supplies the background and border colours.

    Returns:
        The background (unless the theme is transparent) and the border.
    """
    ring = _rounded_rect(rect)
    shapes: list[Fill | Stroke] = []
    background = panel_color(theme)
    if background is not None:
        shapes.append(Fill((ring,), background, PANEL_OPACITY))
    shapes.append(Stroke((np.vstack([ring, ring[:1]]),), theme.major_color, 1.0))
    return shapes


def row_primitives(
    row: Row, theme: GridTheme, color: ColorLike
) -> tuple[list[Fill | Stroke | Markers], TextItem]:
    """Return what draws one control's row: its shapes, back to front, and its text.

    A slider's are its track, the filled part, the hover ring and the
    handle, then its play button (with its own hover ring, and a play or a
    pause sign); a checkbox's are its hover ring and its box, outlined, or
    filled with a tick.

    Args:
        row: Where the control is drawn.
        theme: Supplies the track and text colours and the font.
        color: The control's resolved colour.

    Returns:
        The shapes, back to front, and the text.
    """
    if isinstance(row, CheckboxRow):
        return _checkbox_primitives(row, theme, color)
    slider = row.slider
    opacity = 1.0 if slider.enabled else DISABLED_OPACITY
    background = panel_color(theme)
    handle = np.array([[row.handle_x, row.y]])
    shapes: list[Fill | Stroke | Markers] = [
        Stroke(
            (np.array([[row.start, row.y], [row.end, row.y]]),),
            theme.major_color,
            TRACK_WIDTH_PX,
            opacity,
        ),
        Stroke(
            (np.array([[row.start, row.y], [row.handle_x, row.y]]),),
            color,
            TRACK_WIDTH_PX,
            opacity,
        ),
    ]
    if slider.enabled and (slider.hovered or slider.dragging):
        radius = HANDLE_RADIUS_PX + HANDLE_EDGE_PX + RING_PX
        shapes.append(Markers(handle, "circle", radius, color, opacity=RING_OPACITY))
    shapes.append(
        Markers(
            handle,
            "circle",
            HANDLE_RADIUS_PX,
            color,
            edge_color=background,
            edge_width=HANDLE_EDGE_PX if background is not None else 0.0,
            opacity=opacity,
        )
    )
    if row.button is not None:
        shapes.extend(_play_primitives(row, row.button, theme, color, opacity))
    text = TextItem(
        text=row.text,
        left=row.text_left,
        top=row.text_top,
        font=theme.label_font_spec,
        color=theme.label_color,
        opacity=opacity,
    )
    return shapes, text


def _play_primitives(
    row: SliderRow,
    center: tuple[float, float],
    theme: GridTheme,
    color: ColorLike,
    opacity: float,
) -> list[Fill | Stroke | Markers]:
    """Return a play button: its hover ring, the disc, and a play or pause sign.

    The sign is in the panel's colour, cut out of the slider's: a triangle
    while paused, two bars while playing.
    """
    slider = row.slider
    cx, cy = center
    here = np.array([[cx, cy]])
    shapes: list[Fill | Stroke | Markers] = []
    if slider.enabled and slider.play_hovered:
        radius = PLAY_RADIUS_PX + RING_PX
        shapes.append(Markers(here, "circle", radius, color, opacity=RING_OPACITY))
    shapes.append(Markers(here, "circle", PLAY_RADIUS_PX, color, opacity=opacity))
    r = PLAY_RADIUS_PX
    if slider.playing:
        bar, half = 0.22 * r, 0.42 * r
        rings = tuple(
            np.array(
                [
                    [x - bar / 2, cy - half],
                    [x + bar / 2, cy - half],
                    [x + bar / 2, cy + half],
                    [x - bar / 2, cy + half],
                ]
            )
            for x in (cx - 0.24 * r, cx + 0.24 * r)
        )
    else:  # a triangle pointing right, centred by its area
        rings = (
            np.array(
                [
                    [cx - 0.3 * r, cy - 0.48 * r],
                    [cx + 0.5 * r, cy],
                    [cx - 0.3 * r, cy + 0.48 * r],
                ]
            ),
        )
    sign = panel_color(theme)
    shapes.append(Fill(rings, sign if sign is not None else "white", opacity))
    return shapes


def _checkbox_primitives(
    row: CheckboxRow, theme: GridTheme, color: ColorLike
) -> tuple[list[Fill | Stroke | Markers], TextItem]:
    """Return what draws a checkbox: its hover ring and box, and its label."""
    checkbox = row.checkbox
    opacity = 1.0 if checkbox.enabled else DISABLED_OPACITY
    left, top, right, bottom = row.box
    shapes: list[Fill | Stroke | Markers] = []
    if checkbox.enabled and checkbox.hovered:
        ring = _rounded_box(
            left - RING_PX + 1,
            top - RING_PX + 1,
            right + RING_PX - 1,
            bottom + RING_PX - 1,
            BOX_RADIUS_PX + RING_PX - 1,
        )
        shapes.append(Fill((ring,), color, RING_OPACITY))
    if checkbox.value:
        box = _rounded_box(left, top, right, bottom, BOX_RADIUS_PX)
        shapes.append(Fill((box,), color, opacity))
        size = right - left
        tick = np.array(
            [
                [left + 0.24 * size, top + 0.52 * size],
                [left + 0.43 * size, top + 0.70 * size],
                [left + 0.77 * size, top + 0.32 * size],
            ]
        )
        sign = panel_color(theme)
        shapes.append(
            Stroke(
                (tick,), sign if sign is not None else "white", TICK_WIDTH_PX, opacity
            )
        )
    else:
        inset = BOX_EDGE_PX / 2
        box = _rounded_box(
            left + inset, top + inset, right - inset, bottom - inset, BOX_RADIUS_PX
        )
        shapes.append(
            Stroke(
                (np.vstack([box, box[:1]]),),
                theme.label_color,
                BOX_EDGE_PX,
                BOX_EDGE_OPACITY * opacity,
            )
        )
    text = TextItem(
        text=row.text,
        left=row.text_left,
        top=row.text_top,
        font=theme.label_font_spec,
        color=theme.label_color,
        opacity=opacity,
    )
    return shapes, text


def row_key(row: Row, color: ColorLike) -> tuple[object, ...]:
    """Return everything a row's primitives depend on, beyond the theme.

    Equal keys in two frames mean the same picture, so the plane can draw
    the previous frame's primitives again, and the renderer reuse their
    rasterised coverage.

    Args:
        row: Where the control is drawn.
        color: Its resolved colour.

    Returns:
        The key.
    """
    if isinstance(row, CheckboxRow):
        box = row.checkbox
        return (
            "checkbox",
            row.text,
            row.text_left,
            row.text_top,
            row.box,
            box.value,
            box.enabled,
            box.enabled and box.hovered,
            color,
        )
    slider = row.slider
    active = slider.enabled and (slider.hovered or slider.dragging)
    return (
        row.text,
        row.text_left,
        row.text_top,
        row.start,
        row.end,
        row.y,
        row.handle_x,
        slider.enabled,
        active,
        color,
        row.button,
        slider.playing,
        slider.enabled and slider.play_hovered,
    )


def place_in_corner(
    width: int,
    height: int,
    corner: str,
    rect: ScreenRect,
    occupied: ScreenRect | None = None,
) -> tuple[int, int]:
    """Return where a panel goes in a corner of the plane.

    Panels in a corner stack: a panel goes beyond the ones already there
    (``occupied``), away from the corner, :data:`MARGIN_PX` apart.

    Args:
        width: The panel's width.
        height: The panel's height.
        corner: ``"top-left"``, ``"top-right"``, ``"bottom-left"`` or
            ``"bottom-right"``.
        rect: The plane's rectangle.
        occupied: The area of the panels already in the corner, if any.

    Returns:
        The panel's ``(left, top)``.
    """
    if corner.endswith("left"):
        left = rect.left + MARGIN_PX
    else:
        left = rect.right - MARGIN_PX - width
    if corner.startswith("top"):
        top = rect.top + MARGIN_PX
        if occupied is not None:
            top = max(top, occupied.bottom + MARGIN_PX)
    else:
        bottom = rect.bottom - MARGIN_PX
        if occupied is not None:
            bottom = min(bottom, occupied.top - MARGIN_PX)
        top = bottom - height
    return left, top


def panel_color(theme: GridTheme) -> ColorLike | None:
    """The panel's colour: the theme's background, else its halo colour."""
    return (
        theme.background if theme.background is not None else theme.resolved_halo_color
    )


def _rounded_rect(rect: ScreenRect) -> np.ndarray:
    """Return the outline of a panel with rounded corners, in continuous pixels.

    The edges are on the pixel boundaries of ``rect`` (pixel centres are
    integers), and each corner is a quarter circle of eight segments.
    """
    left, top = rect.left - 0.5, rect.top - 0.5
    right, bottom = rect.right - 0.5, rect.bottom - 0.5
    r = min(CORNER_RADIUS_PX, rect.width / 2, rect.height / 2)
    return _rounded_box(left, top, right, bottom, r)


def _rounded_box(
    left: float, top: float, right: float, bottom: float, r: float
) -> np.ndarray:
    """Return the outline of a box with corners rounded to radius ``r``.

    Each corner is a quarter circle of eight segments, clockwise on screen
    from the top right.
    """
    angles = np.linspace(0.0, math.pi / 2, 9)
    corners = [
        (right - r, top + r, -math.pi / 2),
        (right - r, bottom - r, 0.0),
        (left + r, bottom - r, math.pi / 2),
        (left + r, top + r, math.pi),
    ]
    points = [
        np.column_stack(
            (cx + r * np.cos(angles + start), cy + r * np.sin(angles + start))
        )
        for cx, cy, start in corners
    ]
    return np.vstack(points)
