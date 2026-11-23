"""The panels of sliders in the corners of a plane: layout and primitives.

A plane's sliders are drawn in a panel in a corner of the plane, one panel
per corner, stacked in the order they are drawn (creation order by
default). Each slider takes a row: its text (``label = value``) above, its
track below, with the handle on the track. :func:`layout_panels` places the
rows in screen pixels, and :func:`panel_primitives` returns what to draw,
as the same backend-neutral primitives as every other plot (a
:class:`~pygridplot.core.primitives.Fill` for the panel, strokes for the
tracks, markers for the handles, and text items), so the renderer needs
nothing new.

The layout is kept by the plane until the next frame: it is what the user
sees, so presses and the wheel are matched against it
(:class:`SliderRow` gives the band that grabs each slider, and maps
the cursor to a point of the track). A slider not drawn in the last frame
cannot be dragged.

The panel has the theme's background colour, nearly opaque, and a thin
border in its major grid line colour, so it reads as a control on top of
the plot in light and dark themes alike; the text uses the theme's label
font and colour.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from pygridplot.core.coords import ColorLike, ScreenRect
from pygridplot.core.grid.theme import GridTheme
from pygridplot.core.handles.slider import CORNERS, Slider
from pygridplot.core.primitives import (
    Fill,
    LabelText,
    Markers,
    Stroke,
    TextItem,
    TextMeasurer,
)

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
    """

    slider: Slider[Any]
    text: LabelText
    text_left: int
    text_top: int
    start: float
    end: float
    y: float
    band: tuple[float, float, float, float]

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
class Panel:
    """A corner's panel of sliders.

    Attributes:
        rect: The panel's area on screen (presses there never reach the
            plane).
        rows: Its sliders, top to bottom.
    """

    rect: ScreenRect
    rows: tuple[SliderRow, ...]


def layout_panels(
    sliders: Sequence[Slider[Any]],
    rect: ScreenRect,
    theme: GridTheme,
    measurer: TextMeasurer,
) -> tuple[Panel, ...]:
    """Place sliders in panels, one per corner of the plane.

    Each slider's :attr:`~Slider.track_px` is set to the length of its track
    (which its text's precision depends on).

    Args:
        sliders: The sliders to draw, in order (top to bottom in each panel).
        rect: The plane's rectangle.
        theme: Supplies the font and number format.
        measurer: Measures the texts.

    Returns:
        The panels (none if the plane is too narrow for a track).
    """
    track = min(TRACK_PX, rect.width - 2 * (MARGIN_PX + PADDING_PX + HANDLE_RADIUS_PX))
    if track < MIN_TRACK_PX or rect.height <= 2 * MARGIN_PX:
        return ()
    font = theme.label_font_spec
    panels = []
    for corner in CORNERS:
        chosen = [slider for slider in sliders if slider.corner == corner]
        if not chosen:
            continue
        texts, sizes = [], []
        for slider in chosen:
            slider.track_px = track
            text = slider.text(theme.number_format)
            texts.append(text)
            sizes.append(measurer.measure_text(text, font))
        inner = max(track + 2 * HANDLE_RADIUS_PX, *(size.width for size in sizes))
        width = math.ceil(inner) + 2 * PADDING_PX
        heights = [
            size.overflow_top + size.height + TEXT_GAP_PX + 2 * HANDLE_RADIUS_PX
            for size in sizes
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
        rows = []
        y = float(top + PADDING_PX)
        for slider, text, size, row_height in zip(
            chosen, texts, sizes, heights, strict=True
        ):
            text_top = round(y + size.overflow_top)
            track_y = float(text_top + size.height + TEXT_GAP_PX + HANDLE_RADIUS_PX)
            start = float(left + PADDING_PX + HANDLE_RADIUS_PX)
            reach = HANDLE_RADIUS_PX + GRAB_PX
            rows.append(
                SliderRow(
                    slider,
                    text,
                    left + PADDING_PX,
                    text_top,
                    start,
                    start + track,
                    track_y,
                    (
                        start - reach,
                        track_y - reach,
                        start + track + reach,
                        track_y + reach,
                    ),
                )
            )
            y += row_height + ROW_GAP_PX
        panels.append(Panel(ScreenRect(left, top, width, height), tuple(rows)))
    return tuple(panels)


def panel_primitives(
    panel: Panel, theme: GridTheme, colors: Mapping[Slider[Any], ColorLike]
) -> tuple[list[Fill | Stroke | Markers], list[TextItem]]:
    """Return what draws a panel: its shapes back to front, then its texts.

    Args:
        panel: The panel.
        theme: Supplies the colours, the background and the font.
        colors: The resolved colour of each slider.

    Returns:
        The shapes (the panel, then each slider's track, filled part, hover
        ring and handle) and the texts.
    """
    shapes: list[Fill | Stroke | Markers] = list(
        background_primitives(panel.rect, theme)
    )
    texts = []
    for row in panel.rows:
        row_shapes, text = row_primitives(row, theme, colors[row.slider])
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
    row: SliderRow, theme: GridTheme, color: ColorLike
) -> tuple[list[Stroke | Markers], TextItem]:
    """Return what draws one slider: its track, filled part, ring and handle; its text.

    Args:
        row: Where the slider is drawn.
        theme: Supplies the track and text colours and the font.
        color: The slider's resolved colour.

    Returns:
        The shapes, back to front, and the text.
    """
    slider = row.slider
    opacity = 1.0 if slider.enabled else DISABLED_OPACITY
    background = panel_color(theme)
    handle = np.array([[row.handle_x, row.y]])
    shapes: list[Stroke | Markers] = [
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
    text = TextItem(
        text=row.text,
        left=row.text_left,
        top=row.text_top,
        font=theme.label_font_spec,
        color=theme.label_color,
        opacity=opacity,
    )
    return shapes, text


def row_key(row: SliderRow, color: ColorLike) -> tuple[object, ...]:
    """Return everything a slider's primitives depend on, beyond the theme.

    Equal keys in two frames mean the same picture, so the plane can draw
    the previous frame's primitives again, and the renderer reuse their
    rasterised coverage.

    Args:
        row: Where the slider is drawn.
        color: Its resolved colour.

    Returns:
        The key.
    """
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
