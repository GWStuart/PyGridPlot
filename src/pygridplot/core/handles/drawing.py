"""Drawing draggable points: markers, the hover ring and the coordinate label.

The plane draws its draggable points with these builders, which return the
same backend-neutral primitives as every other plot (:class:`Markers` and
:class:`TextItem`), so the renderer needs nothing new.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

import numpy as np

from pygridplot.core.coords import ColorLike
from pygridplot.core.grid.theme import GridTheme
from pygridplot.core.handles.labels import coordinate_text, label_position
from pygridplot.core.handles.point import DraggablePoint
from pygridplot.core.plot.builders import point_markers
from pygridplot.core.plot.style import PointStyle
from pygridplot.core.primitives import Markers, TextItem, TextMeasurer

RING_PX: Final = 5.0
"""How far the hover ring reaches beyond the point's marker, in pixels."""

RING_OPACITY: Final = 0.3
"""Opacity of the hover ring, relative to the point's own opacity."""


def drawn_style(point: DraggablePoint, background: ColorLike | None) -> PointStyle:
    """Return the style a point is drawn with.

    The point has an outline in its style's ``edge_color``, or else in the
    plane's background colour, which separates it from the curves it sits
    on (no outline if ``edge_width`` is 0 or the background is transparent).

    Args:
        point: The point.
        background: The theme's background colour, or ``None``.

    Returns:
        The style.
    """
    style = point.style
    if style.edge_color is None and background is not None and style.edge_width > 0:
        return replace(style, edge_color=background)
    return style


def reach_px(style: PointStyle) -> float:
    """Return how far a marker drawn with ``style`` extends from its centre.

    Args:
        style: The style.

    Returns:
        The radius plus the outline, in pixels.
    """
    return style.radius + (style.edge_width if style.edge_color is not None else 0.0)


def point_primitives(
    point: DraggablePoint, color: ColorLike, background: ColorLike | None
) -> list[Markers]:
    """Return the markers that draw a draggable point, back to front.

    A hovered or dragged point gets a translucent ring behind it.

    Args:
        point: The point.
        color: Its resolved fill colour.
        background: The theme's background colour (``None`` if transparent).

    Returns:
        The markers (empty if the point is not visible).
    """
    if not point.visible:
        return []
    style = drawn_style(point, background)
    position = np.array([point.position])
    shapes: list[Markers] = []
    if point.hovered or point.dragging:
        ring = PointStyle(
            radius=reach_px(style) + RING_PX,
            marker="circle",
            opacity=style.opacity * RING_OPACITY,
        )
        shapes.extend(_markers(point, position, ring, color))
    shapes.extend(_markers(point, position, style, color))
    return shapes


def coordinate_item(
    point: DraggablePoint,
    theme: GridTheme,
    measurer: TextMeasurer,
) -> TextItem | None:
    """Return the coordinate label of a point, if it should be shown now.

    Args:
        point: The point.
        theme: Supplies the number format, label font, colours and halo.
        measurer: Measures the label's size.

    Returns:
        The label, or ``None`` if it is hidden (see
        :attr:`DraggablePoint.coordinates`) or the point is not on screen
        (the label would otherwise be pushed inside the plane, away from
        the point it belongs to).
    """
    mode = point.coordinates
    active = point.hovered or point.dragging
    if not point.visible or mode == "never" or (mode == "hover" and not active):
        return None
    view = point.view
    sx, sy = point.screen_position
    if not view.rect.contains_point(sx, sy):
        return None
    text = coordinate_text(
        point.x,
        point.y,
        (view.transform_x, view.transform_y),
        (view.scale_x, view.scale_y),
        theme.number_format,
    )
    font = theme.label_font_spec
    size = measurer.measure_text(text, font)
    left, top = label_position(
        point.screen_position,
        reach_px(drawn_style(point, theme.background)),
        size,
        view.rect,
        theme.label_edge_margin,
    )
    halo = theme.resolved_halo_color
    return TextItem(
        text=text,
        left=left,
        top=top,
        font=font,
        color=theme.label_color,
        halo_color=halo,
        halo_width=theme.label_halo_width if halo is not None else 0,
    )


def _markers(
    point: DraggablePoint, position: np.ndarray, style: PointStyle, color: ColorLike
) -> list[Markers]:
    """Return a point's marker in a style, if it is visible.

    Args:
        point: The point.
        position: Its world position, shape ``(1, 2)``.
        style: The marker style.
        color: The fill colour.

    Returns:
        A list of zero or one markers.
    """
    markers = point_markers(point.view, position, style, color)
    return [] if markers is None else [markers]
