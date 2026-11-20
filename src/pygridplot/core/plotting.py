"""Turning plotted objects into screen-space primitives for one frame.

This is the plotting counterpart of :mod:`pygridplot.core.grid`. Each
function takes world-space data, the current :class:`~pygridplot.core.view.View`
and a resolved style, and returns backend-neutral primitives (see
:mod:`pygridplot.core.primitives`) that are already clipped to the plane.
They are pure functions: everything is recomputed from the view every frame
(immediate mode), so plots follow panning and zooming exactly.

* :func:`function_stroke`: the graph of ``y = f(x)``.
* :func:`polyline_stroke`: connected vertices, open or closed.
* :func:`polygon_shapes`: a filled polygon and its outline.
* :func:`point_markers`: fixed-size markers at points.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pygridplot.core.coords import ColorLike
from pygridplot.core.evaluation import Function
from pygridplot.core.geometry import (
    Box,
    as_points,
    clip_path,
    clip_paths,
    clip_polygon,
)
from pygridplot.core.primitives import Fill, Markers, Stroke
from pygridplot.core.sampling import (
    DEFAULT_SAMPLING,
    SamplingConfig,
    sample_function,
)
from pygridplot.core.style import LineStyle, PointStyle, PolygonStyle
from pygridplot.core.view import View

CLIP_MARGIN_PX: Final = 2.0
"""Extra pixels around the plane kept when clipping, beyond half a line width.

Strokes are clipped a little outside the plane so their round ends and
antialiased edges are never visibly cut. The renderer then clips exactly to
the plane.
"""


def function_stroke(
    view: View,
    func: Function,
    style: LineStyle,
    color: ColorLike,
    *,
    domain: tuple[float, float] | None = None,
    sampling: SamplingConfig = DEFAULT_SAMPLING,
) -> Stroke | None:
    """Sample the graph of ``y = func(x)`` and return it as a stroke.

    Args:
        view: The view being drawn.
        func: The function to plot.
        style: Line width and opacity (its colour is ignored).
        color: The resolved line colour.
        domain: Restrict the graph to ``domain[0] <= x <= domain[1]``.
        sampling: Sampling settings.

    Returns:
        The stroke, or ``None`` if no part of the graph is visible.

    Raises:
        ValueError: If ``domain`` is not an increasing pair of numbers.
    """
    if view.rect.is_empty:
        return None
    margin = style.width / 2 + CLIP_MARGIN_PX
    bounds = view.bounds
    lo = bounds.x_min - margin / view.scale_x
    hi = bounds.x_max + margin / view.scale_x
    if domain is not None:
        d0, d1 = float(domain[0]), float(domain[1])
        if not d0 < d1:
            msg = f"domain must satisfy low < high, got {domain!r}"
            raise ValueError(msg)
        lo, hi = max(lo, d0), min(hi, d1)
    band = (bounds.y_min - margin / view.scale_y, bounds.y_max + margin / view.scale_y)
    polylines = sample_function(
        func, (lo, hi), (view.scale_x, view.scale_y), band, sampling
    )
    box = _clip_box(view, margin)
    paths = clip_paths([_to_screen(view, line) for line in polylines], box)
    if not paths:
        return None
    return Stroke(tuple(paths), color, style.width, style.opacity)


def polyline_stroke(
    view: View,
    points: ArrayLike,
    style: LineStyle,
    color: ColorLike,
    *,
    closed: bool = False,
) -> Stroke | None:
    """Return the stroke for a polyline through world-space vertices.

    Args:
        view: The view being drawn.
        points: The vertices (see :func:`pygridplot.core.geometry.as_points`).
            A vertex with a NaN coordinate breaks the line.
        style: Line width and opacity (its colour is ignored).
        color: The resolved line colour.
        closed: Whether to connect the last vertex back to the first.

    Returns:
        The stroke, or ``None`` if nothing is visible.
    """
    pts = as_points(points)
    if view.rect.is_empty or pts.shape[0] == 0:
        return None
    box = _clip_box(view, style.width / 2 + CLIP_MARGIN_PX)
    paths = clip_path(_to_screen(view, pts), box, closed=closed)
    if not paths:
        return None
    return Stroke(tuple(paths), color, style.width, style.opacity)


def polygon_shapes(
    view: View,
    points: ArrayLike,
    style: PolygonStyle,
    color: ColorLike,
) -> tuple[Fill | None, Stroke | None]:
    """Return the fill and outline of a polygon with world-space vertices.

    Args:
        view: The view being drawn.
        points: The vertices of the ring, in order (the ring closes itself).
        style: Outline width/opacity and fill settings (``style.color`` is
            ignored; ``style.fill_color`` overrides ``color`` for the fill).
        color: The resolved outline colour (and fill colour by default).

    Returns:
        ``(fill, outline)``. Either is ``None`` when switched off by the style
        or when nothing of it is visible.
    """
    pts = as_points(points)
    if view.rect.is_empty or pts.shape[0] == 0:
        return (None, None)
    screen = _to_screen(view, pts)
    fill = None
    if style.fill and style.fill_opacity > 0 and screen.shape[0] >= 3:
        ring = clip_polygon(screen, _clip_box(view, CLIP_MARGIN_PX))
        if ring.shape[0] >= 3:
            fill_color = style.fill_color if style.fill_color is not None else color
            fill = Fill((ring,), fill_color, style.fill_opacity, style.fill_rule)
    outline = None
    if style.width > 0 and style.opacity > 0:
        box = _clip_box(view, style.width / 2 + CLIP_MARGIN_PX)
        paths = clip_path(screen, box, closed=True)
        if paths:
            outline = Stroke(tuple(paths), color, style.width, style.opacity)
    return (fill, outline)


def point_markers(
    view: View, points: ArrayLike, style: PointStyle, color: ColorLike
) -> Markers | None:
    """Return markers for the visible points among world-space ``points``.

    Args:
        view: The view being drawn.
        points: The points (see :func:`pygridplot.core.geometry.as_points`).
            Points with a NaN coordinate are skipped.
        style: Marker shape, size, outline and opacity (its colour is ignored).
        color: The resolved fill colour.

    Returns:
        The markers, or ``None`` if none are visible.
    """
    pts = as_points(points)
    if view.rect.is_empty or pts.shape[0] == 0:
        return None
    screen = _to_screen(view, pts)
    edge_width = style.edge_width if style.edge_color is not None else 0.0
    reach = style.radius + edge_width + CLIP_MARGIN_PX
    x_min, y_min, x_max, y_max = _clip_box(view, reach)
    with np.errstate(invalid="ignore"):
        visible = (
            (screen[:, 0] >= x_min)
            & (screen[:, 0] <= x_max)
            & (screen[:, 1] >= y_min)
            & (screen[:, 1] <= y_max)
        )
    if not visible.any():
        return None
    return Markers(
        positions=screen[visible],
        shape=style.marker,
        radius=style.radius,
        color=color,
        edge_color=style.edge_color,
        edge_width=edge_width,
        opacity=style.opacity,
    )


def _to_screen(view: View, points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert world points to screen pixels.

    Args:
        view: The view being drawn.
        points: World points, shape ``(N, 2)``.

    Returns:
        Screen points, shape ``(N, 2)``.
    """
    px, py = view.to_screen(points[:, 0], points[:, 1])
    return np.column_stack((px, py))


def _clip_box(view: View, margin: float) -> Box:
    """Return the plane's pixel area grown by ``margin`` on every side.

    Args:
        view: The view being drawn.
        margin: Extra pixels on each side.

    Returns:
        The clip box in continuous pixel coordinates (pixel centres are
        integers, so the plane covers ``[left - 0.5, right - 0.5)``).
    """
    rect = view.rect
    return (
        rect.left - 0.5 - margin,
        rect.top - 0.5 - margin,
        rect.right - 0.5 + margin,
        rect.bottom - 0.5 + margin,
    )
