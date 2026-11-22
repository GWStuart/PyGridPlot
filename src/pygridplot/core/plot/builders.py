"""Turning plotted objects into screen-space primitives for one frame.

This is the plotting counterpart of :mod:`pygridplot.core.grid.scene`. Each
function takes world-space data, the current :class:`~pygridplot.core.view.View`
and a resolved style, and returns backend-neutral primitives (see
:mod:`pygridplot.core.primitives`) that are already clipped to the plane.
They are pure functions: everything is recomputed from the view every frame
(immediate mode), so plots follow panning and zooming exactly.

On logarithmic axes, vertices and points are transformed one by one and
joined by straight lines on screen (the usual convention for plotting data,
as in matplotlib). Functions are sampled in transformed coordinates (see
:func:`transformed_function`), so a power law is drawn as an exactly straight
line on log-log axes and the sampler's handling of jumps, poles and domain
edges works unchanged. A coordinate of exactly zero on a log axis lies
infinitely far off the edge, so a line to it runs off the plane; negative
coordinates cannot be shown at all and are treated like NaN.

* :func:`function_stroke`: the graph of ``y = f(x)``.
* :func:`parametric_stroke`: a parametric curve ``(x(t), y(t))``.
* :func:`polar_stroke`: a polar curve ``r(θ)``.
* :func:`implicit_stroke`: an implicit curve ``F(x, y) = 0``.
* :func:`area_fill`: the area between two graphs.
* :func:`region_fill`: the region where a condition holds (an inequality).
* :func:`arrow_fill`: arrows between points.
* :func:`vector_field_fills`: the arrows of a vector field.
* :func:`slope_field_strokes`: the segments of a slope field.
* :func:`polyline_stroke`: connected vertices, open or closed.
* :func:`polygon_shapes`: a filled polygon and its outline.
* :func:`point_markers`: fixed-size markers at points.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pygridplot.core.coords import ColorLike
from pygridplot.core.plot.areas import rings_between
from pygridplot.core.plot.arrows import arrow_rings, head_shape
from pygridplot.core.plot.curves import (
    DEFAULT_CURVE_SAMPLING,
    CurveSampling,
    sample_curve,
)
from pygridplot.core.plot.evaluation import (
    Function,
    PlaneFunction,
    VectorFunction,
    evaluate_quietly,
    evaluate_vectors_quietly,
    evaluate_xy_quietly,
)
from pygridplot.core.plot.fields import (
    ARROW_FRACTION,
    DEFAULT_SPACING,
    SLOPE_FRACTION,
    FieldLattice,
    check_spacing,
    field_lattice,
    screen_directions,
)
from pygridplot.core.plot.geometry import (
    Box,
    as_points,
    clamp_coordinates,
    clip_path,
    clip_paths,
    clip_polygon,
)
from pygridplot.core.plot.implicit import (
    DEFAULT_IMPLICIT_SAMPLING,
    ImplicitSampling,
    sample_implicit,
)
from pygridplot.core.plot.regions import sample_region
from pygridplot.core.plot.sampling import (
    DEFAULT_SAMPLING,
    SamplingConfig,
    sample_function,
)
from pygridplot.core.plot.style import (
    ArrowStyle,
    FillStyle,
    LineStyle,
    PointStyle,
    PolygonStyle,
)
from pygridplot.core.primitives import Fill, Markers, Stroke
from pygridplot.core.transforms import AxisTransform
from pygridplot.core.view import View

CurveComponent: TypeAlias = Function | float
"""One coordinate of a parametric curve: a function of ``t``, or a constant."""

Boundary: TypeAlias = Function | float
"""One side of an area between graphs: a function of ``x``, or a constant."""

WorldCurve: TypeAlias = Callable[
    [NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]
]
"""Maps parameter values to world points ``(x, y)``, with NaN where undefined."""

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
    span = _x_span(view, margin, domain)
    if span is None:
        return None
    band = _y_band(view, margin)
    target = transformed_function(func, view.transform_x, view.transform_y)
    scale = (view.scale_x, view.scale_y)
    polylines = sample_function(target, span, scale, band, sampling)
    box = _clip_box(view, margin)
    paths = clip_paths([_transformed_to_screen(view, line) for line in polylines], box)
    if not paths:
        return None
    return Stroke(tuple(paths), color, style.width, style.opacity)


def transformed_function(
    func: Function, x_transform: AxisTransform, y_transform: AxisTransform
) -> Function:
    """Return ``func`` as seen in transformed coordinates: ``w = T_y(f(T_x⁻¹(u)))``.

    Sampling this function on a uniform grid of ``u`` is sampling the graph
    uniformly *on screen*, which is what the adaptive sampler expects, whatever
    the kind of axes. Where ``f(x)`` is zero on a log y axis the result is
    ``-inf`` (undefined for the sampler, which then chases the curve down to
    the edge of the plane); negative values give NaN (a gap).

    Args:
        func: The user's function of world ``x``.
        x_transform: The x axis' transform.
        y_transform: The y axis' transform.

    Returns:
        ``func`` itself when both axes are linear, otherwise a vectorised
        wrapper around it (``func`` is still evaluated robustly, see
        :mod:`pygridplot.core.plot.evaluation`).
    """
    if not (x_transform.is_log or y_transform.is_log):
        return func

    def transformed(u: NDArray[np.float64]) -> NDArray[np.float64]:
        xs = x_transform.inverse(np.asarray(u, dtype=np.float64))
        return y_transform.forward(evaluate_quietly(func, xs))

    return transformed


def parametric_stroke(
    view: View,
    x: CurveComponent,
    y: CurveComponent,
    t_range: tuple[float, float],
    style: LineStyle,
    color: ColorLike,
    *,
    sampling: CurveSampling = DEFAULT_CURVE_SAMPLING,
) -> Stroke | None:
    """Sample the parametric curve ``(x(t), y(t))`` and return it as a stroke.

    Args:
        view: The view being drawn.
        x: The x coordinate as a function of ``t`` (or a constant).
        y: The y coordinate as a function of ``t`` (or a constant).
        t_range: ``(t_min, t_max)``, the parameter values to draw.
        style: Line width and opacity (its colour is ignored).
        color: The resolved line colour.
        sampling: Curve sampling settings.

    Returns:
        The stroke, or ``None`` if no part of the curve is visible.

    Raises:
        ValueError: If ``t_range`` is not an increasing pair of finite numbers.
    """
    return _curve_stroke(
        view, world_curve(x, y), check_range("t_range", t_range), style, color, sampling
    )


def polar_stroke(
    view: View,
    r: CurveComponent,
    theta_range: tuple[float, float],
    style: LineStyle,
    color: ColorLike,
    *,
    sampling: CurveSampling = DEFAULT_CURVE_SAMPLING,
) -> Stroke | None:
    """Sample the polar curve ``r(θ)`` and return it as a stroke.

    The point at angle ``θ`` (in radians, anticlockwise from the positive x
    axis) is ``(r cos θ, r sin θ)``. A negative ``r`` lies on the opposite
    side of the origin, as usual in mathematics.

    Args:
        view: The view being drawn.
        r: The radius as a function of ``θ`` (or a constant, for a circle).
        theta_range: ``(θ_min, θ_max)``, the angles to draw.
        style: Line width and opacity (its colour is ignored).
        color: The resolved line colour.
        sampling: Curve sampling settings.

    Returns:
        The stroke, or ``None`` if no part of the curve is visible.

    Raises:
        ValueError: If ``theta_range`` is not an increasing pair of finite
            numbers.
    """
    radius = _component(r)

    def world(
        theta: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        values = radius(theta)
        return values * np.cos(theta), values * np.sin(theta)

    span = check_range("theta_range", theta_range)
    return _curve_stroke(view, world, span, style, color, sampling)


def implicit_stroke(
    view: View,
    func: PlaneFunction,
    style: LineStyle,
    color: ColorLike,
    *,
    level: float | Sequence[float] = 0.0,
    sampling: ImplicitSampling = DEFAULT_IMPLICIT_SAMPLING,
) -> Stroke | None:
    """Find the implicit curve ``func(x, y) = level`` and return it as a stroke.

    The curve is found on a grid in *transformed* coordinates (see
    :mod:`pygridplot.core.plot.implicit`), whose nodes are multiples of
    :attr:`ImplicitSampling.resolution_px` pixels, so the work is even on
    screen on linear and log axes alike, and panning moves the curve
    rigidly.

    Args:
        view: The view being drawn.
        func: ``F(x, y)``, vectorised or scalar-only, evaluated robustly (see
            :mod:`pygridplot.core.plot.evaluation`).
        style: Line width and opacity (its colour is ignored).
        color: The resolved line colour.
        level: The value of ``func`` along the curve, or several values
            for several curves (one stroke), found together.
        sampling: Implicit curve sampling settings.

    Returns:
        The stroke, or ``None`` if no part of the curve is visible.

    Raises:
        ValueError: If a level is not a finite number.
    """
    levels = np.atleast_1d(np.asarray(level, dtype=np.float64))
    if levels.ndim != 1 or not np.isfinite(levels).all():
        msg = "level must be a finite number or a sequence of them"
        raise ValueError(msg)
    if view.rect.is_empty:
        return None
    margin = style.width / 2 + CLIP_MARGIN_PX
    sx, sy = view.scale_x, view.scale_y
    bounds = view.transformed_bounds
    u_range = (bounds.x_min - margin / sx, bounds.x_max + margin / sx)
    w_range = (bounds.y_min - margin / sy, bounds.y_max + margin / sy)
    step = (sampling.resolution_px / sx, sampling.resolution_px / sy)
    x_transform, y_transform = view.transform_x, view.transform_y

    def transformed(
        u: NDArray[np.float64], w: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return evaluate_xy_quietly(func, x_transform.inverse(u), y_transform.inverse(w))

    contours = sample_implicit(
        transformed, u_range, w_range, step, sampling, levels=tuple(levels)
    )
    if contours is None:
        return None
    screen = _grid_to_screen(view, contours.origin, contours.step, contours.lines)
    paths = clip_paths(screen, _clip_box(view, margin))
    if not paths:
        return None
    return Stroke(tuple(paths), color, style.width, style.opacity)


def area_fill(
    view: View,
    first: Boundary,
    second: Boundary,
    style: FillStyle,
    color: ColorLike,
    *,
    domain: tuple[float, float] | None = None,
    sampling: SamplingConfig = DEFAULT_SAMPLING,
) -> Fill | None:
    """Return the fill of the area between the graphs of ``first`` and ``second``.

    Both graphs are sampled as :func:`function_stroke` samples one (so the
    area's edges are exactly where the curves are drawn), and the area is
    built where both are defined (see :mod:`pygridplot.core.plot.areas`). A
    constant side is a horizontal line; on a log y axis, a constant of zero
    or below lies infinitely far below the plane, so the area reaches the
    bottom edge.

    Args:
        view: The view being drawn.
        first: One side: a function of ``x`` (vectorised or scalar-only,
            evaluated robustly), or a constant (``±inf`` reaches the edge of
            the plane; NaN is undefined everywhere).
        second: The other side, likewise.
        style: The fill's opacity (its colour is ignored).
        color: The resolved fill colour.
        domain: Restrict the area to ``domain[0] <= x <= domain[1]``.
        sampling: Sampling settings, for both graphs.

    Returns:
        The fill, or ``None`` if no part of the area is visible.

    Raises:
        ValueError: If ``domain`` is not an increasing pair of numbers.
        TypeError: If a side is neither callable nor a real number.
    """
    if view.rect.is_empty or style.opacity <= 0:
        return None
    span = _x_span(view, CLIP_MARGIN_PX, domain)
    if span is None:
        return None
    band = _y_band(view, CLIP_MARGIN_PX)
    sides = [_side(view, part, span, band, sampling) for part in (first, second)]
    box = _clip_box(view, CLIP_MARGIN_PX)
    rings = []
    for ring in rings_between(*sides):
        clipped = clip_polygon(_transformed_to_screen(view, ring), box)
        if clipped.shape[0] >= 3:
            rings.append(clipped)
    if not rings:
        return None
    return Fill(tuple(rings), color, style.opacity, "nonzero")


def region_fill(
    view: View,
    func: PlaneFunction,
    style: FillStyle,
    color: ColorLike,
    *,
    sampling: ImplicitSampling = DEFAULT_IMPLICIT_SAMPLING,
) -> Fill | None:
    """Return the fill of the region where ``func(x, y)`` holds (an inequality).

    The region is found on a grid in transformed coordinates, like an
    implicit curve (see :mod:`pygridplot.core.plot.regions`), so panning
    moves it rigidly and log axes need no special cases.

    Args:
        view: The view being drawn.
        func: The condition, as ``x**2 + y**2 < 4``: true (not zero) inside,
            vectorised or scalar-only, evaluated robustly (an error or NaN
            counts as outside).
        style: The fill's opacity (its colour is ignored).
        color: The resolved fill colour.
        sampling: Sampling settings (the same as for implicit curves).

    Returns:
        The fill, or ``None`` if no part of the region is visible.
    """
    if view.rect.is_empty or style.opacity <= 0:
        return None
    margin = CLIP_MARGIN_PX
    sx, sy = view.scale_x, view.scale_y
    bounds = view.transformed_bounds
    u_range = (bounds.x_min - margin / sx, bounds.x_max + margin / sx)
    w_range = (bounds.y_min - margin / sy, bounds.y_max + margin / sy)
    step = (sampling.resolution_px / sx, sampling.resolution_px / sy)
    x_transform, y_transform = view.transform_x, view.transform_y

    def transformed(
        u: NDArray[np.float64], w: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return evaluate_xy_quietly(func, x_transform.inverse(u), y_transform.inverse(w))

    region = sample_region(transformed, u_range, w_range, step, sampling)
    if region is None:
        return None
    rings = _grid_to_screen(view, region.origin, region.step, region.rings)
    return Fill(tuple(rings), color, style.opacity, "evenodd")


def arrow_fill(
    view: View,
    starts: ArrayLike,
    ends: ArrayLike,
    style: ArrowStyle,
    color: ColorLike,
) -> Fill | None:
    """Return arrows from world points ``starts`` to world points ``ends``.

    The ends are transformed one by one and joined by straight arrows on
    screen, like the vertices of a polyline; the shaft and head have fixed
    sizes in pixels (see :mod:`pygridplot.core.plot.arrows`).

    Args:
        view: The view being drawn.
        starts: Where the arrows start: one point or many (see
            :func:`~pygridplot.core.plot.geometry.as_points`).
        ends: Where they point to, likewise. One point and many are paired
            up as one with each.
        style: Shaft width, head size and opacity (its colour is ignored).
        color: The resolved colour.

    Returns:
        The arrows as one fill, or ``None`` if none is visible. Arrows with
        a NaN coordinate, or of zero length on screen, are left out.

    Raises:
        ValueError: If ``starts`` and ``ends`` have different numbers of
            points, neither of them one.
    """
    tails, tips = as_points(starts), as_points(ends)
    if tails.shape[0] != tips.shape[0] and 1 not in (tails.shape[0], tips.shape[0]):
        msg = (
            f"got {tails.shape[0]} starts and {tips.shape[0]} ends: give as many "
            "of each, or a single start or end for all"
        )
        raise ValueError(msg)
    tails, tips = np.broadcast_arrays(tails, tips)
    if view.rect.is_empty or tails.shape[0] == 0 or style.opacity <= 0:
        return None
    tails = clamp_coordinates(_to_screen(view, tails))
    tips = clamp_coordinates(_to_screen(view, tips))
    head = head_shape(style.width, style.head_size)
    # Leave out the arrows that lie wholly beyond the plane.
    reach = max(style.width, head[1]) / 2 + CLIP_MARGIN_PX
    x0, y0, x1, y1 = _clip_box(view, reach)
    with np.errstate(invalid="ignore"):
        visible = (
            (np.maximum(tails[:, 0], tips[:, 0]) >= x0)
            & (np.minimum(tails[:, 0], tips[:, 0]) <= x1)
            & (np.maximum(tails[:, 1], tips[:, 1]) >= y0)
            & (np.minimum(tails[:, 1], tips[:, 1]) <= y1)
        )
    rings = arrow_rings(tails[visible], tips[visible], style.width, head)
    if rings.shape[0] == 0:
        return None
    return Fill(tuple(rings), color, style.opacity, "nonzero")


def vector_field_fills(
    view: View,
    field: VectorFunction,
    style: ArrowStyle,
    color: ColorLike,
    *,
    spacing: float = DEFAULT_SPACING,
    scale: float | None = None,
) -> tuple[Fill, ...]:
    """Return the arrows of the vector field ``(u, v) = field(x, y)``.

    The field is sampled on an anchored lattice (see
    :mod:`pygridplot.core.plot.fields`), and an arrow is centred on each
    node, pointing along the field. Arrows still fading in (while zooming)
    are shorter and fainter; each group of equally faded arrows is one
    :class:`~pygridplot.core.primitives.Fill`.

    Args:
        view: The view being drawn.
        field: The field: returns a pair ``(u, v)``, vectorised or
            scalar-only, evaluated robustly (an error or NaN leaves out the
            arrow).
        style: Shaft width, head size and opacity (its colour is ignored).
        color: The resolved colour.
        spacing: Distance between neighbouring arrows in pixels.
        scale: ``None`` to draw every arrow the same length (the field's
            direction only), or pixels of length per unit of the vector's
            magnitude. Lengths are capped at :data:`ARROW_FRACTION` of the
            spacing, so arrows never run into each other.

    Returns:
        The fills, from the most to the least faded in (empty if nothing is
        visible).

    Raises:
        ValueError: If ``spacing`` or ``scale`` is invalid.
    """
    spacing = check_spacing(spacing)
    if scale is not None and not (math.isfinite(scale) and scale > 0):
        msg = f"scale must be a positive number or None, got {scale!r}"
        raise ValueError(msg)
    longest = ARROW_FRACTION * spacing
    lattice = field_lattice(view, spacing, longest / 2 + CLIP_MARGIN_PX)
    if lattice is None or style.opacity <= 0:
        return ()
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vectors = evaluate_vectors_quietly(field, lattice.x, lattice.y)
    u, v = vectors[:, 0], vectors[:, 1]
    directions = screen_directions(view, lattice.x, lattice.y, u, v)
    if scale is None:
        lengths = np.full(u.shape, longest)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            lengths = np.minimum(scale * np.hypot(u, v), longest)
    head = head_shape(style.width, style.head_size)
    fills = []
    for fade, half in _centred(lattice, directions, lengths):
        tails = np.column_stack((lattice.px, lattice.py))[fade.indices] - half
        tips = tails + 2 * half
        rings = arrow_rings(tails, tips, style.width, head)
        if rings.shape[0]:
            fills.append(
                Fill(tuple(rings), color, style.opacity * fade.value, "nonzero")
            )
    return tuple(fills)


def slope_field_strokes(
    view: View,
    slope: PlaneFunction,
    style: LineStyle,
    color: ColorLike,
    *,
    spacing: float = DEFAULT_SPACING,
) -> tuple[Stroke, ...]:
    """Return the segments of the slope field of ``dy/dx = slope(x, y)``.

    Each node of an anchored lattice (see :mod:`pygridplot.core.plot.fields`)
    gets a short segment centred on it, with the slope there: tangent to the
    solution curve through it. An infinite slope gives a vertical segment.

    Args:
        view: The view being drawn.
        slope: ``dy/dx`` as a function of ``(x, y)``, vectorised or
            scalar-only, evaluated robustly (an error or NaN leaves out the
            segment).
        style: Line width and opacity (its colour is ignored).
        color: The resolved colour.
        spacing: Distance between neighbouring segments in pixels.

    Returns:
        The strokes, from the most to the least faded in (empty if nothing
        is visible).

    Raises:
        ValueError: If ``spacing`` is invalid.
    """
    spacing = check_spacing(spacing)
    length = SLOPE_FRACTION * spacing
    margin = length / 2 + style.width / 2 + CLIP_MARGIN_PX
    lattice = field_lattice(view, spacing, margin)
    if lattice is None or style.opacity <= 0:
        return ()
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slopes = evaluate_xy_quietly(slope, lattice.x, lattice.y)
    # The direction (1, slope), or straight up for an infinite slope.
    steep = np.isinf(slopes)
    u = np.where(steep, 0.0, 1.0)
    v = np.where(steep, np.sign(slopes), slopes)
    directions = screen_directions(view, lattice.x, lattice.y, u, v)
    lengths = np.full(slopes.shape, length)
    strokes = []
    for fade, half in _centred(lattice, directions, lengths):
        centres = np.column_stack((lattice.px, lattice.py))[fade.indices]
        paths = np.stack((centres - half, centres + half), axis=1)
        strokes.append(
            Stroke(tuple(paths), color, style.width, style.opacity * fade.value)
        )
    return tuple(strokes)


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
        points: The vertices (see :func:`pygridplot.core.plot.geometry.as_points`).
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
        points: The points (see :func:`pygridplot.core.plot.geometry.as_points`).
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


def _curve_stroke(
    view: View,
    world: WorldCurve,
    t_range: tuple[float, float],
    style: LineStyle,
    color: ColorLike,
    sampling: CurveSampling,
) -> Stroke | None:
    """Sample a curve given in world coordinates and return it as a stroke.

    The curve is sampled in screen space (see :mod:`pygridplot.core.plot.curves`),
    through the view's transforms, so it works on linear and log axes alike.

    Args:
        view: The view being drawn.
        world: The curve in world coordinates.
        t_range: The parameter values to draw.
        style: Line width and opacity.
        color: The resolved line colour.
        sampling: Curve sampling settings.

    Returns:
        The stroke, or ``None`` if no part of the curve is visible.
    """
    if view.rect.is_empty:
        return None

    def screen(
        t: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        xs, ys = world(t)
        return view.to_screen(xs, ys)

    box = _clip_box(view, style.width / 2 + CLIP_MARGIN_PX)
    paths = clip_paths(sample_curve(screen, t_range, box, sampling), box)
    if not paths:
        return None
    return Stroke(tuple(paths), color, style.width, style.opacity)


def world_curve(x: CurveComponent, y: CurveComponent) -> WorldCurve:
    """Combine two curve components into a robust curve in world coordinates.

    Args:
        x: The x coordinate as a function of ``t`` (or a constant).
        y: The y coordinate as a function of ``t`` (or a constant).

    Returns:
        A function from parameter values to world points, never raising (NaN
        where a component is undefined). It does not silence NumPy warnings;
        callers evaluate it inside ``np.errstate``.

    Raises:
        TypeError: If a component is neither callable nor a real number.
    """
    fx, fy = _component(x), _component(y)

    def world(
        t: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return fx(t), fy(t)

    return world


@dataclass(frozen=True, slots=True, eq=False)
class _FadeGroup:
    """The lattice nodes that have faded in equally far."""

    value: float
    indices: NDArray[np.intp]


def _centred(
    lattice: FieldLattice,
    directions: NDArray[np.float64],
    lengths: NDArray[np.float64],
) -> list[tuple[_FadeGroup, NDArray[np.float64]]]:
    """Group a field's usable nodes by fade, with half of each mark's extent.

    A mark (an arrow, or a segment) is centred on its node. Marks still
    fading in are shortened by their fade, so that they grow into place
    without touching their neighbours.

    Args:
        lattice: The nodes.
        directions: Unit screen direction at each node (NaN where there is
            none).
        lengths: Length of each node's mark in pixels.

    Returns:
        ``(group, half)`` for each fade value, fully faded in first:
        ``half`` is half of each mark's extent, from its centre to its end,
        shape ``(len(group.indices), 2)``.
    """
    with np.errstate(invalid="ignore"):
        usable = np.isfinite(directions[:, 0]) & np.isfinite(lengths) & (lengths > 0)
    groups = []
    for value in np.unique(lattice.fade[usable])[::-1]:
        indices = np.flatnonzero(usable & (lattice.fade == value))
        half = directions[indices] * (lengths[indices] * value / 2)[:, None]
        groups.append((_FadeGroup(float(value), indices), half))
    return groups


def _side(
    view: View,
    part: Boundary,
    span: tuple[float, float],
    band: tuple[float, float],
    sampling: SamplingConfig,
) -> list[NDArray[np.float64]]:
    """Return one side of an area as pieces in transformed coordinates.

    Args:
        view: The view being drawn.
        part: A function of ``x``, or a constant.
        span: The transformed x interval to cover.
        band: The transformed y interval on screen (with a margin).
        sampling: Sampling settings.

    Returns:
        The pieces of the graph, in order of x (one straight piece for a
        constant, none for NaN).

    Raises:
        TypeError: If ``part`` is neither callable nor a real number.
    """
    if callable(part):
        target = transformed_function(part, view.transform_x, view.transform_y)
        scale = (view.scale_x, view.scale_y)
        return sample_function(target, span, scale, band, sampling, keep_offscreen=True)
    # A constant: a horizontal line. Beyond the band, any height draws the
    # same, so values are kept near it (infinity, or zero or below on a log
    # axis, lies far below or above). NaN is undefined everywhere.
    if isinstance(part, bool) or not isinstance(part, (int, float, np.number)):
        msg = f"expected a function of x or a number, got {part!r}"
        raise TypeError(msg)
    value = float(part)
    if math.isnan(value):
        return []
    reach = band[1] - band[0]
    low, high = band[0] - reach, band[1] + reach
    if view.log_y and value <= 0:
        level = low
    elif math.isinf(value):
        level = high if value > 0 else low
    else:
        level = min(max(float(view.transform_y.forward(value)), low), high)
    return [np.array([[span[0], level], [span[1], level]], dtype=np.float64)]


def _x_span(
    view: View, margin: float, domain: tuple[float, float] | None
) -> tuple[float, float] | None:
    """Return the transformed x interval to sample: the visible one, cut to a domain.

    Args:
        view: The view being drawn.
        margin: Pixels to add on both sides.
        domain: A world interval to restrict to, if any.

    Returns:
        ``(low, high)``, or ``None`` if the domain is not visible at all.

    Raises:
        ValueError: If ``domain`` is not an increasing pair of numbers.
    """
    bounds = view.transformed_bounds
    sx = view.scale_x
    lo, hi = bounds.x_min - margin / sx, bounds.x_max + margin / sx
    if domain is not None:
        d0, d1 = float(domain[0]), float(domain[1])
        if not d0 < d1:
            msg = f"domain must satisfy low < high, got {domain!r}"
            raise ValueError(msg)
        span = _transformed_domain(view.transform_x, d0, d1)
        if span is None:
            return None
        lo, hi = max(lo, span[0]), min(hi, span[1])
    return (lo, hi)


def _y_band(view: View, margin: float) -> tuple[float, float]:
    """Return the transformed y interval on screen, with ``margin`` pixels."""
    bounds = view.transformed_bounds
    sy = view.scale_y
    return (bounds.y_min - margin / sy, bounds.y_max + margin / sy)


def _grid_to_screen(
    view: View,
    origin: tuple[int, int],
    step: tuple[float, float],
    paths: list[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    """Map paths in grid coordinates (implicit curves, regions) to the screen.

    The grid origin's pixel is computed precisely (see
    :meth:`View.multiples_to_screen`); from there each grid step is a fixed
    number of pixels, so the result moves rigidly with the view.

    Args:
        view: The view being drawn.
        origin: Grid indices ``(i, j)`` of the paths' point ``(0, 0)``.
        step: The grid step ``(step_u, step_w)`` in transformed units.
        paths: The paths, ``(column, row)`` in grid steps.

    Returns:
        The paths in screen pixels.
    """
    step_u, step_w = step
    i0, j0 = origin
    x0 = float(view.multiples_to_screen("x", step_u, np.array([i0], dtype=np.int64))[0])
    y0 = float(view.multiples_to_screen("y", step_w, np.array([j0], dtype=np.int64))[0])
    px_u, px_w = step_u * view.scale_x, step_w * view.scale_y
    return [
        np.column_stack((x0 + path[:, 0] * px_u, y0 - path[:, 1] * px_w))
        for path in paths
    ]


def _component(
    value: CurveComponent,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Turn one coordinate of a curve into a robust, vectorised function.

    Args:
        value: A function (vectorised or scalar-only, evaluated robustly as
            described in :mod:`pygridplot.core.plot.evaluation`) or a constant.

    Returns:
        A function from parameter values to coordinates, never raising.

    Raises:
        TypeError: If ``value`` is neither callable nor a real number.
    """
    if callable(value):
        func = value
        return lambda t: evaluate_quietly(func, t)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        msg = f"expected a function of the parameter or a number, got {value!r}"
        raise TypeError(msg)
    constant = float(value)
    return lambda t: np.full(t.shape, constant)


def check_range(name: str, span: tuple[float, float]) -> tuple[float, float]:
    """Validate a parameter range.

    Args:
        name: Parameter name for error messages.
        span: ``(start, end)``.

    Returns:
        The range as floats.

    Raises:
        ValueError: If the range is not an increasing pair of finite numbers.
    """
    start, end = float(span[0]), float(span[1])
    if not (math.isfinite(start) and math.isfinite(end) and start < end):
        msg = f"{name} must be finite with start < end, got {span!r}"
        raise ValueError(msg)
    return (start, end)


def _transformed_domain(
    transform: AxisTransform, low: float, high: float
) -> tuple[float, float] | None:
    """Convert a world ``domain`` to transformed coordinates.

    Args:
        transform: The x axis' transform.
        low: Start of the domain in world units.
        high: End of the domain in world units (``high > low``).

    Returns:
        The domain in transformed coordinates (a log axis drops the part at
        or below zero), or ``None`` if none of it can be shown.
    """
    if not transform.is_log:
        return (low, high)
    if high <= 0:
        return None
    start = transform.forward(low) if low > 0 else -math.inf
    return (start, transform.forward(high))


def _transformed_to_screen(
    view: View, points: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert points in transformed coordinates to screen pixels.

    Args:
        view: The view being drawn.
        points: Transformed points, shape ``(N, 2)``.

    Returns:
        Screen points, shape ``(N, 2)``.
    """
    px, py = view.transformed_to_screen(points[:, 0], points[:, 1])
    return np.column_stack((px, py))


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
