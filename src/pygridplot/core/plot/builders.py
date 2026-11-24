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
* :func:`point_markers`: fixed-size markers at points (one colour and size,
  or each its own).
* :func:`error_bar_stroke`: error bars on points.
* :func:`bar_shapes`: bars from a baseline (a bar chart).
* :func:`histogram_shapes`: the bars of a histogram of samples.
* :func:`box_plot_primitives`: box plots of groups of samples.
* :func:`violin_primitives`: violin plots of groups of samples.
* :func:`sequence_primitives`: the stems and markers of a sequence's terms.
* :func:`text_items`: text anchored to a point.

Curves, polylines and polygon outlines can be dashed or dotted (``dash=`` in
their style). Their dashes are anchored to the plane, so they stay put while
panning (see :mod:`pygridplot.core.plot.dashes`); samplers then look a little
further beyond the plane, so that every stretch of curve the pattern is
fitted to is found whole.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pygridplot.core.coords import ColorLike
from pygridplot.core.plot.areas import rings_between
from pygridplot.core.plot.arrows import arrow_rings, head_shape
from pygridplot.core.plot.curves import (
    DEFAULT_CURVE_SAMPLING,
    CurveSampling,
    sample_curve_pieces,
)
from pygridplot.core.plot.dashes import (
    DashPattern,
    dash_lattice,
    dash_margin,
    dash_paths,
    dash_pattern,
)
from pygridplot.core.plot.data import (
    AUTO_BAR_FRACTION,
    PointColors,
    Step,
    auto_bar_width,
    step_vertices,
)
from pygridplot.core.plot.distributions import (
    DEFAULT_WHISKER,
    BoxStats,
    box_stats,
    violin_shape,
)
from pygridplot.core.plot.evaluation import (
    Function,
    VectorFunction,
    XYFunction,
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
    COORDINATE_LIMIT,
    Box,
    as_points,
    clamp_coordinates,
    clip_path,
    clip_paths,
    clip_polygon,
    clip_segments,
)
from pygridplot.core.plot.histograms import (
    Bins,
    auto_bin_width,
    bin_by_edges,
    bin_samples,
)
from pygridplot.core.plot.implicit import (
    DEFAULT_IMPLICIT_SAMPLING,
    ImplicitSampling,
    sample_implicit,
)
from pygridplot.core.plot.regions import sample_region
from pygridplot.core.plot.sampling import (
    DEFAULT_SAMPLING,
    FunctionSampling,
    sample_function,
)
from pygridplot.core.plot.sequences import sequence_lattice, visible_indices
from pygridplot.core.plot.style import (
    ArrowStyle,
    BarStyle,
    FillStyle,
    LineStyle,
    PointStyle,
    PolygonStyle,
    SequenceStyle,
    TextStyle,
)
from pygridplot.core.plot.text import Anchor, place_lines, text_lines
from pygridplot.core.primitives import (
    Fill,
    FontSpec,
    Markers,
    Stroke,
    TextItem,
    TextMeasurer,
)
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

BOX_FRACTION: Final = 0.5
"""An automatic box is this fraction of the smallest gap between groups."""

OUTLIER_RADIUS_PX: Final = 2.5
"""The size of a box plot's outliers."""

SEPARATOR_FADE_START_PX: Final = 3.0
"""Bins narrower than this on screen have no lines between them."""

SEPARATOR_FADE_END_PX: Final = 8.0
"""Bins at least this wide on screen have their lines between them in full.

Between the two, the lines fade in, so zooming never makes them pop.
"""

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
    sampling: FunctionSampling = DEFAULT_SAMPLING,
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
    pattern = dash_pattern(style.dash, style.width)
    reach = margin + _dash_reach(pattern)
    span = _x_span(view, reach, domain)
    if span is None:
        return None
    band = _y_band(view, reach)
    target = transformed_function(func, view.transform_x, view.transform_y)
    scale = (view.scale_x, view.scale_y)
    polylines = sample_function(target, span, scale, band, sampling)
    screen = [_transformed_to_screen(view, line) for line in polylines]
    # A piece starting or ending at a limit of the domain ends there in the
    # world, so the pattern can start with a dash there.
    seen = _x_span(view, reach, None)
    assert seen is not None
    cut = (span[0] > seen[0], span[1] < seen[1])
    ends = [
        (cut[0] and line[0, 0] == span[0], cut[1] and line[-1, 0] == span[1])
        for line in polylines
    ]
    box = _clip_box(view, margin)
    return _line_stroke(view, screen, style, color, box, pattern, ends=ends)


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
    func: XYFunction,
    style: LineStyle,
    color: ColorLike,
    *,
    level: ArrayLike = 0.0,
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
    pattern = dash_pattern(style.dash, style.width)
    reach = margin + _dash_reach(pattern)
    sx, sy = view.scale_x, view.scale_y
    bounds = view.transformed_bounds
    u_range = (bounds.x_min - reach / sx, bounds.x_max + reach / sx)
    w_range = (bounds.y_min - reach / sy, bounds.y_max + reach / sy)
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
    return _line_stroke(view, screen, style, color, _clip_box(view, margin), pattern)


def area_fill(
    view: View,
    first: Boundary,
    second: Boundary,
    style: FillStyle,
    color: ColorLike,
    *,
    domain: tuple[float, float] | None = None,
    sampling: FunctionSampling = DEFAULT_SAMPLING,
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
    func: XYFunction,
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
    slope: XYFunction,
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
    step: Step | None = None,
) -> Stroke | None:
    """Return the stroke for a polyline through world-space vertices.

    Args:
        view: The view being drawn.
        points: The vertices (see :func:`pygridplot.core.plot.geometry.as_points`).
            A vertex with a NaN coordinate breaks the line.
        style: Line width and opacity (its colour is ignored).
        color: The resolved line colour.
        closed: Whether to connect the last vertex back to the first.
        step: Draw steps between the vertices instead of straight lines
            (see :data:`pygridplot.core.plot.data.Step`). They are made in
            transformed coordinates, so a ``"mid"`` step is half way on
            screen on a log axis too.

    Returns:
        The stroke, or ``None`` if nothing is visible.
    """
    pts = as_points(points)
    if view.rect.is_empty or pts.shape[0] == 0:
        return None
    box = _clip_box(view, style.width / 2 + CLIP_MARGIN_PX)
    if step is None:
        screen = _to_screen(view, pts)
    else:
        u, w = step_vertices(
            view.transform_x.forward(pts[:, 0]),
            view.transform_y.forward(pts[:, 1]),
            step,
        )
        with np.errstate(invalid="ignore"):
            screen = np.column_stack(view.transformed_to_screen(u, w))
    pattern = dash_pattern(style.dash, style.width)
    if pattern is not None:
        runs = _runs(screen, closed=closed)
        return _dashed_stroke(
            view,
            runs,
            pattern,
            box,
            color,
            style,
            ends=[(True, True)] * len(runs),
            corners=True,
        )
    paths = clip_path(screen, box, closed=closed)
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
        pattern = dash_pattern(style.dash, style.width)
        if pattern is not None:
            runs = _runs(screen, closed=True)
            outline = _dashed_stroke(
                view,
                runs,
                pattern,
                box,
                color,
                style,
                ends=[(True, True)] * len(runs),
                corners=True,
            )
        else:
            paths = clip_path(screen, box, closed=True)
            if paths:
                outline = Stroke(tuple(paths), color, style.width, style.opacity)
    return (fill, outline)


def point_markers(
    view: View,
    points: ArrayLike,
    style: PointStyle,
    color: ColorLike,
    *,
    colors: PointColors | None = None,
    radii: NDArray[np.float64] | None = None,
) -> Markers | None:
    """Return markers for the visible points among world-space ``points``.

    Args:
        view: The view being drawn.
        points: The points (see :func:`pygridplot.core.plot.geometry.as_points`).
            Points with a NaN coordinate are skipped.
        style: Marker shape, size, outline and opacity (its colour is ignored).
        color: The resolved fill colour.
        colors: A colour per point, replacing ``color`` (see
            :func:`pygridplot.core.plot.data.per_point_colors`), or ``None``.
        radii: A size per point, replacing ``style.radius``; points of size 0
            get no marker.

    Returns:
        The markers, or ``None`` if none are visible.
    """
    pts = as_points(points)
    if view.rect.is_empty or pts.shape[0] == 0:
        return None
    if radii is None and style.radius <= 0:
        return None
    screen = _to_screen(view, pts)
    edge_width = style.edge_width if style.edge_color is not None else 0.0
    size = style.radius if radii is None else radii
    reach = size + edge_width + CLIP_MARGIN_PX
    rect = view.rect
    with np.errstate(invalid="ignore"):
        visible = (
            (screen[:, 0] >= rect.left - 0.5 - reach)
            & (screen[:, 0] <= rect.right - 0.5 + reach)
            & (screen[:, 1] >= rect.top - 0.5 - reach)
            & (screen[:, 1] <= rect.bottom - 0.5 + reach)
        )
    if radii is not None:
        visible &= radii > 0
    if not visible.any():
        return None
    shown: PointColors | None = None
    if isinstance(colors, np.ndarray):
        shown = colors[visible]
    elif colors is not None:
        shown = tuple(
            c for c, keep in zip(colors, visible.tolist(), strict=True) if keep
        )
    return Markers(
        positions=screen[visible],
        shape=style.marker,
        radius=style.radius,
        color=color,
        edge_color=style.edge_color,
        edge_width=edge_width,
        opacity=style.opacity,
        colors=shown,
        radii=None if radii is None else radii[visible],
    )


def error_bar_stroke(
    view: View,
    points: ArrayLike,
    style: PointStyle,
    color: ColorLike,
    *,
    x_error: NDArray[np.float64] | None = None,
    y_error: NDArray[np.float64] | None = None,
) -> Stroke | None:
    """Return the error bars of points: a segment across each, with caps.

    A y error bar runs from ``y - below`` to ``y + above`` at the point's x,
    and an x error bar likewise across x; both are in world units, so they
    scale with the zoom, while their width and caps are in pixels. On a log
    axis an end at or below zero runs off the plane, and has no cap.

    Args:
        view: The view being drawn.
        points: The points (see :func:`pygridplot.core.plot.geometry.as_points`).
        style: The error bars' width and caps, and the opacity.
        color: The resolved colour.
        x_error: Lengths left and right of each point, shape ``(2, N)``
            (see :func:`pygridplot.core.plot.data.as_errors`), or ``None``.
        y_error: Lengths below and above each point, likewise.

    Returns:
        One stroke with every bar and cap, or ``None`` if none is visible.
    """
    pts = as_points(points)
    if view.rect.is_empty or pts.shape[0] == 0:
        return None
    half_cap = style.error_cap / 2
    pieces: list[NDArray[np.float64]] = []
    sides: tuple[tuple[Literal["x", "y"], NDArray[np.float64] | None], ...] = (
        ("y", y_error),
        ("x", x_error),
    )
    for along, errors in sides:
        if errors is None:
            continue
        across: Literal["x", "y"] = "x" if along == "y" else "y"
        index = 1 if along == "y" else 0
        value, fixed = pts[:, index], pts[:, 1 - index]
        transform = view.axis_transform(along)
        with np.errstate(invalid="ignore", divide="ignore"):
            low = transform.forward(value - errors[0])
            high = transform.forward(value + errors[1])
            if transform.is_log:  # an end at or below zero is off the plane
                low = np.where(value - errors[0] <= 0, -np.inf, low)
            at = view.axis_to_screen(across, view.axis_transform(across).forward(fixed))
        keep = np.isfinite(at) & ~np.isnan(low) & ~np.isnan(high) & (high > low)
        if not keep.any():
            continue
        at, low, high = at[keep], low[keep], high[keep]
        ends = np.column_stack(
            (view.axis_to_screen(along, low), view.axis_to_screen(along, high))
        )
        ends = np.clip(ends, -COORDINATE_LIMIT, COORDINATE_LIMIT)
        bars = np.stack((np.column_stack((at, ends[:, 0])),
                         np.column_stack((at, ends[:, 1]))), axis=1)  # fmt: skip
        caps = []
        if half_cap > 0:
            for end, finite in (
                (ends[:, 0], np.isfinite(low)),
                (ends[:, 1], np.isfinite(high)),
            ):
                cap = np.stack((np.column_stack((at[finite] - half_cap, end[finite])),
                                np.column_stack((at[finite] + half_cap, end[finite]))),
                               axis=1)  # fmt: skip
                caps.append(cap)
        segments = np.concatenate([bars, *caps])
        if along == "x":
            segments = segments[:, :, ::-1]  # (along, across) -> (x, y)
        pieces.append(segments)
    if not pieces:
        return None
    # The bars and caps are separate segments: clipped all at once, as many
    # thousands of them would be slow one path at a time.
    segments = np.concatenate(pieces)
    box = _clip_box(view, style.error_width / 2 + CLIP_MARGIN_PX)
    keep, t0, t1 = clip_segments(segments[:, 0], segments[:, 1], box)
    if not keep.any():
        return None
    start, end, t0, t1 = segments[keep, 0], segments[keep, 1], t0[keep], t1[keep]
    step = end - start
    clipped = np.stack((start + t0[:, None] * step, start + t1[:, None] * step), axis=1)
    return Stroke(clipped, color, style.error_width, style.opacity)


def bar_shapes(
    view: View,
    bars: NDArray[np.float64],
    style: BarStyle,
    color: ColorLike,
    *,
    width: float | None = None,
    baseline: float | NDArray[np.float64] = 0.0,
    horizontal: bool = False,
) -> tuple[Fill | None, Stroke | None]:
    """Return the fill and outlines of bars standing on a baseline.

    Each bar is a rectangle from its baseline to its top, ``width`` wide and
    centred on its position. Upright bars stand at an x and reach a y;
    horizontal bars stand at a y and reach an x. Positions and values are
    transformed (``log10`` on a log axis), so on a log axis along the bars'
    positions the width is in decades and bars look alike, and on a log axis
    along their values a baseline of 0 (or below) is infinitely far off, so
    the bars reach the plane's edge.

    Args:
        view: The view being drawn.
        bars: The bars' ``(position, top)``, shape ``(N, 2)``: ``(x, y)``
            upright, ``(y, x)`` horizontal. A NaN leaves a bar out.
        style: Fill opacity and outline width (its colour is ignored).
        color: The resolved colour.
        width: The bars' width in transformed position units, or ``None``
            for :func:`~pygridplot.core.plot.data.auto_bar_width`.
        baseline: Where the bars stand: one value for all, or one per bar
            (stacked or floating bars).
        horizontal: Lay the bars along x instead of upright.

    Returns:
        ``(fill, outlines)``: all the bars as one fill, and their outlines
        as one stroke (drawn once where neighbouring bars share an edge);
        either is ``None`` when switched off or nothing is visible.
    """
    if view.rect.is_empty or bars.shape[0] == 0:
        return (None, None)
    along: Literal["x", "y"] = "y" if horizontal else "x"
    across: Literal["x", "y"] = "x" if horizontal else "y"
    values = view.axis_transform(across)
    bases = np.broadcast_to(np.asarray(baseline, dtype=np.float64), bars.shape[:1])
    with np.errstate(invalid="ignore", divide="ignore"):
        u = view.axis_transform(along).forward(bars[:, 0])
        top = values.forward(bars[:, 1])
        base = values.forward(bases)
    if values.is_log:  # zero or below is infinitely far off the edge
        top = np.where(bars[:, 1] <= 0, -np.inf, top)
        base = np.where(bases <= 0, -np.inf, base)
    half = (width if width is not None else auto_bar_width(u)) / 2
    keep = np.isfinite(u) & ~np.isnan(top) & ~np.isnan(base) & (top != base)
    if not keep.any():
        return (None, None)
    u, top, base = u[keep], top[keep], base[keep]
    rings = _rectangles(
        view, u - half, u + half, top, base, style.edge_width, horizontal=horizontal
    )
    return _bar_primitives(rings, style, color)


def histogram_shapes(
    view: View,
    samples: NDArray[np.float64],
    style: BarStyle,
    color: ColorLike,
    *,
    width: float | None = None,
    edges: NDArray[np.float64] | None = None,
    density: bool = False,
    cumulative: bool = False,
) -> tuple[Fill | None, Stroke | None, Stroke | None]:
    """Return the bars of a histogram of samples.

    The samples are counted in bins (see
    :mod:`pygridplot.core.plot.histograms`): anchored bins of ``width``, or
    of an automatic width, or bins with explicit ``edges``. Each bin with
    samples is a bar from 0 to its count, or to the density the samples
    estimate there (count over the number of samples and the bin's width in
    x units, so that the bars' area is 1 and a probability density plotted
    over them matches). Neighbouring bars are drawn as one shape with one
    outline, and the edges between them as separate lines, which fade out
    as the bins narrow to a few pixels, so a histogram zoomed far out shows
    its outline rather than a comb of lines.

    Args:
        view: The view being drawn.
        samples: The samples; NaN and infinite ones (and on a log x axis,
            ones not positive) are left out.
        style: Fill opacity and outline width (its colour is ignored).
        color: The resolved colour.
        width: The bins' width in transformed x units (decades on a log x
            axis), or ``None`` for an automatic one.
        edges: Explicit bin edges in x units, replacing ``width``.
        density: Draw densities instead of counts (with ``cumulative``,
            fractions of the samples).
        cumulative: Draw each bin as high as all the samples up to its
            right edge, rising to the number of samples (or 1).

    Returns:
        ``(fill, outline, separators)``; each is ``None`` when switched off
        or nothing is visible.
    """
    nothing: tuple[None, None, None] = (None, None, None)
    with np.errstate(invalid="ignore"):
        chosen = samples[np.isfinite(samples) & ((samples > 0) | (not view.log_x))]
    if view.rect.is_empty or chosen.shape[0] == 0:
        return nothing
    transform = view.transform_x
    if edges is not None:
        bins = bin_by_edges(chosen, edges, transform)
        total = float(np.count_nonzero((chosen >= edges[0]) & (chosen <= edges[-1])))
    else:
        u = transform.forward(chosen)
        bins = bin_samples(u, width if width is not None else auto_bin_width(u))
        total = float(bins.counts.sum())
    if bins.index.shape[0] == 0:
        return nothing
    if cumulative:
        # Each bin as high as all the samples up to its right edge (as a
        # fraction with density); the stretches between bins keep the
        # height reached, so the histogram is one rising shape.
        rising = np.cumsum(bins.counts)
        if density:
            rising = rising / total
        bins = _without_gaps(Bins(bins.index, bins.left, bins.right, rising))
    elif density:
        spans = transform.inverse(bins.right) - transform.inverse(bins.left)
        bins = Bins(bins.index, bins.left, bins.right, bins.counts / (total * spans))
    visible = _visible_bins(view, bins, style.edge_width / 2 + CLIP_MARGIN_PX)
    if visible is None:
        return nothing
    bins = visible
    heights = bins.counts
    with np.errstate(divide="ignore"):
        top = view.transform_y.forward(heights)
        base = float(view.transform_y.forward(0.0))
    margin = style.edge_width / 2 + CLIP_MARGIN_PX
    x_min, y_min, x_max, y_max = _clip_box(view, margin)
    # Everything on screen at once, clamped to just beyond the plane (a
    # shape of horizontal and vertical edges stays the same inside the box).
    with np.errstate(invalid="ignore"):
        lx = np.clip(view.axis_to_screen("x", bins.left), x_min, x_max)
        rx = np.clip(view.axis_to_screen("x", bins.right), x_min, x_max)
        ty = np.clip(view.axis_to_screen("y", top), y_min, y_max)
        by = float(np.clip(view.axis_to_screen("y", base), y_min, y_max))
    breaks = np.flatnonzero(np.diff(bins.index) != 1) + 1
    starts, stops = np.r_[0, breaks], np.r_[breaks, bins.index.shape[0]]
    # A run of neighbouring bins is one ring: up the left edge, along the
    # tops (stepping at each shared edge) and down the right edge.
    corners_x = np.column_stack((lx, rx)).ravel()
    corners_y = np.repeat(ty, 2)
    rings = [
        np.column_stack(
            (
                np.r_[lx[first], corners_x[2 * first : 2 * last], rx[last - 1]],
                np.r_[by, corners_y[2 * first : 2 * last], by],
            )
        )
        for first, last in zip(starts.tolist(), stops.tolist(), strict=True)
    ]
    # The edges shared within runs: from the baseline to the lower bar.
    inner = np.ones(bins.index.shape[0] - 1, dtype=bool)
    inner[breaks - 1] = False
    shared_y = np.where(np.abs(ty[:-1] - by) < np.abs(ty[1:] - by), ty[:-1], ty[1:])
    separators = list(
        np.stack(
            (
                np.column_stack((rx[:-1][inner], np.full(int(inner.sum()), by))),
                np.column_stack((rx[:-1][inner], shared_y[inner])),
            ),
            axis=1,
        )
    )
    fill, outline = _bar_primitives(rings, style, color)
    lines = None
    if style.edge_width > 0 and separators:
        pixels = float(np.median(bins.right - bins.left)) * view.scale_x
        fade = (pixels - SEPARATOR_FADE_START_PX) / (
            SEPARATOR_FADE_END_PX - SEPARATOR_FADE_START_PX
        )
        opacity = min(max(fade, 0.0), 1.0)
        if opacity > 0:
            lines = Stroke(tuple(separators), color, style.edge_width, opacity)
    return (fill, outline, lines)


def _without_gaps(bins: Bins) -> Bins:
    """Return bins with the stretches between them filled at the height before.

    For a cumulative histogram: where there are no samples, the count so far
    holds until the next bin.

    Args:
        bins: The bins, their ``counts`` the heights.

    Returns:
        The bins and the gaps between them, numbered consecutively.
    """
    gaps = np.flatnonzero(np.diff(bins.index) != 1)
    if gaps.size:
        left = np.insert(bins.left, gaps + 1, bins.right[gaps])
        right = np.insert(bins.right, gaps + 1, bins.left[gaps + 1])
        heights = np.insert(bins.counts, gaps + 1, bins.counts[gaps])
    else:
        left, right, heights = bins.left, bins.right, bins.counts
    return Bins(np.arange(left.shape[0], dtype=np.int64), left, right, heights)


def box_plot_primitives(
    view: View,
    groups: list[NDArray[np.float64]],
    positions: NDArray[np.float64],
    style: BarStyle,
    color: ColorLike,
    *,
    width: float | None = None,
    whisker: float = DEFAULT_WHISKER,
    horizontal: bool = False,
) -> tuple[Fill | None, Stroke | None, Stroke | None, Markers | None]:
    """Return the boxes, whiskers, medians and outliers of groups of samples.

    Each group's box spans its quartiles, ``width`` wide about its position,
    with the median across it and whiskers with caps out to the furthest
    samples within reach (see :func:`pygridplot.core.plot.distributions.box_stats`);
    samples beyond are outliers, drawn as small points. The statistics are
    of the transformed samples (``log10`` on a log axis), as shown.

    Args:
        view: The view being drawn.
        groups: The groups of samples.
        positions: Each group's position (x, or y when ``horizontal``).
        style: The boxes' fill opacity and outline width.
        color: The resolved colour.
        width: The boxes' width in transformed position units, or ``None``
            for half the smallest gap between groups.
        whisker: The whiskers' reach, in interquartile ranges.
        horizontal: Lay the boxes along x.

    Returns:
        ``(boxes, lines, medians, outliers)``: the boxes' fill, their
        outlines with the whiskers and caps, the medians (a little thicker),
        and the outliers; each ``None`` when there is nothing to draw.
    """
    along: Literal["x", "y"] = "y" if horizontal else "x"
    across: Literal["x", "y"] = "x" if horizontal else "y"
    places = view.axis_transform(along).forward(positions)
    half = (width if width is not None else _auto_width(places, BOX_FRACTION)) / 2
    values = view.axis_transform(across)
    boxes: list[tuple[float, BoxStats]] = []
    for place, group in zip(places.tolist(), groups, strict=True):
        if not math.isfinite(place):
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            stats = box_stats(np.asarray(values.forward(group)), whisker)
        if stats is not None:
            boxes.append((place, stats))
    if view.rect.is_empty or not boxes:
        return (None, None, None, None)
    u = np.array([place for place, _ in boxes])
    q1 = np.array([b.q1 for _, b in boxes])
    q3 = np.array([b.q3 for _, b in boxes])
    rings = _rectangles(
        view, u - half, u + half, q3, q1, style.edge_width, horizontal=horizontal
    )
    fill = (
        Fill(tuple(rings), color, style.opacity)
        if rings and style.opacity > 0
        else None
    )
    lines_width = max(style.edge_width, 1.0)
    segments = []
    medians = []
    for place, b in boxes:
        corners = [
            (place - half, b.q1),
            (place + half, b.q1),
            (place + half, b.q3),
            (place - half, b.q3),
            (place - half, b.q1),
        ]
        segments.append(corners)
        segments.append([(place, b.q1), (place, b.low)])
        segments.append([(place, b.q3), (place, b.high)])
        for end in (b.low, b.high):
            segments.append([(place - half / 2, end), (place + half / 2, end)])
        medians.append([(place - half, b.median), (place + half, b.median)])
    box = _clip_box(view, lines_width + CLIP_MARGIN_PX)
    lines = _place_paths(view, segments, box, horizontal)
    middle = _place_paths(view, medians, box, horizontal)
    outline = Stroke(tuple(lines), color, lines_width) if lines else None
    median = Stroke(tuple(middle), color, lines_width + 1.0) if middle else None
    spots = [(place, float(v)) for place, b in boxes for v in b.outliers.tolist()]
    markers = None
    if spots:
        where = _place_points(view, np.array(spots), horizontal)
        markers = point_markers_at(view, where, OUTLIER_RADIUS_PX, color)
    return (fill, outline, median, markers)


def violin_primitives(
    view: View,
    groups: list[NDArray[np.float64]],
    positions: NDArray[np.float64],
    style: BarStyle,
    color: ColorLike,
    *,
    width: float | None = None,
    bandwidth: float | None = None,
    horizontal: bool = False,
) -> tuple[Fill | None, Stroke | None, Stroke | None]:
    """Return the violins of groups of samples: their densities, mirrored.

    Each group's violin is its density (see
    :func:`pygridplot.core.plot.distributions.violin_shape`) drawn either side
    of its position, ``width`` wide where the density is greatest, from the
    smallest sample to the largest, with the median as a line across.

    Args:
        view: The view being drawn.
        groups: The groups of samples.
        positions: Each group's position (x, or y when ``horizontal``).
        style: The violins' fill opacity and outline width.
        color: The resolved colour.
        width: The violins' greatest width in transformed position units, or
            ``None`` for 0.8 of the smallest gap between groups.
        bandwidth: The kernel's bandwidth in transformed value units, or
            ``None`` for Silverman's rule.
        horizontal: Lay the violins along x.

    Returns:
        ``(fill, outline, medians)``, each ``None`` with nothing to draw.
    """
    along: Literal["x", "y"] = "y" if horizontal else "x"
    across: Literal["x", "y"] = "x" if horizontal else "y"
    places = view.axis_transform(along).forward(positions)
    half = (width if width is not None else _auto_width(places, AUTO_BAR_FRACTION)) / 2
    values = view.axis_transform(across)
    rings = []
    medians = []
    for place, group in zip(places.tolist(), groups, strict=True):
        if not math.isfinite(place):
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            data = np.asarray(values.forward(group))
        shape = violin_shape(data, bandwidth)
        if shape is None:
            continue
        v, density = shape
        spread = half * density / np.max(density)
        ring = np.column_stack(
            (np.r_[place + spread, place - spread[::-1]], np.r_[v, v[::-1]])
        )
        rings.append(ring)
        centre = float(np.median(data[np.isfinite(data)]))
        reach = float(np.interp(centre, v, spread))
        medians.append([(place - reach, centre), (place + reach, centre)])
    if view.rect.is_empty or not rings:
        return (None, None, None)
    screen = [_place_points(view, ring, horizontal) for ring in rings]
    fill = None
    if style.opacity > 0:
        clipped = [
            clip_polygon(ring, _clip_box(view, CLIP_MARGIN_PX)) for ring in screen
        ]
        kept = tuple(ring for ring in clipped if ring.shape[0] >= 3)
        fill = Fill(kept, color, style.opacity) if kept else None
    box = _clip_box(view, max(style.edge_width, 1.0) + CLIP_MARGIN_PX)
    outline = None
    if style.edge_width > 0:
        paths = [p for ring in screen for p in clip_path(ring, box, closed=True)]
        outline = Stroke(tuple(paths), color, style.edge_width) if paths else None
    middle = _place_paths(view, medians, box, horizontal)
    median = (
        Stroke(tuple(middle), color, max(style.edge_width, 1.0) + 1.0)
        if middle
        else None
    )
    return (fill, outline, median)


def _auto_width(places: NDArray[np.float64], fraction: float) -> float:
    """Return a fraction of the smallest gap between positions (of 1 for one)."""
    return auto_bar_width(places) / AUTO_BAR_FRACTION * fraction


def _place_points(
    view: View, points: NDArray[np.float64], horizontal: bool
) -> NDArray[np.float64]:
    """Return ``(position, value)`` points in transformed units on screen.

    Args:
        view: The view being drawn.
        points: Rows of ``(position, value)``, transformed.
        horizontal: Whether positions are along y.

    Returns:
        Screen points, clamped to ``±COORDINATE_LIMIT``.
    """
    if horizontal:
        u, w = points[:, 1], points[:, 0]
    else:
        u, w = points[:, 0], points[:, 1]
    with np.errstate(invalid="ignore"):
        px, py = view.transformed_to_screen(u, w)
    return clamp_coordinates(np.column_stack((px, py)))


def _place_paths(
    view: View,
    paths: list[list[tuple[float, float]]],
    box: Box,
    horizontal: bool,
) -> list[NDArray[np.float64]]:
    """Return polylines of ``(position, value)`` vertices on screen, clipped.

    Args:
        view: The view being drawn.
        paths: The polylines, transformed.
        box: The clip box.
        horizontal: Whether positions are along y.

    Returns:
        The visible pieces.
    """
    if not paths:
        return []
    placed = [
        _place_points(view, np.array(path, dtype=np.float64), horizontal)
        for path in paths
    ]
    return clip_paths(placed, box)


def point_markers_at(
    view: View, screen: NDArray[np.float64], radius: float, color: ColorLike
) -> Markers | None:
    """Return small filled markers at screen points that reach the plane.

    Args:
        view: The view being drawn.
        screen: The points, in screen pixels.
        radius: The markers' size.
        color: Their colour.

    Returns:
        The markers, or ``None`` if none is visible.
    """
    x_min, y_min, x_max, y_max = _clip_box(view, radius + CLIP_MARGIN_PX)
    with np.errstate(invalid="ignore"):
        visible = (
            (screen[:, 0] >= x_min)
            & (screen[:, 0] <= x_max)
            & (screen[:, 1] >= y_min)
            & (screen[:, 1] <= y_max)
        )
    if not visible.any():
        return None
    return Markers(screen[visible], "circle", radius, color)


def _visible_bins(view: View, bins: Bins, margin: float) -> Bins | None:
    """Return the bins that reach the plane (grown by ``margin`` pixels).

    Bins are in order along x, so the visible ones are a contiguous range;
    a run of bins cut at the ends of the range then ends beyond the plane.

    Args:
        view: The view being drawn.
        bins: The bins.
        margin: Extra pixels on each side.

    Returns:
        The visible bins, or ``None`` if none is.
    """
    rect = view.rect
    low = view.screen_to_axis("x", rect.left - 0.5 - margin)
    high = view.screen_to_axis("x", rect.right - 0.5 + margin)
    first = int(np.searchsorted(bins.right, low, side="left"))
    last = int(np.searchsorted(bins.left, high, side="right"))
    if last <= first:
        return None
    return Bins(
        bins.index[first:last],
        bins.left[first:last],
        bins.right[first:last],
        bins.counts[first:last],
    )


def _rectangles(
    view: View,
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    top: NDArray[np.float64],
    base: NDArray[np.float64],
    edge_width: float,
    *,
    horizontal: bool = False,
) -> list[NDArray[np.float64]]:
    """Return the visible ones of rectangles given in transformed coordinates.

    Args:
        view: The view being drawn.
        start: Each rectangle's edge before its position (transformed x, or
            y when ``horizontal``).
        end: Each rectangle's edge after its position.
        top: Each rectangle's far end (transformed y, or x; may be infinite).
        base: Each rectangle's near end, its baseline (likewise).
        edge_width: The outline's width, which they keep room for.
        horizontal: Whether positions are along y and values along x.

    Returns:
        Each visible rectangle as a ring in screen pixels, clamped to just
        beyond the plane (an axis-aligned rectangle stays one when clamped).
    """
    box = _clip_box(view, edge_width / 2 + CLIP_MARGIN_PX)
    x_min, y_min, x_max, y_max = box
    along: Literal["x", "y"] = "y" if horizontal else "x"
    across: Literal["x", "y"] = "x" if horizontal else "y"
    limit = COORDINATE_LIMIT
    with np.errstate(invalid="ignore"):
        a0 = np.clip(view.axis_to_screen(along, start), -limit, limit)
        a1 = np.clip(view.axis_to_screen(along, end), -limit, limit)
        v0 = np.clip(view.axis_to_screen(across, base), -limit, limit)
        v1 = np.clip(view.axis_to_screen(across, top), -limit, limit)
    if horizontal:
        left, right = np.minimum(v0, v1), np.maximum(v0, v1)
        upper, lower = np.minimum(a0, a1), np.maximum(a0, a1)
    else:
        left, right = np.minimum(a0, a1), np.maximum(a0, a1)
        upper, lower = np.minimum(v0, v1), np.maximum(v0, v1)
    visible = (right >= x_min) & (left <= x_max) & (lower >= y_min) & (upper <= y_max)
    if not visible.any():
        return []
    left, right = (
        np.clip(left[visible], x_min, x_max),
        np.clip(right[visible], x_min, x_max),
    )
    upper, lower = (
        np.clip(upper[visible], y_min, y_max),
        np.clip(lower[visible], y_min, y_max),
    )
    rings = np.stack(
        (
            np.column_stack((left, upper)),
            np.column_stack((right, upper)),
            np.column_stack((right, lower)),
            np.column_stack((left, lower)),
        ),
        axis=1,
    )
    return list(rings)


def _bar_primitives(
    rings: list[NDArray[np.float64]], style: BarStyle, color: ColorLike
) -> tuple[Fill | None, Stroke | None]:
    """Return the fill and outline of bars' rings.

    Args:
        rings: The rings in screen pixels.
        style: Fill opacity and outline width.
        color: The colour.

    Returns:
        ``(fill, outline)``, each ``None`` when switched off or empty.
    """
    if not rings:
        return (None, None)
    fill = Fill(tuple(rings), color, style.opacity) if style.opacity > 0 else None
    outline = None
    if style.edge_width > 0:
        closed = tuple(np.vstack((ring, ring[:1])) for ring in rings)
        outline = Stroke(closed, color, style.edge_width, 1.0)
    return (fill, outline)


def sequence_primitives(
    view: View,
    terms: Function | NDArray[np.float64],
    start: int,
    style: SequenceStyle,
    color: ColorLike,
    *,
    baseline: float = 0.0,
) -> tuple[Stroke | Markers, ...]:
    """Return the stems and markers of the visible terms of a sequence.

    Term ``aₙ`` gets a stem from ``(n, baseline)`` to ``(n, aₙ)`` and a marker
    at its end. A finite sequence (an array) has its terms at ``start``,
    ``start + 1``, ... and is drawn in full; an infinite one (a function of
    ``n``) has a term at every ``n >= start``, thinned where the terms are
    closer than a pixel apart (see :mod:`pygridplot.core.plot.sequences`).
    On a log y axis, a baseline of 0 (or below) is infinitely far down, so
    the stems reach the bottom edge; terms that cannot be shown (NaN or
    infinite, or not positive on a log y axis) are left out.

    Args:
        view: The view being drawn.
        terms: The terms: an array, or a function of ``n`` taking a float
            array of whole numbers, evaluated robustly.
        start: The index of the first term.
        style: Stem width, marker size and shape, and opacity (its colour is
            ignored).
        color: The resolved colour.
        baseline: Where the stems start (a finite number).

    Returns:
        The stems (one stroke per level of fade) and then the markers (one
        primitive per level of fade); empty if nothing is visible. Where
        several terms fall in one pixel column, their stems are drawn as one,
        from the column's lowest point to its highest, which covers exactly
        what they cover together: so a dense sequence costs one stem per
        pixel, however many terms it has. Every term keeps its marker.
    """
    reach = max(style.radius, style.width / 2) + CLIP_MARGIN_PX
    if callable(terms):
        lattice = sequence_lattice(view, start, reach)
        if lattice is None:
            return ()
        n, fade = lattice.n, lattice.fade
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = evaluate_quietly(terms, n.astype(np.float64))
    else:
        visible = visible_indices(view, start, reach)
        if visible is None:
            return ()
        first, last = visible[0], min(visible[1], start + terms.shape[0] - 1)
        if last < first:
            return ()
        n = np.arange(first, last + 1, dtype=np.int64)
        values = terms[n - start]
        fade = np.ones(n.shape)
    with np.errstate(all="ignore"):
        px = (
            view.to_screen_x(n.astype(np.float64))
            if view.log_x
            else view.multiples_to_screen("x", 1.0, n)
        )
        py = view.to_screen_y(values)
        base = float(
            view.to_screen_y(baseline if baseline > 0 or not view.log_y else 0.0)
        )
    shown = np.isfinite(px) & np.isfinite(py) & (fade > 0)
    if not math.isnan(base) and shown.any():
        # Clamp to just beyond the plane: a term (or a baseline of 0 on a log
        # y axis) far off the top or bottom still has its stem run off the edge.
        top, bottom = view.rect.top - 0.5 - reach, view.rect.bottom - 0.5 + reach
        py = np.clip(py, top, bottom)
        base = min(max(base, top), bottom)
        outside = ((py <= top) & (base <= top)) | ((py >= bottom) & (base >= bottom))
        shown &= ~outside
    if not shown.any():
        return ()
    px, py, fade = px[shown], py[shown], fade[shown]
    strokes: list[Stroke | Markers] = []
    markers: list[Stroke | Markers] = []
    for level in np.unique(fade)[::-1]:  # the most faded in first
        group = fade == level
        x, y = px[group], py[group]
        opacity = style.opacity * float(level)
        if style.width > 0:
            strokes.append(Stroke(_stems(x, y, base), color, style.width, opacity))
        if style.radius > 0:
            markers.append(
                Markers(np.column_stack((x, y)), style.marker, style.radius, color,
                        opacity=opacity)
            )  # fmt: skip
    return (*strokes, *markers)


def _stems(
    x: NDArray[np.float64], y: NDArray[np.float64], base: float
) -> NDArray[np.float64]:
    """Return the stems from ``base`` to the terms at screen points ``(x, y)``.

    Args:
        x: Screen x of each term, increasing.
        y: Screen y of each term.
        base: Screen y of the stems' foot (the x axis, clamped to the plane).

    Returns:
        Two-point paths, shape ``(M, 2, 2)``: one per term, or one per pixel
        column where several terms share it, at the middle of their x and
        spanning all of them.
    """
    top, bottom = np.minimum(y, base), np.maximum(y, base)
    column = np.floor(x + 0.5)
    starts = np.flatnonzero(np.r_[True, column[1:] != column[:-1]])
    if starts.size < x.size:
        ends = np.r_[starts[1:], x.size] - 1
        top = np.minimum.reduceat(top, starts)
        bottom = np.maximum.reduceat(bottom, starts)
        x = (x[starts] + x[ends]) / 2
    return np.stack((np.column_stack((x, top)), np.column_stack((x, bottom))), axis=1)


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

    margin = style.width / 2 + CLIP_MARGIN_PX
    pattern = dash_pattern(style.dash, style.width)
    box = _clip_box(view, margin)
    near = _clip_box(view, margin + _dash_reach(pattern))
    pieces = sample_curve_pieces(screen, t_range, near, sampling)
    lines = [points for points, _ in pieces]
    # The curve's two ends are fixed in the world (unlike its breaks, found
    # only as finely as the view needs).
    ends = [(t[0] == t_range[0], t[-1] == t_range[1]) for _, t in pieces]
    return _line_stroke(view, lines, style, color, box, pattern, ends=ends)


def _line_stroke(
    view: View,
    paths: list[NDArray[np.float64]],
    style: LineStyle,
    color: ColorLike,
    box: Box,
    pattern: DashPattern | None,
    *,
    ends: list[tuple[bool, bool]] | None = None,
) -> Stroke | None:
    """Return the stroke of curves sampled in screen pixels, solid or dashed.

    Args:
        view: The view being drawn.
        paths: The curves' polylines in screen pixels, not clipped.
        style: Line width and opacity.
        color: The resolved line colour.
        box: The clip box.
        pattern: The dash pattern, or ``None`` for a solid line.
        ends: Which ends of each polyline are fixed in the world (see
            :func:`~pygridplot.core.plot.dashes.dash_paths`), or ``None``
            for none.

    Returns:
        The stroke, or ``None`` if nothing is visible.
    """
    if pattern is not None:
        return _dashed_stroke(view, paths, pattern, box, color, style, ends=ends)
    clipped = clip_paths(paths, box)
    if not clipped:
        return None
    return Stroke(tuple(clipped), color, style.width, style.opacity)


def _dashed_stroke(
    view: View,
    paths: list[NDArray[np.float64]],
    pattern: DashPattern,
    box: Box,
    color: ColorLike,
    style: LineStyle | PolygonStyle,
    *,
    ends: list[tuple[bool, bool]] | None = None,
    corners: bool = False,
) -> Stroke | None:
    """Return the dashes of polylines, anchored to the view's dash lattice.

    Args:
        view: The view being drawn.
        paths: Finite polylines in screen pixels, not clipped (a closed one
            ends where it starts).
        pattern: The dash pattern.
        box: The clip box.
        color: The resolved line colour.
        style: Line width and opacity.
        ends: Which ends of each polyline are fixed in the world, or
            ``None`` for none.
        corners: Whether the polylines' corners are anchors: for the
            vertices of polylines and polygons, which are fixed in the world.

    Returns:
        The stroke (one array of short segments), or ``None`` if no dash is
        visible.
    """
    lattice = dash_lattice(view, pattern)
    segments = dash_paths(paths, pattern, lattice, box, ends=ends, corners=corners)
    if segments is None:
        return None
    return Stroke(segments, color, style.width, style.opacity)


def _dash_reach(pattern: DashPattern | None) -> float:
    """Return how much further than usual to sample a curve drawn with a pattern."""
    return 0.0 if pattern is None else dash_margin(pattern)


def _runs(points: NDArray[np.float64], *, closed: bool) -> list[NDArray[np.float64]]:
    """Split vertices into the runs between undefined ones (NaN breaks a line).

    Args:
        points: Vertices in screen pixels, shape ``(N, 2)``.
        closed: Whether the last vertex connects back to the first. A closed
            ring with no breaks ends where it starts; one with breaks joins
            its last run to its first.

    Returns:
        The runs, each with at least one vertex (one vertex is a dot).
    """
    finite = np.isfinite(points).all(axis=1)
    if closed and points.shape[0] > 1 and finite.all():
        return [np.vstack((points, points[:1]))]
    breaks = np.flatnonzero(~finite)
    runs = [run for run in np.split(points, breaks) if run.shape[0]]
    runs = [run[np.isfinite(run).all(axis=1)] for run in runs]
    runs = [run for run in runs if run.shape[0]]
    if closed and len(runs) > 1 and finite[0] and finite[-1]:
        runs = [np.vstack((runs[-1], runs[0])), *runs[1:-1]]
    return runs


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
    sampling: FunctionSampling,
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


def text_items(
    view: View,
    text: str,
    position: tuple[float, float],
    style: TextStyle,
    color: ColorLike,
    *,
    font: FontSpec,
    halo: tuple[ColorLike, int] | None,
    measurer: TextMeasurer,
    anchor: Anchor = "center",
    gap: float = 0.0,
) -> list[TextItem]:
    r"""Return the lines of a text anchored to a world point.

    See :mod:`pygridplot.core.plot.text` for how anchors, gaps and lines
    work.

    Args:
        view: The view being drawn.
        text: The text (``"\n"`` starts a new line).
        position: The world point the text is anchored to.
        style: The text's opacity (its colour and size are resolved into
            ``color`` and ``font``).
        color: The resolved text colour.
        font: The font to draw with.
        halo: The halo's colour and width, or ``None`` for no halo.
        measurer: Measures the lines.
        anchor: Which point of the text's box is placed at the position.
        gap: Pixels between the position and the box, on the anchored sides.

    Returns:
        A text item per line that reaches the plane (none if the position
        is undefined, for example negative on a log axis).
    """
    rect = view.rect
    if rect.is_empty or not text:
        return []
    px, py = view.to_screen(float(position[0]), float(position[1]))
    if not (math.isfinite(px) and math.isfinite(py)):
        return []
    # Text far off the plane is left out before it is placed, so pixel
    # positions always stay small enough for the renderer.
    reach = gap + 2 * max(rect.width, rect.height)
    if not (
        rect.left - reach <= px <= rect.right + reach
        and rect.top - reach <= py <= rect.bottom + reach
    ):
        return []
    lines = text_lines(text)
    sizes = [measurer.measure_text(line, font) for line in lines]
    places = place_lines((px, py), sizes, anchor, gap)
    halo_color, halo_width = halo if halo is not None else (None, 0)
    items = []
    for line, size, (left, top) in zip(lines, sizes, places, strict=True):
        if (
            left >= rect.right
            or left + size.width + halo_width <= rect.left - halo_width
            or top - size.overflow_top - halo_width >= rect.bottom
            or top + size.height + size.descent + halo_width <= rect.top
        ):
            continue  # entirely outside the plane
        items.append(
            TextItem(
                text=line,
                left=left,
                top=top,
                font=font,
                color=color,
                opacity=style.opacity,
                halo_color=halo_color,
                halo_width=halo_width,
            )
        )
    return items


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
