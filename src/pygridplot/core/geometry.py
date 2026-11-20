"""Geometry helpers: normalising point input and clipping to the screen.

Clipping happens in **screen space**, just before shapes are handed to a
renderer. It keeps pixel coordinates small and well-conditioned: when you
zoom far into a polygon, its vertices can be millions of pixels away, and
rasterisers (and pygame) misbehave with such coordinates. It also means
nothing is ever processed for parts of a shape that are off-screen.

* :func:`as_points` accepts the point formats the drawing calls take.
* :func:`clip_path` clips a polyline to a rectangle (Liang–Barsky, per
  segment, vectorised), splitting it where it leaves and re-enters.
  :func:`clip_paths` does the same for many polylines in one pass.
* :func:`clip_polygon` clips a polygon to a rectangle (Sutherland–Hodgman,
  vectorised per edge of the rectangle), keeping it a single closed ring.
"""

from __future__ import annotations

from typing import Final, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

Box: TypeAlias = tuple[float, float, float, float]
"""An axis-aligned clip box ``(x_min, y_min, x_max, y_max)`` in screen pixels."""

COORDINATE_LIMIT: Final = 1e9
"""Pixel coordinates are clamped to ``±COORDINATE_LIMIT`` before clipping.

Far-off points (zoomed deep into a shape, or a curve shooting off towards
an asymptote) then stay numerically harmless. The resulting change of
direction of an edge is far below a pixel for anything on screen.
"""


def as_points(data: ArrayLike) -> NDArray[np.float64]:
    """Normalise point input to an ``(N, 2)`` float array.

    Args:
        data: A single point ``(x, y)``, a sequence of points
            ``[(x0, y0), (x1, y1), ...]``, or an array of shape ``(N, 2)`` or
            ``(2,)``.

    Returns:
        The points as an array of shape ``(N, 2)`` (``N`` may be 0).

    Raises:
        ValueError: If the input cannot be read as 2-D points.
    """
    try:
        array = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "points must be numbers: (x, y), [(x, y), ...] or an (N, 2) array"
        raise ValueError(msg) from error
    if array.size == 0:
        return np.zeros((0, 2))
    if array.shape == (2,):
        return array.reshape(1, 2)
    if array.ndim == 2 and array.shape[1] == 2:
        return array
    msg = (
        f"expected (x, y), [(x, y), ...] or an (N, 2) array, got shape {array.shape}; "
        "for separate xs and ys use np.column_stack((xs, ys))"
    )
    raise ValueError(msg)


def clamp_coordinates(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clamp pixel coordinates to ``±COORDINATE_LIMIT`` (NaN stays NaN).

    Args:
        points: Points in screen pixels.

    Returns:
        The clamped points.
    """
    return np.clip(points, -COORDINATE_LIMIT, COORDINATE_LIMIT)


def clip_segments(
    a: NDArray[np.float64], b: NDArray[np.float64], box: Box
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Clip line segments ``a[i] -> b[i]`` to a box (Liang–Barsky, vectorised).

    Args:
        a: Segment starts, shape ``(N, 2)``.
        b: Segment ends, shape ``(N, 2)``.
        box: The clip box.

    Returns:
        ``(keep, t0, t1)``: which segments touch the box, and the parameter
        range ``[t0, t1]`` of the part of each segment inside it (the clipped
        segment runs from ``a + t0 (b - a)`` to ``a + t1 (b - a)``). Segments
        with a non-finite end are never kept.
    """
    x_min, y_min, x_max, y_max = box
    d = b - a
    t0 = np.zeros(a.shape[0])
    t1 = np.ones(a.shape[0])
    reject = ~(np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1))
    edges = (
        (-d[:, 0], a[:, 0] - x_min),
        (d[:, 0], x_max - a[:, 0]),
        (-d[:, 1], a[:, 1] - y_min),
        (d[:, 1], y_max - a[:, 1]),
    )
    with np.errstate(all="ignore"):
        for p, q in edges:
            ratio = q / np.where(p != 0, p, 1.0)
            t0 = np.where(p < 0, np.maximum(t0, ratio), t0)
            t1 = np.where(p > 0, np.minimum(t1, ratio), t1)
            reject |= (p == 0) & (q < 0)
    keep = ~reject & (t0 <= t1)
    return keep, t0, t1


def clip_paths(paths: list[NDArray[np.float64]], box: Box) -> list[NDArray[np.float64]]:
    """Clip several open polylines at once (see :func:`clip_path`).

    The polylines are joined, separated by NaN rows (which break lines), and
    clipped in one vectorised pass.

    Args:
        paths: Polylines in screen pixels, each of shape ``(N, 2)``.
        box: The clip box.

    Returns:
        The visible pieces of all the polylines.
    """
    if not paths:
        return []
    if len(paths) == 1:
        return clip_path(paths[0], box)
    gap = np.full((1, 2), np.nan)
    joined = np.concatenate([part for path in paths for part in (path, gap)][:-1])
    return clip_path(joined, box)


def clip_path(
    points: NDArray[np.float64], box: Box, *, closed: bool = False
) -> list[NDArray[np.float64]]:
    """Clip a polyline to a box, splitting it where it leaves the box.

    Points with a NaN coordinate break the polyline, like a pen lifted off
    the paper (the same convention as matplotlib).

    Args:
        points: Vertices in screen pixels, shape ``(N, 2)``.
        box: The clip box.
        closed: Whether the last vertex connects back to the first.

    Returns:
        The visible pieces, each an array of shape ``(M, 2)``. A single
        isolated vertex inside the box is returned as a one-point piece (it
        is drawn as a dot).
    """
    pts = clamp_coordinates(points)
    if pts.shape[0] == 0:
        return []
    if pts.shape[0] == 1:
        return [pts] if _inside(pts[0], box) else []
    if closed:
        pts = np.vstack((pts, pts[:1]))
    a, b = pts[:-1], pts[1:]
    keep, t0, t1 = clip_segments(a, b, box)
    if not keep.any():
        return []
    d = b - a
    start = a + t0[:, None] * d
    end = a + t1[:, None] * d
    # Segment i continues the path of segment i-1 when both are kept and the
    # shared vertex was not clipped away.
    joined = np.zeros(a.shape[0], dtype=bool)
    joined[1:] = keep[:-1] & keep[1:] & (t1[:-1] == 1.0) & (t0[1:] == 0.0)
    pieces = []
    kept = np.flatnonzero(keep)
    group_starts = kept[~joined[kept]]
    bounds = np.searchsorted(kept, group_starts)
    for first, last in zip(bounds, np.append(bounds[1:], kept.shape[0]), strict=True):
        run = kept[first:last]
        pieces.append(np.vstack((start[run[:1]], end[run])))
    return pieces


def clip_polygon(points: NDArray[np.float64], box: Box) -> NDArray[np.float64]:
    """Clip a polygon to a box (Sutherland–Hodgman).

    Concave and self-intersecting polygons are handled. Parts that are cut
    off may leave edges running along the box boundary, which is harmless for
    filling.

    Args:
        points: Vertices of the closed ring in screen pixels, shape ``(N, 2)``.
            Rows with NaN are ignored.
        box: The clip box.

    Returns:
        The clipped ring, shape ``(M, 2)`` (``M`` is 0 if nothing is inside).
    """
    pts = clamp_coordinates(points[np.isfinite(points).all(axis=1)])
    x_min, y_min, x_max, y_max = box
    for axis, value, keep_above in (
        (0, x_min, True),
        (0, x_max, False),
        (1, y_min, True),
        (1, y_max, False),
    ):
        if pts.shape[0] == 0:
            break
        pts = _clip_against(pts, axis, value, keep_above=keep_above)
    return pts


def _clip_against(
    pts: NDArray[np.float64], axis: int, value: float, *, keep_above: bool
) -> NDArray[np.float64]:
    """Clip a ring against one half-plane (one Sutherland–Hodgman pass).

    Args:
        pts: Ring vertices, shape ``(N, 2)``.
        axis: 0 for a vertical boundary ``x = value``, 1 for ``y = value``.
        value: Position of the boundary.
        keep_above: Keep the side with coordinates ``>= value`` (else ``<=``).

    Returns:
        The clipped ring.
    """
    cur = pts
    nxt = np.roll(pts, -1, axis=0)
    coord_cur, coord_nxt = cur[:, axis], nxt[:, axis]
    if keep_above:
        cur_in, nxt_in = coord_cur >= value, coord_nxt >= value
    else:
        cur_in, nxt_in = coord_cur <= value, coord_nxt <= value
    # Crossings are computed for every edge but only used where an edge really
    # crosses the boundary (0 <= t <= 1); the others may overflow harmlessly.
    with np.errstate(all="ignore"):
        delta = coord_nxt - coord_cur
        t = (value - coord_cur) / np.where(delta != 0, delta, 1.0)
        crossing = cur + t[:, None] * (nxt - cur)
    crossing[:, axis] = value  # exact, so rounding cannot leave it outside
    # Per edge emit: in->in: [nxt]; in->out: [crossing]; out->in: [crossing, nxt].
    first = np.where((cur_in & nxt_in)[:, None], nxt, crossing)
    has_first = cur_in | nxt_in
    has_second = ~cur_in & nxt_in
    out = np.stack((first, nxt), axis=1)  # (N, 2 slots, 2 coords)
    mask = np.stack((has_first, has_second), axis=1)
    return out[mask]


def _inside(point: NDArray[np.float64], box: Box) -> bool:
    """Return whether a point lies inside a box (edges included).

    Args:
        point: The point ``(x, y)``.
        box: The box.

    Returns:
        ``True`` if the point is finite and inside.
    """
    x, y = float(point[0]), float(point[1])
    x_min, y_min, x_max, y_max = box
    return x_min <= x <= x_max and y_min <= y <= y_max


def polygon_area(points: NDArray[np.float64]) -> float:
    """Return the signed area of a ring (shoelace formula).

    Args:
        points: Ring vertices, shape ``(N, 2)``.

    Returns:
        The area. Positive when the vertices run anticlockwise in a y-up frame
        (clockwise on screen, where y points down).
    """
    if points.shape[0] < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
