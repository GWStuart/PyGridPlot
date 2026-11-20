"""Software rasterisation into antialiased coverage masks (pure NumPy).

The pygame backend uses these rasterisers instead of pygame's own drawing
functions, because those cannot draw thick antialiased curves with clean
joins or antialiased filled polygons. Everything here works on continuous
screen coordinates where integer values are pixel centres, and returns
*coverage*: how much of each pixel the shape covers, from 0 to 1. Blending
coverage onto a surface is done by :mod:`pygridplot.render.composite`.

* :func:`stroke_coverage`: polylines drawn with a round pen. Each segment
  is a capsule (a rectangle with round ends) given by a distance field, and
  overlapping capsules are combined with ``max``. Joins and caps are round,
  and a curve split into many short segments looks exactly like a single
  smooth stroke, with no dark spots where segments meet.
* :func:`fill_coverage`: polygons, by scanlines. Each pixel row is sampled
  by several sub-scanlines with exact horizontal coverage on each, using
  either the nonzero or the even-odd fill rule.
* :func:`marker_coverage`: point markers (circles, squares, ...) as small
  distance-field stamps, with an optional outline ring.

None of this imports pygame, so it is tested directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.primitives import FillRule, MarkerShape

PixelBox: TypeAlias = tuple[int, int, int, int]
"""A box of whole pixels ``(x0, y0, x1, y1)``; ``x1`` and ``y1`` are exclusive."""

FILL_SUBSAMPLES: Final = 5
"""Sub-scanlines per pixel row when filling (vertical antialiasing levels)."""

_SQRT_PI: Final = math.sqrt(math.pi)

_SNAP: Final = 1e-5
"""Fill coverage this close to 0 or 1 is round-off, and is snapped."""

_BUFFER_FACTOR: Final = 32
"""Merge duplicates in a buffer while its area is at most this many times the
number of candidates (see :func:`_unique_max`).

Clearing and scanning the buffer costs about a nanosecond per pixel of its
area, and sorting about 50 to 70 nanoseconds per candidate, so the buffer
wins until the area is a few dozen times the candidate count.
"""


@dataclass(frozen=True, slots=True, eq=False)
class SparseCoverage:
    """Coverage of individual pixels (each pixel appears once).

    Attributes:
        x: Pixel columns.
        y: Pixel rows.
        alpha: Coverage of each pixel in ``(0, 1]``.
    """

    x: NDArray[np.intp]
    y: NDArray[np.intp]
    alpha: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.x.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class DenseCoverage:
    """Coverage of a rectangular block of pixels.

    Attributes:
        left: Column of the block's first pixel.
        top: Row of the block's first pixel.
        alpha: Coverage in ``[0, 1]``, shape ``(height, width)``.
    """

    left: int
    top: int
    alpha: NDArray[np.floating[Any]]


# ----------------------------------------------------------------------
# Strokes
# ----------------------------------------------------------------------
def stroke_coverage(
    paths: tuple[NDArray[np.float64], ...] | list[NDArray[np.float64]],
    width: float,
    clip: PixelBox,
) -> SparseCoverage | None:
    """Rasterise polylines drawn with a round pen of ``width`` pixels.

    Args:
        paths: Polylines, each of shape ``(N, 2)``. One-point paths are dots.
        width: Pen width in pixels. Widths below 1 are drawn 1 pixel wide
            with proportionally lower coverage, so hairlines stay visible
            without looking jagged.
        clip: Only pixels inside this box are produced.

    Returns:
        The covered pixels, or ``None`` if none are covered.
    """
    a, b = _segments(paths)
    if a.shape[0] == 0:
        return None
    radius = width / 2 + 0.5 if width >= 1.0 else 1.0
    strength = 1.0 if width >= 1.0 else width
    a, b = _split_long(a, b, radius)
    candidates = _segment_distances(a, b, radius, clip)
    if candidates is None:
        return None
    qx, qy, distance, box = candidates
    coverage = _unique_max(qx, qy, radius - distance, box)
    if coverage is None:
        return None
    value = np.minimum(coverage.alpha, 1.0)
    if strength != 1.0:
        value *= strength
    return SparseCoverage(coverage.x, coverage.y, value)


def _segments(
    paths: tuple[NDArray[np.float64], ...] | list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Collect the segments of all paths (a one-point path is a zero-length segment).

    Args:
        paths: Polylines of shape ``(N, 2)``.

    Returns:
        ``(starts, ends)``, both of shape ``(M, 2)``.
    """
    starts, ends = [], []
    for path in paths:
        if path.shape[0] == 1:
            starts.append(path)
            ends.append(path)
        elif path.shape[0] > 1:
            starts.append(path[:-1])
            ends.append(path[1:])
    if not starts:
        empty = np.zeros((0, 2))
        return empty, empty
    a, b = np.concatenate(starts), np.concatenate(ends)
    finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    return a[finite], b[finite]


def _split_long(
    a: NDArray[np.float64], b: NDArray[np.float64], radius: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split long segments into pieces with tight bounding boxes.

    Every pixel in a piece's bounding box is tested, so pieces must not have
    much more box than line. A diagonal piece's box grows with the square of
    its length, so diagonal segments are cut into pieces about ``2 * radius``
    long. A piece that is nearly horizontal or vertical has a thin box at any
    length, so those are cut much less finely (up to 64 pixels). That matters
    for steep curves near asymptotes and for fast oscillations.

    Args:
        a: Segment starts.
        b: Segment ends.
        radius: Distance at which coverage drops to zero.

    Returns:
        ``(starts, ends)`` of the pieces.
    """
    d = b - a
    length = np.hypot(d[:, 0], d[:, 1])
    short_side = np.minimum(np.abs(d[:, 0]), np.abs(d[:, 1]))
    reach = 2.0 * radius + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        best = np.where(short_side > 0, reach * length / short_side, np.inf)
    piece = np.clip(best, max(2.0, 2.0 * radius), 64.0)
    pieces = np.maximum(1, np.ceil(length / piece)).astype(np.intp)
    if int(np.max(pieces)) == 1:
        return a, b
    seg = np.repeat(np.arange(a.shape[0]), pieces)
    k = np.arange(seg.shape[0]) - np.repeat(np.cumsum(pieces) - pieces, pieces)
    t0 = (k / pieces[seg])[:, None]
    t1 = ((k + 1) / pieces[seg])[:, None]
    return a[seg] + d[seg] * t0, a[seg] + d[seg] * t1


def _segment_distances(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    radius: float,
    clip: PixelBox,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float32], PixelBox] | None:
    """Measure the distance to each segment from every pixel near it.

    The candidates are the pixels of each segment's bounding box grown by
    ``radius``. They are generated in two levels, which avoids integer
    division and per-pixel lookups of segment data:

    1. one entry per (segment, pixel row), holding everything that is
       constant along the row;
    2. one entry per pixel, where the projection parameter ``t`` onto the
       segment changes by a constant step from pixel to pixel along a row.

    The arithmetic is done in ``float32``: coordinates are already clipped
    to the screen, so single precision is accurate to about 1/10000 of a
    pixel and halves the memory traffic.

    Args:
        a: Segment starts.
        b: Segment ends.
        radius: Distance at which coverage drops to zero.
        clip: Pixel box to stay within.

    Returns:
        ``(x, y, distance, box)``: candidate pixel centres, the distance
        from each to its segment, and the pixel box enclosing all of them. A
        pixel near several segments appears several times. ``None`` if there
        are no candidates.
    """
    cx0, cy0, cx1, cy1 = clip
    x0 = np.maximum(np.ceil(np.minimum(a[:, 0], b[:, 0]) - radius), cx0).astype(np.intp)
    x1 = np.minimum(np.floor(np.maximum(a[:, 0], b[:, 0]) + radius), cx1 - 1).astype(
        np.intp
    )
    y0 = np.maximum(np.ceil(np.minimum(a[:, 1], b[:, 1]) - radius), cy0).astype(np.intp)
    y1 = np.minimum(np.floor(np.maximum(a[:, 1], b[:, 1]) + radius), cy1 - 1).astype(
        np.intp
    )
    nx = np.maximum(x1 - x0 + 1, 0)
    ny = np.where(nx > 0, np.maximum(y1 - y0 + 1, 0), 0)
    used = ny > 0
    if not used.any():
        return None
    box = (
        int(np.min(x0[used])),
        int(np.min(y0[used])),
        int(np.max(x1[used])) + 1,
        int(np.max(y1[used])) + 1,
    )
    f32 = np.float32
    ax, ay = a[:, 0].astype(f32), a[:, 1].astype(f32)
    vx, vy = (b[:, 0] - a[:, 0]).astype(f32), (b[:, 1] - a[:, 1]).astype(f32)
    length2 = vx * vx + vy * vy
    inverse = np.divide(
        f32(1.0), length2, out=np.zeros_like(length2), where=length2 > 0
    )

    # Level 1: one entry per (segment, row).
    segment = np.repeat(np.arange(a.shape[0], dtype=np.intp), ny)
    row_y = y0[segment] + _ranks(ny)
    row_x = x0[segment]
    wx_row = row_x.astype(f32) - ax[segment]  # from the start to the row's first pixel
    wy_row = row_y.astype(f32) - ay[segment]
    ux, uy, inv = vx[segment], vy[segment], inverse[segment]
    t_row = (wx_row * ux + wy_row * uy) * inv
    dt = ux * inv  # change of t per pixel along the row

    # Level 2: one entry per pixel.
    width = nx[segment]
    col = _ranks(width)
    colf = col.astype(f32)
    t = np.repeat(t_row, width) + colf * np.repeat(dt, width)
    np.clip(t, f32(0.0), f32(1.0), out=t)
    wx = np.repeat(wx_row, width) + colf - t * np.repeat(ux, width)
    wy = np.repeat(wy_row, width) - t * np.repeat(uy, width)
    distance = np.sqrt(wx * wx + wy * wy)
    return np.repeat(row_x, width) + col, np.repeat(row_y, width), distance, box


def _ranks(counts: NDArray[np.intp]) -> NDArray[np.intp]:
    """Return ``0, 1, ..., n - 1`` for each count ``n``, concatenated.

    Args:
        counts: How many entries each group has.

    Returns:
        Each entry's position within its group.
    """
    total = int(counts.sum())
    starts = np.cumsum(counts) - counts
    return np.arange(total, dtype=np.intp) - np.repeat(starts, counts)


def _unique_max(
    qx: NDArray[np.intp],
    qy: NDArray[np.intp],
    alpha: NDArray[np.float32],
    box: PixelBox,
) -> SparseCoverage | None:
    """Merge duplicate pixels, keeping the largest coverage of each.

    Neighbouring segments of a stroke share many candidate pixels (dense
    tangles of curve can list each pixel dozens of times), so this step
    sees most of the work. Two strategies are used:

    * **Buffer** (usual): accumulate with ``np.maximum.at`` into a buffer
      covering the candidates' bounding box, then read out the covered
      pixels. Linear in the number of candidates, so it is fast however
      many duplicates there are.
    * **Sort**: when the bounding box is much larger than the number of
      candidates (a long, thin stroke across the screen), sorting the
      covered pixels is cheaper than clearing and scanning the buffer.

    Args:
        qx: Pixel columns (may repeat).
        qy: Pixel rows (may repeat).
        alpha: Raw coverage of each entry; zero or less means not covered.
        box: Pixel box enclosing all candidates.

    Returns:
        Each covered pixel once with its largest raw coverage, or ``None`` if
        there are none.
    """
    left, top, right, bottom = box
    width = right - left
    area = width * (bottom - top)
    flat = (qy - top) * width + (qx - left)
    if area <= _BUFFER_FACTOR * qx.shape[0]:
        best = np.zeros(area, dtype=np.float32)
        np.maximum.at(best, flat, alpha)
        unique = np.flatnonzero(best > 0.0)
        values = best[unique]
    else:
        covered = alpha > 0.0
        flat, alpha = flat[covered], alpha[covered]
        if flat.shape[0] == 0:
            return None
        order = np.argsort(flat, kind="stable")
        flat = flat[order]
        starts = np.flatnonzero(np.concatenate(([True], flat[1:] != flat[:-1])))
        values = np.maximum.reduceat(alpha[order], starts)
        unique = flat[starts]
    if unique.shape[0] == 0:
        return None
    return SparseCoverage(
        unique % width + left, unique // width + top, values.astype(np.float64)
    )


# ----------------------------------------------------------------------
# Fills
# ----------------------------------------------------------------------
def fill_coverage(
    rings: tuple[NDArray[np.float64], ...] | list[NDArray[np.float64]],
    rule: FillRule,
    clip: PixelBox,
    subsamples: int = FILL_SUBSAMPLES,
) -> DenseCoverage | None:
    """Rasterise filled polygons with antialiased edges.

    Each pixel row is sampled at ``subsamples`` evenly spaced heights. On each
    of those sub-scanlines the covered spans are computed exactly, and each
    pixel gets the fraction of its width they cover.

    Args:
        rings: Closed rings, each of shape ``(N, 2)`` (``N >= 3``).
        rule: ``"nonzero"`` or ``"evenodd"``.
        clip: Only pixels inside this box are produced.
        subsamples: Sub-scanlines per pixel row.

    Returns:
        The coverage of the polygons' bounding box (within ``clip``), or
        ``None`` if nothing is covered.
    """
    edges = _ring_edges(rings)
    if edges is None:
        return None
    ex0, ey0, ex1, ey1 = edges
    cx0, cy0, cx1, cy1 = clip
    row0 = max(cy0, math.floor(float(min(np.min(ey0), np.min(ey1))) + 0.5))
    row1 = min(cy1, math.floor(float(max(np.max(ey0), np.max(ey1))) + 0.5) + 1)
    col0 = max(cx0, math.floor(float(min(np.min(ex0), np.min(ex1))) + 0.5) - 1)
    col1 = min(cx1, math.floor(float(max(np.max(ex0), np.max(ex1))) + 0.5) + 2)
    if row1 <= row0 or col1 <= col0:
        return None
    spans = _spans(ex0, ey0, ex1, ey1, rule, row0, row1, subsamples)
    if spans is None:
        return None
    sub, start, end = spans
    alpha = _accumulate_spans(
        sub // subsamples,
        start - col0,
        end - col0,
        row1 - row0,
        col1 - col0,
        subsamples,
    )
    if not alpha.any():
        return None
    return DenseCoverage(col0, row0, alpha)


def _ring_edges(
    rings: tuple[NDArray[np.float64], ...] | list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], ...] | None:
    """Collect the non-horizontal edges of closed rings.

    Args:
        rings: Ring vertices.

    Returns:
        ``(x0, y0, x1, y1)`` arrays of edge endpoints, or ``None`` if there are
        no usable edges.
    """
    starts, ends = [], []
    for ring in rings:
        pts = ring[np.isfinite(ring).all(axis=1)]
        if pts.shape[0] >= 3:
            starts.append(pts)
            ends.append(np.roll(pts, -1, axis=0))
    if not starts:
        return None
    a, b = np.concatenate(starts), np.concatenate(ends)
    sloped = a[:, 1] != b[:, 1]
    if not sloped.any():
        return None
    a, b = a[sloped], b[sloped]
    return a[:, 0], a[:, 1], b[:, 0], b[:, 1]


def _spans(
    x0: NDArray[np.float64],
    y0: NDArray[np.float64],
    x1: NDArray[np.float64],
    y1: NDArray[np.float64],
    rule: FillRule,
    row0: int,
    row1: int,
    subsamples: int,
) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]] | None:
    """Find the covered spans on every sub-scanline.

    Args:
        x0: Edge start x.
        y0: Edge start y.
        x1: Edge end x.
        y1: Edge end y.
        rule: The fill rule.
        row0: First pixel row.
        row1: One past the last pixel row.
        subsamples: Sub-scanlines per row.

    Returns:
        ``(sub, start, end)``: sub-scanline index (from the top of ``row0``)
        and the x extent of each inside span, or ``None`` if there are none.
    """
    base = row0 - 0.5  # top edge of the first row
    count = (row1 - row0) * subsamples
    direction = np.where(y1 > y0, 1, -1)
    lo, hi = np.minimum(y0, y1), np.maximum(y0, y1)
    # Sub-scanline k sits at y = base + (k + 0.5) / subsamples. An edge
    # crosses the sub-scanlines with lo <= y < hi (half-open, so a shared
    # vertex is counted exactly once).
    first = np.maximum(np.ceil((lo - base) * subsamples - 0.5), 0).astype(np.intp)
    last = np.minimum(np.ceil((hi - base) * subsamples - 0.5) - 1, count - 1).astype(
        np.intp
    )
    crossings = np.maximum(last - first + 1, 0)
    total = int(crossings.sum())
    if total == 0:
        return None
    edge = np.repeat(np.arange(x0.shape[0]), crossings)
    k = first[edge] + (
        np.arange(total) - np.repeat(np.cumsum(crossings) - crossings, crossings)
    )
    y = base + (k + 0.5) / subsamples
    slope = (x1 - x0) / (y1 - y0)
    x = x0[edge] + (y - y0[edge]) * slope[edge]
    order = np.lexsort((x, k))
    k, x, wind = k[order], x[order], direction[edge][order]
    if rule == "evenodd":
        wind = np.ones_like(wind)
    running = np.cumsum(wind)
    new_line = np.concatenate(([True], k[1:] != k[:-1]))
    line_start = np.flatnonzero(new_line)
    before = (running - wind)[line_start]
    winding = running - np.repeat(before, np.diff(np.append(line_start, k.shape[0])))
    inside = winding % 2 == 1 if rule == "evenodd" else winding != 0
    inside[-1] = False
    spans = np.flatnonzero(inside[:-1] & (k[1:] == k[:-1]))
    if spans.shape[0] == 0:
        return None
    return k[spans], x[spans], x[spans + 1]


def _accumulate_spans(
    rows: NDArray[np.intp],
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    height: int,
    width: int,
    subsamples: int,
) -> NDArray[np.float32]:
    """Turn spans into per-pixel coverage with a cumulative-sum trick.

    For a span ``[s, e)`` the coverage of pixel ``i`` (covering ``[i, i+1)``
    once shifted to pixel edges) is ``clamp(min(e, i+1) - max(s, i), 0, 1)``.
    That is the running sum of four point weights at ``floor(s)``,
    ``floor(s) + 1``, ``floor(e)`` and ``floor(e) + 1``. Scattering those
    weights for every span and taking one cumulative sum per row gives every
    pixel's coverage at once.

    Args:
        rows: Pixel row (relative to the block) of each span.
        start: Span start, in pixel-centre coordinates relative to the block.
        end: Span end, likewise.
        height: Rows in the block.
        width: Columns in the block.
        subsamples: Sub-scanlines per row (each contributes ``1/subsamples``).

    Returns:
        Coverage of shape ``(height, width)`` (``float32``).
    """
    # float32 is ample for coverage and halves the memory traffic of the
    # block-sized arrays, which dominate the cost of large fills.
    weights = np.zeros((height, width + 2), dtype=np.float32)
    u0 = np.clip(start + 0.5, 0.0, float(width))  # to pixel-edge coordinates
    u1 = np.clip(end + 0.5, 0.0, float(width))
    share = 1.0 / subsamples
    for u, sign in ((u0, share), (u1, -share)):
        i = np.floor(u).astype(np.intp)
        f = u - i
        np.add.at(weights, (rows, i), (sign * (1.0 - f)).astype(np.float32))
        np.add.at(weights, (rows, i + 1), (sign * f).astype(np.float32))
    np.cumsum(weights, axis=1, out=weights)
    coverage = weights[:, :width]
    # Round-off in the running sums leaves tiny errors: snap them, so empty
    # pixels (such as a hole) are exactly 0 and solid ones exactly 1.
    np.clip(coverage, 0.0, 1.0, out=coverage)
    coverage[coverage < _SNAP] = 0.0
    coverage[coverage > 1.0 - _SNAP] = 1.0
    return coverage


# ----------------------------------------------------------------------
# Markers
# ----------------------------------------------------------------------
def marker_coverage(
    shape: MarkerShape,
    radius: float,
    edge_width: float,
    phase: tuple[float, float] = (0.0, 0.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Rasterise one marker into a small square stamp.

    Args:
        shape: The marker shape.
        radius: Marker size in pixels. Squares and diamonds get the same area
            as a circle of this radius, so markers look equally heavy.
        edge_width: Width of the outline ring drawn just outside the marker,
            so the marker body keeps its size (``0`` for none).
        phase: Sub-pixel offset of the marker centre from the stamp's
            centre pixel, each in ``[0, 1)``.

    Returns:
        ``(fill, edge, half)``: coverage of the marker body and of its outline
        ring (they do not overlap, so their sum never exceeds 1), both of
        shape ``(2 * half + 1, 2 * half + 1)``, and ``half``, the offset from
        the stamp's top-left pixel to its centre pixel.
    """
    half = math.ceil(radius * 1.45 + edge_width + 1.5)  # plus/cross arms reach 1.4r
    coords = np.arange(-half, half + 1, dtype=np.float64)
    dx = coords[None, :] - phase[0]
    dy = coords[:, None] - phase[1]
    sdf = _marker_sdf(shape, radius, dx, dy)
    fill = np.clip(0.5 - sdf, 0.0, 1.0)
    edge = np.zeros_like(fill)
    if edge_width > 0:
        edge = np.clip(edge_width + 0.5 - sdf, 0.0, 1.0) - fill
    return fill, edge, half


def _marker_sdf(
    shape: MarkerShape, radius: float, dx: NDArray[np.float64], dy: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the signed distance (negative inside) to a marker's outline.

    Args:
        shape: The marker shape.
        radius: Marker size in pixels.
        dx: Horizontal offsets of pixel centres from the marker centre.
        dy: Vertical offsets of pixel centres from the marker centre.

    Returns:
        Signed distances in pixels.
    """
    ax, ay = np.abs(dx), np.abs(dy)
    if shape == "circle":
        return np.hypot(dx, dy) - radius
    if shape == "square":
        half = radius * _SQRT_PI / 2  # same area as the circle
        qx, qy = ax - half, ay - half
        outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
        return outside + np.minimum(np.maximum(qx, qy), 0)
    if shape == "diamond":
        reach = radius * math.sqrt(math.pi / 2)  # same area as the circle
        return (ax + ay - reach) / math.sqrt(2)
    arm = radius * 1.15
    thickness = max(1.5, radius * 0.5)
    if shape == "plus":
        bar_a = _bar_distance(ax, ay, arm)
        bar_b = _bar_distance(ay, ax, arm)
    else:  # "cross": the plus rotated by 45 degrees
        u, v = (dx + dy) / math.sqrt(2), (dx - dy) / math.sqrt(2)
        bar_a = _bar_distance(np.abs(u), np.abs(v), arm)
        bar_b = _bar_distance(np.abs(v), np.abs(u), arm)
    return np.minimum(bar_a, bar_b) - thickness / 2


def _bar_distance(
    along: NDArray[np.float64], across: NDArray[np.float64], arm: float
) -> NDArray[np.float64]:
    """Return the distance to a centred segment from ``-arm`` to ``arm``.

    Args:
        along: Absolute offset along the segment's direction.
        across: Absolute offset perpendicular to it.
        arm: Half-length of the segment.

    Returns:
        Distances in pixels.
    """
    return np.hypot(np.maximum(along - arm, 0.0), across)
