"""Dashed and dotted lines whose dashes stay put while the view pans.

A dash pattern is laid out along a stroke's centre line, in screen pixels.
The hard part is deciding where it starts. Measured from the start of what
is on screen, the dashes would crawl along a curve whenever the view pans,
as the visible start moves. Measured from the start of the whole curve,
they would need the whole curve, which the graph of a function does not
have, and they would race along it while zooming, as a small zoom moves a
point far from the start by many periods.

So the pattern is anchored to the world, locally:

1. **Anchor lines** (:func:`dash_lattice`): vertical and horizontal lines at
   the multiples of a power of two in transformed coordinates, between
   :data:`ANCHOR_PERIODS` and twice as many pattern periods apart on screen.
   They are fixed in the world, so panning moves them with the curves, and
   while zooming they stay until their spacing leaves its octave (then
   lines are added between them, or every other one is dropped).
2. **Anchors** are where a curve crosses them: a vertical line where the
   curve runs closer to horizontal (``|dx| >= |dy|`` along the segment that
   crosses it), and a horizontal line where it is steeper. So a curve
   always meets its lines at 45° or more: a straight line has anchors
   evenly spaced along it, and a curve that turns has no anchors bunched
   where it touches a line.
3. **Fitting.** Each stretch of curve between two consecutive anchors gets
   the whole number of pattern periods nearest to its length, stretched or
   squeezed to fit (by at most an eighth, as stretches are four periods or
   more). Every anchor is the middle of the pattern's first dash. A stretch
   shorter than half a period has no gap at all. Before the first anchor
   of an open curve and after its last, the pattern runs on at its own
   length from the anchor; a curve with no anchor starts it at its first
   point, and a closed curve with none is fitted around itself.

Each stretch depends only on the curve between its two anchors, so panning
by whole pixels moves every dash exactly with the curve: the invariant the
rest of the library keeps for plotted pixels. It needs the curve near the
plane to be the same whatever the view: samplers that only look at what is
on screen look :func:`dash_margin` pixels further when drawing dashes, so
that every stretch reaching the plane is sampled from anchor to anchor.

Dashes have round ends, like every stroke: a dash of ``d`` visible pixels
on a line ``w`` wide is a centre-line piece ``d - w`` long, and a dash
shorter than the width is a round dot as wide as the line. The result is
one array of short segments (see :func:`dash_paths`), which the renderer
draws in one pass.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pygridplot.core.plot.geometry import Box, clamp_coordinates, clip_segments
from pygridplot.core.view import View

Dash: TypeAlias = Literal["solid", "dashed", "dotted"] | tuple[float, ...]
"""How a line is drawn: whole, in dashes, or in dots.

``"solid"``, ``"dashed"`` and ``"dotted"`` are drawn in proportion to the
line's width. A tuple gives the lengths of the dashes and the gaps between
them in pixels, alternately, starting with a dash: ``(12, 6)`` is 12-pixel
dashes with 6-pixel gaps, and ``(12, 5, 0, 5)`` alternates dashes and dots.
"""

DASH_NAMES: Final[tuple[str, ...]] = ("solid", "dashed", "dotted")
"""The dash patterns that have names."""

ANCHOR_PERIODS: Final = 4.0
"""Anchor lines are at least this many pattern periods apart on screen.

They are fewer than twice as many apart (see :func:`dash_lattice`), so
stretches between anchors fit four to eight periods along an axis, and at
most an eighth of a period is gained or lost in fitting each. More would
fit the pattern more closely but make samplers look further beyond the
plane (see :func:`dash_margin`).
"""

MAX_MARGIN_PX: Final = 512.0
"""The furthest samplers look beyond the plane for dashes, however long the pattern."""

_CLOSED_PX: Final = 1e-6
"""A path whose ends are this close (in pixels) is a closed loop."""

_ON_LINE: Final = 1e-9
"""A vertex this close to an anchor line (in line spacings) lies on it."""

FIT_WINDOW: Final = 0.3
"""Over what fraction of a period a stretch gains a period (see :func:`_fit`).

Outside the window, every period of a stretch is the same length, at most
``(1 - FIT_WINDOW) / 2`` periods longer or shorter in all than the pattern:
under 9% on a stretch of four periods. Inside it, one period grows from
nothing to its full length, so zooming never makes the dashes jump; the
narrower the window, the more often the stretches are regular and the
faster the new period grows.
"""

CORNER_DEGREES: Final = 20.0
"""A polyline turning by more than this at a vertex has a corner there.

Corners of polylines and polygons are anchors (see :func:`dash_paths`), so
a dash is bent round each of them, if both sides of the corner are at least
a period long: vertices that turn less are taken to be points along a curve
(a circle drawn as a 36-gon turns 10° at each), and dense data (a step plot
with steps a pixel wide) has its dashes laid out as along a curve.
"""


@dataclass(frozen=True, slots=True)
class DashPattern:
    """A dash pattern resolved for a line width, along the centre line.

    One period of the pattern starts in the middle of its first dash.

    Attributes:
        period: Length of one period in pixels: the sum of all the dashes
            and gaps as given.
        centres: Where the middle of each dash is within the period, the
            first at ``0``.
        halves: Half of each dash's length along the centre line (``0`` for
            a dot); its round ends reach half the line's width further.
    """

    period: float
    centres: tuple[float, ...]
    halves: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DashLattice:
    """The anchor lines near the plane, in screen pixels (see the module docs).

    Attributes:
        x0: Screen x of one vertical anchor line.
        dx: Spacing of the vertical anchor lines in pixels.
        y0: Screen y of one horizontal anchor line.
        dy: Spacing of the horizontal anchor lines in pixels.
    """

    x0: float
    dx: float
    y0: float
    dy: float


def check_dash(value: object) -> Dash:
    """Validate a dash setting and return it in canonical form.

    Args:
        value: A name from :data:`DASH_NAMES`, or a sequence of lengths in
            pixels: dashes and gaps alternately, starting with a dash.

    Returns:
        The name, or the lengths as a tuple of floats (hashable, so styles
        holding it can be compared and cached).

    Raises:
        ValueError: If the name is unknown, or the lengths are not pairs of
            finite numbers, with dashes of zero or more and gaps above zero.
    """
    if isinstance(value, str):
        if value not in DASH_NAMES:
            msg = f"dash must be one of {DASH_NAMES} or a sequence of lengths"
            raise ValueError(f"{msg}, got {value!r}")
        return value  # type: ignore[return-value]
    try:
        lengths = tuple(float(v) for v in value)  # type: ignore[attr-defined]
    except (TypeError, ValueError) as error:
        msg = f"dash must be one of {DASH_NAMES} or a sequence of lengths"
        raise ValueError(f"{msg}, got {value!r}") from error
    if not lengths or len(lengths) % 2:
        msg = (
            "dash lengths come in pairs: (dash, gap) or (dash, gap, dash, gap, ...), "
            f"got {value!r}"
        )
        raise ValueError(msg)
    if not all(math.isfinite(v) and v >= 0 for v in lengths):
        msg = f"dash lengths must be finite and not negative, got {value!r}"
        raise ValueError(msg)
    if not all(gap > 0 for gap in lengths[1::2]):
        msg = f"the gaps of a dash pattern must be longer than 0, got {value!r}"
        raise ValueError(msg)
    return lengths


def dash_lengths(dash: Dash, width: float) -> tuple[float, ...] | None:
    """Return the visible lengths of a pattern's dashes and gaps, in pixels.

    Args:
        dash: The dash setting (already checked).
        width: The line's width in pixels.

    Returns:
        Dashes and gaps alternately, or ``None`` for a solid line. Named
        patterns grow with the width, so a thick dashed line looks like a
        thin one enlarged.
    """
    w = max(width, 1.0)
    if dash == "solid":
        return None
    if dash == "dashed":
        return (3.0 * w + 2.0, 2.0 * w + 2.0)
    if dash == "dotted":
        return (w, 1.5 * w + 1.0)
    return tuple(dash)


def dash_pattern(dash: Dash, width: float) -> DashPattern | None:
    """Resolve a dash setting for a line width.

    Args:
        dash: The dash setting (already checked).
        width: The line's width in pixels. Lines thinner than a pixel are
            drawn a pixel wide (fainter), so they count as a pixel wide.

    Returns:
        The pattern, or ``None`` for a solid line.
    """
    lengths = dash_lengths(dash, width)
    if lengths is None:
        return None
    w = max(width, 1.0)
    on, off = lengths[0::2], lengths[1::2]
    centres = []
    position = 0.0
    for i, gap in enumerate(off):
        centres.append(position)
        position += on[i] / 2 + gap + on[(i + 1) % len(on)] / 2
    halves = tuple(max(d - w, 0.0) / 2 for d in on)
    return DashPattern(position, tuple(centres), halves)


def dash_margin(pattern: DashPattern) -> float:
    """Return how far beyond the plane, in pixels, a dashed curve must be sampled.

    A stretch between anchors that reaches the plane lies between two
    neighbouring anchor lines of each family it uses, so both its anchors
    are within one line spacing (fewer than ``2 × ANCHOR_PERIODS`` periods)
    of the plane. A stretch where a curve turns from shallow to steep can
    reach a little further; there, the dashes next to the plane's edge may
    shift slightly while panning.

    Args:
        pattern: The pattern.

    Returns:
        The margin, at most :data:`MAX_MARGIN_PX`.
    """
    return min(2.0 * ANCHOR_PERIODS * pattern.period, MAX_MARGIN_PX)


def dash_lattice(view: View, pattern: DashPattern) -> DashLattice:
    """Return the anchor lines for a pattern in this view.

    Along each axis the lines are the multiples of a power of two in
    transformed units (world units, or decades on a log axis) whose spacing
    on screen is at least ``ANCHOR_PERIODS`` periods and less than twice
    that. The line nearest the centre is placed precisely (see
    :meth:`View.multiples_to_screen`), so its pixel moves exactly with the
    view.

    Args:
        view: The view being drawn.
        pattern: The pattern.

    Returns:
        The lattice.
    """
    spacing = ANCHOR_PERIODS * pattern.period

    def axis(name: Literal["x", "y"], scale: float) -> tuple[float, float]:
        step = math.ldexp(1.0, math.ceil(math.log2(spacing / scale)))
        ratio = view.axis_center(name) / step
        index = round(ratio) if abs(ratio) < 2.0**52 else 0
        indices = np.array([index], dtype=np.int64)
        return float(view.multiples_to_screen(name, step, indices)[0]), step * scale

    x0, dx = axis("x", view.scale_x)
    y0, dy = axis("y", view.scale_y)
    return DashLattice(x0, dx, y0, dy)


def dash_paths(
    paths: Sequence[NDArray[np.float64]],
    pattern: DashPattern,
    lattice: DashLattice | None,
    box: Box,
    *,
    ends: Sequence[tuple[bool, bool]] | None = None,
    corners: bool = False,
) -> NDArray[np.float64] | None:
    """Cut polylines into the dashes of a pattern, clipped to a box.

    Args:
        paths: Polylines in screen pixels, each of shape ``(N, 2)`` and
            finite, **not clipped**: the pattern is fitted to the whole
            stretch between two anchors, on screen or not. A path whose
            last point is its first is a closed loop; a path of one point
            is a dot.
        pattern: The dash pattern.
        lattice: The anchor lines, or ``None`` for none (as in a legend's
            swatch).
        box: The clip box. Only dashes inside it are returned, so a curve
            running far off the plane costs nothing there.
        ends: For each path, whether its first and its last point are
            anchors: only for ends fixed in the world, such as a polyline's
            or a parametric curve's (a curve cut off at the edge of what was
            sampled must not have an anchor there). Then the line starts and
            ends with a dash.
        corners: Whether the vertices where a path turns by more than
            :data:`CORNER_DEGREES` are anchors (for polylines and polygons,
            whose vertices are fixed in the world; a sampled curve's are not).

    Returns:
        The dashes as segments, shape ``(M, 2, 2)`` (a dash along a bent
        curve is several segments; a dot is a segment of zero length), or
        ``None`` if none is inside the box.
    """
    fixed = list(ends) if ends is not None else [(False, False)] * len(paths)
    usable = [
        (clamp_coordinates(p), end)
        for p, end in zip(paths, fixed, strict=True)
        if p.shape[0] > 0
    ]
    dots = [p for p, _ in usable if p.shape[0] == 1 and _inside(p[0], box)]
    lines = [(p, end) for p, end in usable if p.shape[0] > 1]
    pieces = [np.stack((p, p), axis=1) for p in dots]
    if lines:
        dashed = _dash_lines(
            [p for p, _ in lines],
            pattern,
            lattice,
            box,
            ends=np.array([end for _, end in lines], dtype=bool).reshape(-1, 2),
            corners=corners,
        )
        if dashed is not None:
            pieces.append(dashed)
    if not pieces:
        return None
    return np.concatenate(pieces)


def swatch_dashes(
    start: ArrayLike, end: ArrayLike, pattern: DashPattern
) -> NDArray[np.float64]:
    """Return the dashes of a short straight line that starts and ends with a dash.

    For legend swatches: the pattern is stretched or squeezed so that whole
    dashes sit at both ends (at least two of them), whatever its length.

    Args:
        start: Where the first dash's centre line starts, ``(x, y)``.
        end: Where the last dash's centre line ends.
        pattern: The pattern.

    Returns:
        The dashes as segments, shape ``(M, 2, 2)``.
    """
    a, b = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    length = float(np.hypot(*(b - a)))
    if length == 0:
        return np.stack((a, a))[None]
    half = pattern.halves[0]
    periods = max(round((length - 2 * half) / pattern.period), 1)
    stretch = length / (periods * pattern.period + 2 * half)
    begin, finish, _ = _pieces(
        np.array([0.0]),
        np.array([length]),
        np.array([half * stretch]),
        np.array([stretch]),
        np.array([False]),
        pattern,
        np.array([True]),
    )
    direction = (b - a) / length
    return np.stack(
        (a + begin[:, None] * direction, a + finish[:, None] * direction), axis=1
    )


@dataclass(frozen=True, slots=True, eq=False)
class _Segments:
    """The segments of all the paths being dashed, joined end to end.

    Attributes:
        a: Segment starts, shape ``(S, 2)``.
        b: Segment ends.
        path: The path each segment belongs to.
        start: Arc length along its path at each segment's start.
        length: Each segment's length.
        path_length: Each path's total length.
        closed: Whether each path is a closed loop.
    """

    a: NDArray[np.float64]
    b: NDArray[np.float64]
    path: NDArray[np.intp]
    start: NDArray[np.float64]
    length: NDArray[np.float64]
    path_length: NDArray[np.float64]
    closed: NDArray[np.bool_]


@dataclass(frozen=True, slots=True, eq=False)
class _Anchors:
    """Where the paths cross their anchor lines, sorted by path, then arc length.

    Attributes:
        path: The path of each anchor.
        s: Its arc length along the path.
        first: Index of each path's first anchor (meaningless without any).
        count: Each path's number of anchors.
    """

    path: NDArray[np.intp]
    s: NDArray[np.float64]
    first: NDArray[np.intp]
    count: NDArray[np.intp]


def _dash_lines(
    lines: list[NDArray[np.float64]],
    pattern: DashPattern,
    lattice: DashLattice | None,
    box: Box,
    *,
    ends: NDArray[np.bool_],
    corners: bool,
) -> NDArray[np.float64] | None:
    """Dash polylines of two points or more (see :func:`dash_paths`).

    Args:
        lines: The polylines, clamped to harmless coordinates.
        pattern: The pattern.
        lattice: The anchor lines, or ``None``.
        box: The clip box.
        ends: For each line, whether its first and last points are anchors,
            shape ``(len(lines), 2)``.
        corners: Whether the lines' corners are anchors.

    Returns:
        The dashes as segments, or ``None`` if none is inside the box.
    """
    segments = _segments(lines)
    found = [_end_anchors(segments, ends)]
    if lattice is not None:
        found.append(_crossings(segments, lattice, box, dash_margin(pattern)))
    if corners:
        found.append(_corners(segments, pattern.period))
    anchors = _anchor_set(found, segments)
    regions = _layout(segments, anchors, pattern)
    seg, lo, hi, region = _visible_parts(segments, regions, box)
    if seg.shape[0] == 0:
        return None
    origin, stretch = regions.origin[region], regions.stretch[region]
    solid = regions.solid[region]
    path = segments.path[seg]
    final = ~segments.closed[path] & (hi >= segments.path_length[path])
    start, end, owner = _pieces(lo, hi, origin, stretch, solid, pattern, final)
    if owner.shape[0] == 0:
        return None
    on = seg[owner]
    a, direction = segments.a[on], segments.b[on] - segments.a[on]
    base, length = segments.start[on], segments.length[on]
    # A dot can fall on a segment of no length (a repeated last vertex).
    safe = np.where(length > 0, length, 1.0)
    t0 = np.where(length > 0, (start - base) / safe, 0.0)[:, None]
    t1 = np.where(length > 0, (end - base) / safe, 0.0)[:, None]
    return np.stack((a + t0 * direction, a + t1 * direction), axis=1)


def _segments(lines: list[NDArray[np.float64]]) -> _Segments:
    """Join the paths' segments into flat arrays, with arc lengths.

    Args:
        lines: Polylines of two points or more.

    Returns:
        The segments.
    """
    counts = np.array([line.shape[0] for line in lines])
    vertices = np.concatenate(lines)
    owner = np.repeat(np.arange(len(lines)), counts)
    inside = np.flatnonzero(owner[:-1] == owner[1:])
    a, b = vertices[inside], vertices[inside + 1]
    path = owner[inside]
    d = b - a
    length = np.hypot(d[:, 0], d[:, 1])
    total = np.concatenate(([0.0], np.cumsum(length)))
    first = np.searchsorted(path, np.arange(len(lines)))
    start = total[:-1] - total[first[path]]
    # Where each path's last segment ends, computed as every segment's end
    # is, so that comparing the two is exact.
    last = np.r_[first[1:] - 1, path.shape[0] - 1]
    path_length = start[last] + length[last]
    ends = np.array([np.abs(line[-1] - line[0]).max() for line in lines])
    closed = (counts > 2) & (ends <= _CLOSED_PX)
    return _Segments(a, b, path, start, length, path_length, closed)


def _crossings(
    segments: _Segments, lattice: DashLattice, box: Box, margin: float
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Find where the paths cross their anchor lines near the plane.

    A segment closer to horizontal crosses the vertical lines, a steeper one
    the horizontal lines. A crossing belongs to the segment it lies in, or
    to the one that ends on it (never to the one that starts there), so it
    counts once. Only lines within ``margin`` of the box are used: they are
    all the anchors a stretch on screen can have, and this bounds the work
    for segments running far off the plane.

    Args:
        segments: The paths' segments.
        lattice: The anchor lines.
        box: The clip box.
        margin: How far beyond it to look for lines.

    Returns:
        ``(path, s)``: each anchor's path, and its arc length along it.
    """
    a, b = segments.a, segments.b
    d = b - a
    # At 45° exactly (y = x), rounding after a pan must not change the lines
    # a segment crosses.
    shallow = np.abs(d[:, 0]) >= np.abs(d[:, 1]) * (1.0 - _ON_LINE)
    axis = np.where(shallow, 0, 1)
    rows = np.arange(a.shape[0])
    origin = np.where(shallow, lattice.x0, lattice.y0)
    spacing = np.where(shallow, lattice.dx, lattice.dy)
    qa = _snap((a[rows, axis] - origin) / spacing)
    qb = _snap((b[rows, axis] - origin) / spacing)
    x_min, y_min, x_max, y_max = box
    low = np.where(
        shallow,
        math.ceil((x_min - margin - lattice.x0) / lattice.dx),
        math.ceil((y_min - margin - lattice.y0) / lattice.dy),
    )
    high = np.where(
        shallow,
        math.floor((x_max + margin - lattice.x0) / lattice.dx),
        math.floor((y_max + margin - lattice.y0) / lattice.dy),
    )
    rising = qb > qa
    first = np.maximum(np.where(rising, np.floor(qa) + 1, np.ceil(qb)), low)
    last = np.minimum(np.where(rising, np.floor(qb), np.ceil(qa) - 1), high)
    count = np.maximum(last - first + 1, 0).astype(np.intp)
    total = int(count.sum())
    owner = np.repeat(rows, count)
    offsets = np.concatenate(([0], np.cumsum(count)[:-1]))
    line = first[owner] + (np.arange(total) - offsets[owner])
    t = (line - qa[owner]) / (qb[owner] - qa[owner])
    s = segments.start[owner] + t * segments.length[owner]
    return segments.path[owner], s


def _end_anchors(
    segments: _Segments, ends: NDArray[np.bool_]
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Return anchors at the ends of the open paths whose ends are fixed.

    Args:
        segments: The paths' segments.
        ends: For each path, whether its first and last points are anchors.

    Returns:
        ``(path, s)`` of the anchors.
    """
    open_ = ~segments.closed
    first = np.flatnonzero(ends[:, 0] & open_)
    last = np.flatnonzero(ends[:, 1] & open_)
    path = np.concatenate((first, last)).astype(np.intp)
    s = np.concatenate((np.zeros(first.shape[0]), segments.path_length[last]))
    return path, s


def _corners(
    segments: _Segments, period: float
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Return anchors at the vertices where paths turn sharply.

    The turn between consecutive segments of a path is compared with
    :data:`CORNER_DEGREES`; segments of zero length are skipped over (a
    repeated vertex is no corner). A closed path's first point is a corner
    if the path turns there. Both segments must be at least ``period`` long.

    Args:
        segments: The paths' segments.
        period: The pattern's period in pixels.

    Returns:
        ``(path, s)`` of the anchors.
    """
    long = np.flatnonzero(segments.length > 0)
    if long.shape[0] == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0)
    d = (segments.b - segments.a)[long]
    unit = d / segments.length[long][:, None]
    path = segments.path[long]
    # Each long segment with the one before it on its path (wrapping round
    # on closed paths).
    previous = np.roll(np.arange(long.shape[0]), 1)
    starts = np.r_[True, path[1:] != path[:-1]]
    ends = np.r_[path[1:] != path[:-1], True]
    last_of_path = np.flatnonzero(ends)
    first_of_path = np.flatnonzero(starts)
    previous[first_of_path] = last_of_path
    wraps = starts & segments.closed[path]
    pairs = ~starts | wraps
    turn = np.einsum("ij,ij->i", unit, unit[previous])
    sides = segments.length[long]
    long_enough = (sides >= period) & (sides[previous] >= period)
    corner = pairs & long_enough & (turn < math.cos(math.radians(CORNER_DEGREES)))
    s = np.where(starts, 0.0, segments.start[long])
    return path[corner].astype(np.intp), s[corner]


def _anchor_set(
    found: list[tuple[NDArray[np.intp], NDArray[np.float64]]], segments: _Segments
) -> _Anchors:
    """Combine anchors found in different ways, sorted along each path.

    Args:
        found: ``(path, s)`` arrays of anchors.
        segments: The paths' segments.

    Returns:
        The anchors.
    """
    path = np.concatenate([p for p, _ in found]).astype(np.intp)
    s = np.concatenate([s for _, s in found])
    order = np.lexsort((s, path))
    path, s = path[order], s[order]
    paths = segments.path_length.shape[0]
    first = np.searchsorted(path, np.arange(paths)).astype(np.intp)
    count = np.bincount(path, minlength=paths).astype(np.intp)
    return _Anchors(path, s, first, count)


def _snap(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Put positions measured in line spacings that are all but on a line onto it.

    Vertices often lie exactly on an anchor line (an implicit curve's
    vertices lie on its sampling grid, whose lines include the anchor
    lines; a polygon's corners are often whole numbers). Computed after a
    pan, such a position can come out a rounding error to either side of
    the line, which would hand the crossing to the other segment (or to
    neither, if they use different lines), and the pattern would jump.
    Positions near the plane are within a few dozen spacings of the line
    they are measured from, so rounding errors there are far below the
    tolerance.

    Args:
        q: Positions, in line spacings from a line.

    Returns:
        The positions, those within ``1e-9`` of a whole number set to it.
    """
    nearest = np.rint(q)
    return np.where(np.abs(q - nearest) < _ON_LINE, nearest, q)


@dataclass(frozen=True, slots=True, eq=False)
class _Regions:
    """Stretches of the paths along which the pattern is laid out alike.

    Sorted by path, then by start; a region ends where the next one of its
    path starts (none is empty). Every path has a region starting at ``0``.

    Attributes:
        path: The path of each region.
        start: Arc length along the path where it starts.
        origin: Arc length where a period of the pattern starts in it (the
            middle of a first dash).
        stretch: How much the pattern's lengths are multiplied by in it.
        solid: Whether it is drawn whole (a stretch too short for a gap).
    """

    path: NDArray[np.intp]
    start: NDArray[np.float64]
    origin: NDArray[np.float64]
    stretch: NDArray[np.float64]
    solid: NDArray[np.bool_]


def _layout(segments: _Segments, anchors: _Anchors, pattern: DashPattern) -> _Regions:
    """Lay the pattern out along every path, from its anchors.

    Between two anchors (and round a closed path, from its last anchor to
    its first, or from its start right round without one) the pattern is
    fitted (see :func:`_fit`). Before the first anchor of an open path, and
    after its last, it runs at its own length from the anchor; along an
    open path without anchors, from the path's start.

    Args:
        segments: The paths' segments.
        anchors: Their anchors.
        pattern: The pattern.

    Returns:
        The regions.
    """
    count = segments.path_length.shape[0]
    paths = np.arange(count, dtype=np.intp)
    total, closed = segments.path_length, segments.closed
    has = anchors.count > 0
    n = anchors.s.shape[0]
    first_s = anchors.s[np.minimum(anchors.first, max(n - 1, 0))] if n else total * 0
    last_s = (
        anchors.s[np.minimum(anchors.first + anchors.count - 1, max(n - 1, 0))]
        if n
        else total * 0
    )
    # Open paths: at their own length before the first anchor and after the last.
    open_ = ~closed
    before, after = paths[open_], paths[open_ & has]
    natural = (
        np.concatenate((before, after)),
        np.concatenate((np.zeros(before.shape[0]), last_s[after])),
        np.concatenate(
            (np.where(has[before], first_s[before], total[before]), total[after])
        ),
        np.concatenate((np.where(has[before], first_s[before], 0.0), last_s[after])),
        np.concatenate((np.zeros(before.shape[0]), np.full(after.shape[0], 4))),
    )
    # Fitted: between consecutive anchors, round closed paths.
    same = anchors.path[1:] == anchors.path[:-1]
    loops = paths[closed & has]
    bare = paths[closed & ~has]
    a = np.concatenate((anchors.s[:-1][same], last_s[loops], np.zeros(bare.shape[0])))
    b = np.concatenate(
        (anchors.s[1:][same], first_s[loops] + total[loops], total[bare])
    )
    owner = np.concatenate((anchors.path[:-1][same], loops, bare)).astype(np.intp)
    wraps = np.concatenate(
        (
            np.zeros(int(same.sum()), dtype=bool),
            np.ones(loops.shape[0], dtype=bool),
            np.zeros(bare.shape[0], dtype=bool),
        )
    )
    r_start, r_end, r_origin, r_stretch, r_solid, rank = _fit(a, b, pattern)
    index = np.repeat(np.arange(a.shape[0]), 3)
    r_path, r_wrap, r_total = owner[index], wraps[index], total[owner[index]]
    # A region of a wrapped stretch past the path's end continues from its start.
    keep = ~r_wrap | (r_start < r_total)
    carry = r_wrap & (r_end > r_total)
    path = np.concatenate((natural[0], r_path[keep], r_path[carry])).astype(np.intp)
    start = np.concatenate(
        (natural[1], r_start[keep], np.maximum(r_start[carry] - r_total[carry], 0.0))
    )
    end = np.concatenate(
        (
            natural[2],
            np.minimum(r_end[keep], r_total[keep]),
            r_end[carry] - r_total[carry],
        )
    )
    origin = np.concatenate(
        (natural[3], r_origin[keep], r_origin[carry] - r_total[carry])
    )
    stretch = np.concatenate(
        (np.ones(natural[0].shape[0]), r_stretch[keep], r_stretch[carry])
    )
    solid = np.concatenate(
        (np.zeros(natural[0].shape[0], dtype=bool), r_solid[keep], r_solid[carry])
    )
    ranks = np.concatenate((natural[4], rank[keep], rank[carry]))
    # Empty regions (a growing period not yet growing, a stretch of no length)
    # are dropped, so a rounding error cannot put one after the next region's
    # start; a path left without any (of no length) runs from its start.
    full = end > start
    bare_paths = np.setdiff1d(paths, path[full])
    path = np.concatenate((path[full], bare_paths)).astype(np.intp)
    start = np.concatenate((start[full], np.zeros(bare_paths.shape[0])))
    origin = np.concatenate((origin[full], np.zeros(bare_paths.shape[0])))
    stretch = np.concatenate((stretch[full], np.ones(bare_paths.shape[0])))
    solid = np.concatenate((solid[full], np.zeros(bare_paths.shape[0], dtype=bool)))
    ranks = np.concatenate((ranks[full], np.zeros(bare_paths.shape[0])))
    order = np.lexsort((ranks, start, path))
    return _Regions(
        path[order], start[order], origin[order], stretch[order], solid[order]
    )


def _fit(
    a: NDArray[np.float64], b: NDArray[np.float64], pattern: DashPattern
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.float64],
]:
    """Fit the pattern to stretches of path between two anchors.

    A stretch ``n + f`` periods long (``f`` below 1) holds ``n`` periods,
    stretched to fill it, and one more period growing in its middle: from
    nothing while ``f`` is below ``0.5 - FIT_WINDOW / 2``, to its full
    length once ``f`` is above ``0.5 + FIT_WINDOW / 2``, smoothly in
    between, while the others shrink to make room. Periods start in the
    middle of a dash, so the growing period lengthens a dash, and then opens
    a gap in it. So the dashes never jump while a stretch grows or shrinks
    (as when zooming), and outside the window all of a stretch's periods are
    alike. A stretch too short for any gap is drawn whole.

    Each stretch becomes three regions: the periods before the growing one,
    the growing one, and the periods after it.

    Args:
        a: Where each stretch starts (arc length).
        b: Where it ends.
        pattern: The pattern.

    Returns:
        ``(start, end, origin, stretch, solid, rank)`` of the three regions
        of each stretch, in order (``3 × len(a)`` each); ``rank`` is 1, 2, 3.
    """
    period = pattern.period
    length = b - a
    nu = length / period
    whole = np.floor(nu)
    fraction = nu - whole
    t = np.clip((fraction - (0.5 - FIT_WINDOW / 2)) / FIT_WINDOW, 0.0, 1.0)
    growing = t * t * (3.0 - 2.0 * t)
    periods = whole + growing
    solid = periods <= 0
    unit = np.where(solid, 1.0, length / (np.where(solid, 1.0, periods) * period))
    first = np.minimum(a + np.floor(whole / 2) * unit * period, b)
    second = np.minimum(first + growing * unit * period, b)
    starts = np.stack((a, first, second), axis=1)
    ends = np.stack((first, second, b), axis=1)
    origins = starts.copy()
    origins[:, 0] = a
    stretches = np.stack((unit, growing * unit, unit), axis=1)
    stretches[:, 1] = np.where(growing > 0, stretches[:, 1], 1.0)
    solids = np.repeat(solid[:, None], 3, axis=1)
    ends[:, 0] = np.where(solid, b, ends[:, 0])
    ranks = np.tile(np.array([1.0, 2.0, 3.0]), a.shape[0])
    return (
        starts.ravel(),
        ends.ravel(),
        origins.ravel(),
        stretches.ravel(),
        solids.ravel(),
        ranks,
    )


def _visible_parts(
    segments: _Segments, regions: _Regions, box: Box
) -> tuple[
    NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]
]:
    """Return the parts of the segments inside the box, cut where regions start.

    Args:
        segments: The paths' segments.
        regions: The pattern's regions along them.
        box: The clip box.

    Returns:
        ``(segment, lo, hi, region)``: for each part, its segment, the range
        of arc length it covers, and the region it lies in.
    """
    keep, t0, t1 = clip_segments(segments.a, segments.b, box)
    rows = np.flatnonzero(keep)
    lo = segments.start[rows] + t0[rows] * segments.length[rows]
    hi = segments.start[rows] + t1[rows] * segments.length[rows]
    key = _path_key(segments)
    cuts = regions.start + key[regions.path]
    offset = key[segments.path[rows]]
    begin = np.searchsorted(cuts, lo + offset, side="right")
    finish = np.searchsorted(cuts, hi + offset, side="left")
    inner = np.maximum(finish - begin, 0)
    total = rows.shape[0] + int(inner.sum())
    owner = np.repeat(np.arange(rows.shape[0]), inner + 1)
    offsets = np.concatenate(([0], np.cumsum(inner + 1)[:-1]))
    rank = np.arange(total) - offsets[owner]
    # Part k of a visible part runs from cut k - 1 (or its start) to cut k
    # (or its end).
    index = begin[owner] + rank
    last = cuts.shape[0] - 1
    starts = np.where(rank == 0, lo[owner], regions.start[np.clip(index - 1, 0, last)])
    ends = np.where(
        rank == inner[owner], hi[owner], regions.start[np.clip(index, 0, last)]
    )
    middle = (starts + ends) / 2 + offset[owner]
    region = np.searchsorted(cuts, middle, side="right") - 1
    return rows[owner], starts, ends, np.maximum(region, 0).astype(np.intp)


def _path_key(segments: _Segments) -> NDArray[np.float64]:
    """Return an offset per path that makes arc lengths increase across all paths.

    Used only to look things up in one sorted array: each path's offset is
    past the end of the path before it.

    Args:
        segments: The paths' segments.

    Returns:
        The offset of each path.
    """
    sizes = segments.path_length + 1.0
    return np.concatenate(([0.0], np.cumsum(sizes)[:-1]))


def _pieces(
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    origin: NDArray[np.float64],
    stretch: NDArray[np.float64],
    solid: NDArray[np.bool_],
    pattern: DashPattern,
    final: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Intersect the visible parts with the pattern's dashes.

    Args:
        lo: Where each part starts (arc length).
        hi: Where it ends.
        origin: Where a period starts on each part.
        stretch: How much the pattern is stretched on each part.
        solid: Parts drawn whole.
        pattern: The pattern.
        final: Parts that end where an open path ends.

    Returns:
        ``(start, end, part)``: the arc length range of every piece of dash,
        and the part it lies on. Dots are pieces of zero length; they are
        counted on the part where they start (half-open), so a dot where
        two parts meet is drawn once, and on a final part also at its end
        (so a dotted polyline ends with a dot).
    """
    whole = np.flatnonzero(solid & (hi > lo))
    starts, ends, owners = [lo[whole]], [hi[whole]], [whole]
    patterned = np.flatnonzero(~solid)
    p_lo = (lo[patterned] - origin[patterned]) / stretch[patterned]
    p_hi = (hi[patterned] - origin[patterned]) / stretch[patterned]
    period = pattern.period
    for centre, half in zip(pattern.centres, pattern.halves, strict=True):
        if half > 0:
            k_lo = np.ceil((p_lo - centre - half) / period)
            k_hi = np.floor((p_hi - centre + half) / period)
        else:
            # A dot where two parts meet is the later part's, whichever way
            # the positions were rounded.
            k_lo = np.ceil((p_lo - centre) / period - _ON_LINE)
            k_hi = np.where(
                final[patterned],
                np.floor((p_hi - centre) / period + _ON_LINE),
                np.ceil((p_hi - centre) / period - _ON_LINE) - 1,
            )
        count = np.maximum(k_hi - k_lo + 1, 0).astype(np.intp)
        total = int(count.sum())
        if total == 0:
            continue
        part = np.repeat(np.arange(patterned.shape[0]), count)
        offsets = np.concatenate(([0], np.cumsum(count)[:-1]))
        k = k_lo[part] + (np.arange(total) - offsets[part])
        middle = k * period + centre
        first = np.maximum(p_lo[part], middle - half)
        last = np.minimum(p_hi[part], middle + half)
        owner = patterned[part]
        start = origin[owner] + first * stretch[owner]
        end = origin[owner] + last * stretch[owner]
        if half > 0:
            keep = end > start
            start, end, owner = start[keep], end[keep], owner[keep]
        starts.append(start)
        ends.append(end)
        owners.append(owner)
    return np.concatenate(starts), np.concatenate(ends), np.concatenate(owners)


def _inside(point: NDArray[np.float64], box: Box) -> bool:
    """Return whether a point lies inside a box (edges included)."""
    x_min, y_min, x_max, y_max = box
    return bool(x_min <= point[0] <= x_max and y_min <= point[1] <= y_max)
