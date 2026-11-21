"""Following a curve with the cursor: which point of a curve to drag to.

A point dragged along a curve should stay under the cursor as closely as
the curve allows, but a plain "nearest point of the curve" makes it jump
wherever two parts of the curve are about equally close: at the crossing of
a figure-eight, between the turns of a spiral, between nearby branches.
:func:`nearest_on_curve` therefore follows the curve *from where the point
is*: it slides along the curve, in whichever direction brings it closer to
the cursor, until it reaches the nearest point of that stretch (a local
minimum of the distance). Only if another part of the curve is closer than
that by more than ``switch_px`` pixels does the point jump there, so it can
still be pulled straight across to another branch.

Everything here works on a curve already sampled on screen into pieces with
their parameter values (see
:func:`pygridplot.core.plot.curves.sample_curve_pieces`), so it needs no
knowledge of the view: dragging follows the curve as it looks, on unequal
and logarithmic axes alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.plot.curves import CurvePiece


@dataclass(frozen=True, slots=True)
class _Chain:
    """Segments that follow on from each other along the curve.

    Attributes:
        start: First point of each segment, shape ``(S, 2)``.
        end: Last point of each segment, shape ``(S, 2)``.
        t_start: Parameter at each segment's start.
        t_end: Parameter at each segment's end.
        cyclic: Whether the last segment leads back into the first (a closed
            curve in one piece).
    """

    start: NDArray[np.float64]
    end: NDArray[np.float64]
    t_start: NDArray[np.float64]
    t_end: NDArray[np.float64]
    cyclic: bool


def nearest_on_curve(
    pieces: list[CurvePiece],
    target: tuple[float, float],
    current_t: float | None,
    *,
    closed: bool,
    switch_px: float,
) -> float | None:
    """Return the parameter of the point of a curve to drag to.

    Args:
        pieces: The curve sampled on screen, in order of the parameter.
        target: Where the cursor wants the point, in screen pixels.
        current_t: The point's parameter now, or ``None`` to take the
            nearest point of the whole curve (when a point is placed rather
            than dragged).
        closed: Whether the curve ends where it starts, so that sliding past
            its last point continues from its first.
        switch_px: How much closer, in pixels, another part of the curve must
            be before the point jumps to it.

    Returns:
        The parameter, or ``None`` if the curve has no points.
    """
    chains = _chains(pieces, closed)
    if not chains:
        return None
    p = np.asarray(target, dtype=np.float64)
    projected = [_project(chain, p) for chain in chains]
    # The nearest point of the whole curve.
    best_chain = min(range(len(chains)), key=lambda k: float(np.min(projected[k][0])))
    best_distance, best_t = projected[best_chain]
    best = int(np.argmin(best_distance))
    nearest = (float(best_distance[best]), float(best_t[best]))
    if current_t is None:
        return nearest[1]
    found = _containing(chains, current_t)
    if found is None:  # the point is off the sampled curve: take the nearest
        return nearest[1]
    k, i = found
    distance, t = projected[k]
    j = _descend(distance, i, chains[k].cyclic)
    local = (float(distance[j]), float(t[j]))
    return nearest[1] if nearest[0] < local[0] - switch_px else local[1]


def _chains(pieces: list[CurvePiece], closed: bool) -> list[_Chain]:
    """Turn pieces into chains of segments, joining the ends of a closed curve.

    Args:
        pieces: The sampled curve.
        closed: Whether the curve's last point leads back to its first.

    Returns:
        One chain per piece; with ``closed``, the last piece and the first are
        joined into one chain (or a single piece becomes cyclic).
    """
    parts = [(points, t) for points, t in pieces if points.shape[0] >= 2]
    if not parts:
        return []
    if closed and len(parts) == 1:
        points, t = parts[0]
        return [_Chain(points[:-1], points[1:], t[:-1], t[1:], cyclic=True)]
    if closed:  # the last piece continues into the first
        last, first = parts[-1], parts[0]
        joined = _Chain(
            np.concatenate((last[0][:-1], first[0][:-1])),
            np.concatenate((last[0][1:], first[0][1:])),
            np.concatenate((last[1][:-1], first[1][:-1])),
            np.concatenate((last[1][1:], first[1][1:])),
            cyclic=False,
        )
        parts = parts[1:-1]
        return [joined, *(_chain(points, t) for points, t in parts)]
    return [_chain(points, t) for points, t in parts]


def _chain(points: NDArray[np.float64], t: NDArray[np.float64]) -> _Chain:
    """Return the segments of one piece.

    Args:
        points: The piece's screen points.
        t: Their parameters.

    Returns:
        The chain.
    """
    return _Chain(points[:-1], points[1:], t[:-1], t[1:], cyclic=False)


def _project(
    chain: _Chain, p: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project a point onto every segment of a chain.

    Args:
        chain: The segments.
        p: The point, in screen pixels.

    Returns:
        For each segment, the distance from ``p`` to its nearest point and
        that point's parameter (interpolated along the segment).
    """
    d = chain.end - chain.start
    length2 = np.einsum("ij,ij->i", d, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.einsum("ij,ij->i", p - chain.start, d) / length2
    s = np.where(length2 > 0.0, np.clip(s, 0.0, 1.0), 0.0)
    nearest = chain.start + s[:, None] * d
    distance = np.hypot(nearest[:, 0] - p[0], nearest[:, 1] - p[1])
    return distance, chain.t_start + s * (chain.t_end - chain.t_start)


def _containing(chains: list[_Chain], t: float) -> tuple[int, int] | None:
    """Find the segment whose parameters include ``t``.

    Args:
        chains: The chains.
        t: A parameter value.

    Returns:
        ``(chain, segment)``, or ``None`` if ``t`` lies in a gap of the curve.
    """
    for k, chain in enumerate(chains):
        inside = np.flatnonzero((chain.t_start <= t) & (t <= chain.t_end))
        if inside.size:
            return k, int(inside[0])
    return None


def _descend(distance: NDArray[np.float64], i: int, cyclic: bool) -> int:
    """Slide from segment ``i`` along the chain while the distance shrinks.

    Both directions are tried; the one that ends nearer wins (ties go
    forwards, in the direction of increasing parameter).

    Args:
        distance: Distance from the target to each segment.
        i: The segment to start from.
        cyclic: Whether the chain wraps round.

    Returns:
        The segment where the distance stops decreasing.
    """
    n = int(distance.shape[0])
    if cyclic:
        ring = np.concatenate((distance, distance, distance))
        forward = ring[n + i : 2 * n + i]
        backward = ring[i + 1 : n + i + 1][::-1]
    else:
        forward = distance[i:]
        backward = distance[: i + 1][::-1]
    ahead = (i + _run(forward)) % n
    behind = (i - _run(backward)) % n
    return ahead if bool(distance[ahead] <= distance[behind]) else behind


def _run(sequence: NDArray[np.float64]) -> int:
    """Return how many steps a sequence keeps strictly decreasing from its start.

    Args:
        sequence: Distances, starting at the current segment.

    Returns:
        The number of steps to the first local minimum.
    """
    stops = np.flatnonzero(~(np.diff(sequence) < 0.0))
    return int(stops[0]) if stops.size else sequence.shape[0] - 1


def segment_span(pieces: list[CurvePiece], t: float) -> float:
    """Return the parameter width of the sampled segment containing ``t``.

    Args:
        pieces: The sampled curve.
        t: A parameter value.

    Returns:
        The width, or 0 if ``t`` is not on the sampled curve.
    """
    for _, ts in pieces:
        if ts[0] <= t <= ts[-1]:
            i = int(np.searchsorted(ts, t, side="right"))
            i = min(max(i, 1), ts.shape[0] - 1)
            return float(ts[i] - ts[i - 1])
    return 0.0
