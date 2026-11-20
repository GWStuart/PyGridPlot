"""Turning a curve ``t -> (x(t), y(t))`` into polylines that look right on screen.

This is the two-dimensional counterpart of :mod:`pygridplot.core.plot.sampling`,
used for parametric and polar curves. A parametric curve can go in any
direction, double back, loop or spiral, so instead of a vertical deviation
the sampler measures everything along the curve, in screen pixels:

1. **Initial samples.** :attr:`CurveSampling.initial_samples` evenly spaced
   values of ``t``. They depend only on the parameter range, so panning
   reuses exactly the same samples and the curve moves rigidly.
2. **Adaptive refinement** (vectorised, a few rounds). A segment is halved
   while it is (near) the screen and either its midpoint lies more than
   :attr:`CurveSampling.tolerance_px` from the straight chord, or the chord
   is longer than :attr:`CurveSampling.max_segment_px` (so that no detail can
   hide between two distant samples). Segments with an undefined end are
   halved too, so the curve reaches the edge of its domain.
3. **Domain edges.** Segments with one defined and one undefined end are
   then bisected like a root search, until the defined end stops moving on
   screen, so curves such as ``r = sqrt(cos 2θ)`` end exactly where they
   should.
4. **Discontinuities.** A segment that is still longer than
   ``max_segment_px`` after the last round cannot belong to a continuous
   curve: an honest curve would have to cross dozens of pixels in a
   ``65536``-th of an initial step. It is a jump (or a pole, where the curve
   runs off to infinity and comes back from the other side), so the curve is
   broken there instead of being joined by a long straight line. The sides
   of a pole have been refined up to it, so they run off the edge of the
   plane.

The sampler works on any function from ``t`` to *screen* coordinates (see
:func:`sample_curve`), so it needs no knowledge of the view, the axes being
linear or logarithmic, or how the curve is defined.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.plot.geometry import Box

ScreenCurve: TypeAlias = Callable[
    [NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]
]
"""Maps parameter values to screen coordinates ``(p_x, p_y)``, never raising.

Undefined points are NaN (or infinite) in either coordinate.
"""

ScalarT = TypeVar("ScalarT", bound=np.generic)

_MAX_DEPTH: Final = 60
"""Most halvings of any segment: beyond this, ``t`` runs out of float precision."""

_EDGE_STEPS: Final = 40
"""Most bisection steps spent closing in on one domain edge."""

_EDGE_SETTLED_PX: Final = 0.05
"""A domain edge is found once its defined end moves less than this."""

_EDGE_MISSES: Final = 10
"""A domain edge is also found after this many undefined midpoints in a row.

The bracket is then 1024 times narrower than when the defined end last
moved, so even an edge where the curve moves like a square root (the
fastest common case) has less than a 32nd of that last move left.
"""


@dataclass(frozen=True, slots=True)
class CurveSampling:
    """How parametric and polar curves are sampled. The defaults suit most curves.

    Attributes:
        initial_samples: Evenly spaced parameter values sampled first.
        tolerance_px: Largest allowed distance, in pixels, between the curve
            and the polyline drawn for it.
        max_segment_px: Longest allowed segment on screen. Longer segments are
            halved even when they look straight, so details between two
            samples cannot be missed. A segment still longer than this after
            the last round is treated as a jump.
        max_rounds: Most halvings of a segment beyond those the zoom level
            needs to make the initial segments ``max_segment_px`` long. They
            refine bends and domain edges, and a segment still longer than
            ``max_segment_px`` after all of them is a jump: 12 rounds mean an
            honest curve would have to cross that many pixels in 1/4096 of a
            step that normally crosses fewer.
        max_samples: Total budget of evaluations per curve per frame. When it
            runs out, the longest segments are refined first.
    """

    initial_samples: int = 256
    tolerance_px: float = 0.25
    max_segment_px: float = 24.0
    max_rounds: int = 12
    max_samples: int = 16_384

    def __post_init__(self) -> None:
        """Validate the settings."""
        if not 2 <= self.initial_samples <= self.max_samples:
            msg = "initial_samples must be at least 2 and at most max_samples"
            raise ValueError(msg)
        if not (self.tolerance_px > 0 and self.max_segment_px > 0):
            msg = "tolerance_px and max_segment_px must be positive"
            raise ValueError(msg)
        if self.max_rounds < 0:
            msg = "max_rounds must not be negative"
            raise ValueError(msg)


DEFAULT_CURVE_SAMPLING: Final = CurveSampling()
"""The curve sampling settings used when none are given."""


def sample_curve(
    curve: ScreenCurve,
    t_range: tuple[float, float],
    box: Box,
    config: CurveSampling = DEFAULT_CURVE_SAMPLING,
) -> list[NDArray[np.float64]]:
    """Sample a curve over ``t_range`` into polylines in screen coordinates.

    Args:
        curve: Maps parameter values to screen coordinates (see
            :data:`ScreenCurve`).
        t_range: ``(t_min, t_max)`` to sample, inclusive.
        box: The visible screen area (with any margin). Parts of the curve
            far outside it are not refined.
        config: Sampling settings.

    Returns:
        The curve as screen-space polylines, each of shape ``(N, 2)`` with
        ``N >= 2``, broken wherever the curve is undefined or jumps.
    """
    t0, t1 = float(t_range[0]), float(t_range[1])
    if not (math.isfinite(t0) and math.isfinite(t1) and t1 > t0):
        return []
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = _Samples.initial(curve, t0, t1, config.initial_samples)
        depth = _depth_limit(samples, box, config)
        budget = config.max_samples - config.initial_samples
        budget = _refine(samples, curve, box, config, depth, budget)
        _find_edges(samples, curve, box, budget)
    return samples.polylines(config.max_segment_px, depth)


def _depth_limit(samples: _Samples, box: Box, config: CurveSampling) -> int:
    """Return how many times a segment may be halved at this zoom.

    The initial samples are evenly spaced in ``t``, not on screen, so the
    deeper the zoom the more halvings a smooth curve needs before its
    segments are ``max_segment_px`` long. That number comes from the typical
    (median) length of the initial segments near the screen; the limit is
    that many halvings plus :attr:`CurveSampling.max_rounds`, which are left
    for bends, domain edges and telling jumps apart from fast curves.

    Args:
        samples: The initial samples.
        box: The visible screen area.
        config: Sampling settings.

    Returns:
        The depth limit, at most :data:`_MAX_DEPTH`.
    """
    a, b = samples.points[:-1], samples.points[1:]
    length = _lengths(b - a)
    finite = np.isfinite(length)
    near = finite & _near_box((a, b), np.where(finite, length, 0.0), box)
    needed = 0
    if near.any():
        typical = float(np.median(length[near]))
        if typical > config.max_segment_px:
            needed = math.ceil(math.log2(typical / config.max_segment_px))
    return min(needed + config.max_rounds, _MAX_DEPTH)


class _Samples:
    """The samples of a curve, kept sorted by parameter.

    Attributes:
        t: Parameter values, increasing.
        points: Screen points, shape ``(N, 2)``.
        open: For each segment ``(i, i + 1)``: whether it may still need
            refining.
        depth: For each segment: how many times it was halved.
    """

    def __init__(
        self,
        t: NDArray[np.float64],
        points: NDArray[np.float64],
        open_: NDArray[np.bool_],
        depth: NDArray[np.int64],
    ) -> None:
        self.t = t
        self.points = points
        self.open = open_
        self.depth = depth

    @classmethod
    def initial(cls, curve: ScreenCurve, t0: float, t1: float, count: int) -> _Samples:
        """Sample ``count`` evenly spaced parameter values.

        Args:
            curve: The curve in screen coordinates.
            t0: First parameter value.
            t1: Last parameter value.
            count: Number of samples.

        Returns:
            The samples, with every segment open.
        """
        t = np.linspace(t0, t1, count)
        segments = count - 1
        return cls(
            t,
            _evaluate(curve, t),
            np.ones(segments, dtype=bool),
            np.zeros(segments, dtype=np.int64),
        )

    def split(
        self,
        segments: NDArray[np.int64],
        t_mid: NDArray[np.float64],
        mid: NDArray[np.float64],
        keep_open: NDArray[np.bool_],
        *,
        fresh: bool = False,
    ) -> None:
        """Insert midpoints into segments, halving each of them.

        Args:
            segments: Indices of the segments to halve (sorted).
            t_mid: The parameter of each midpoint.
            mid: The screen point of each midpoint.
            keep_open: Whether the halves of each segment may need more
                refining.
            fresh: Give the halves a depth of 0, so they are never taken for
                jumps (used for the end points found at domain edges).
        """
        # The k-th new point lands at position segments[k] + 1 + k. Points
        # and segments both gain one entry there: the second half of each
        # halved segment, while its first half replaces it.
        inserted = segments + 1 + np.arange(segments.size)
        depth = (
            np.zeros(segments.size, dtype=np.int64)
            if fresh
            else self.depth[segments] + 1
        )
        self.open[segments] = keep_open
        self.depth[segments] = depth
        self.t = _insert(self.t, inserted, t_mid)
        self.points = _insert(self.points, inserted, mid)
        self.open = _insert(self.open, inserted, keep_open)
        self.depth = _insert(self.depth, inserted, depth)

    def polylines(self, max_segment_px: float, depth: int) -> list[NDArray[np.float64]]:
        """Split the samples into polylines at gaps and jumps.

        Args:
            max_segment_px: Segments longer than this that were halved
                ``depth`` times are jumps.
            depth: The depth limit (see :func:`_depth_limit`).

        Returns:
            The polylines, each with at least two points.
        """
        a, b = self.points[:-1], self.points[1:]
        finite = np.isfinite(self.points).all(axis=1)
        defined = finite[:-1] & finite[1:]
        length = _lengths(b - a)
        jump = (length > max_segment_px) & (self.depth >= depth)
        broken = ~defined | jump
        lines = []
        start = 0
        for cut in [*np.flatnonzero(broken).tolist(), len(self.points) - 1]:
            piece = self.points[start : cut + 1]
            if piece.shape[0] >= 2 and np.isfinite(piece).all():
                lines.append(piece)
            start = cut + 1
        return lines


def _refine(
    samples: _Samples,
    curve: ScreenCurve,
    box: Box,
    config: CurveSampling,
    depth: int,
    budget: int,
) -> int:
    """Halve the segments that need it, round by round.

    Args:
        samples: The samples (changed in place).
        curve: The curve in screen coordinates.
        box: The visible screen area.
        config: Sampling settings.
        depth: Most halvings of any segment (see :func:`_depth_limit`).
        budget: Evaluations still allowed.

    Returns:
        The evaluations left over.
    """
    for _ in range(depth):
        candidates = np.flatnonzero(samples.open & (samples.depth < depth))
        if candidates.size == 0 or budget <= 0:
            break
        # Segments far from the screen (further than their own length, which
        # bounds any bulge of a curve sampled finely enough) need nothing.
        a = samples.points[candidates]
        b = samples.points[candidates + 1]
        far = ~_near_box((a, b), _lengths(b - a), box)
        samples.open[candidates[far]] = False
        candidates = candidates[~far]
        if candidates.size == 0:
            break
        if candidates.size > budget:
            candidates = np.sort(_most_urgent(samples, candidates, budget))
        t_mid = 0.5 * (samples.t[candidates] + samples.t[candidates + 1])
        # A segment whose midpoint rounds to an end cannot be split any more.
        stuck = (t_mid <= samples.t[candidates]) | (t_mid >= samples.t[candidates + 1])
        samples.open[candidates[stuck]] = False
        samples.depth[candidates[stuck]] = depth
        candidates, t_mid = candidates[~stuck], t_mid[~stuck]
        if candidates.size == 0:
            break
        budget -= candidates.size
        mid = _evaluate(curve, t_mid)
        a = samples.points[candidates]
        b = samples.points[candidates + 1]
        keep_open = _needs_more(a, mid, b, box, config)
        samples.split(candidates, t_mid, mid, keep_open)
    return budget


def _needs_more(
    a: NDArray[np.float64],
    mid: NDArray[np.float64],
    b: NDArray[np.float64],
    box: Box,
    config: CurveSampling,
) -> NDArray[np.bool_]:
    """Decide which halved segments may need more refining.

    Args:
        a: Segment starts.
        mid: Segment midpoints.
        b: Segment ends.
        box: The visible screen area.
        config: Sampling settings.

    Returns:
        ``True`` where the segment is near the screen and is not yet drawn
        faithfully: it bends by more than the tolerance, is longer than
        ``max_segment_px``, or has an undefined point (a domain edge).
    """
    finite = (
        np.isfinite(a).all(axis=1),
        np.isfinite(mid).all(axis=1),
        np.isfinite(b).all(axis=1),
    )
    all_defined = finite[0] & finite[1] & finite[2]
    some_defined = finite[0] | finite[1] | finite[2]
    longest = np.maximum(_lengths(mid - a), _lengths(b - mid))
    near = _near_box((a, mid, b), 0.5 * longest, box)
    bend = _distance_to_chord(mid, a, b)
    rough = (bend > config.tolerance_px) | (longest > config.max_segment_px / 2)
    return near & np.where(all_defined, rough, some_defined)


def _near_box(
    points: tuple[NDArray[np.float64], ...], reach: NDArray[np.float64], box: Box
) -> NDArray[np.bool_]:
    """Return which segments could show any part of the curve on screen.

    A segment counts when the bounding box of its defined points, grown by
    ``reach``, touches the visible area. The growth allows for the curve
    bulging out between samples that are far apart.

    Args:
        points: Points along each segment (each of shape ``(N, 2)``).
        reach: How far the curve may stray from those points, per segment
            (NaN counts as 0).
        box: The visible screen area.

    Returns:
        A boolean mask over the segments.
    """
    low, high = points[0], points[0]
    for other in points[1:]:  # fmin and fmax ignore undefined (NaN) points
        low, high = np.fmin(low, other), np.fmax(high, other)
    grow = np.where(np.isnan(reach), 0.0, reach)
    x_min, y_min, x_max, y_max = box
    return (
        (low[:, 0] - grow <= x_max)
        & (high[:, 0] + grow >= x_min)
        & (low[:, 1] - grow <= y_max)
        & (high[:, 1] + grow >= y_min)
    )


def _distance_to_chord(
    p: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the distance from each point to its segment ``a -> b``.

    The distance is to the nearest point of the *segment* (not the infinite
    line), so a curve that doubles back past an end counts as bent.

    Args:
        p: Points, shape ``(N, 2)``.
        a: Segment starts.
        b: Segment ends.

    Returns:
        Distances in pixels.
    """
    d = b - a
    length2 = np.einsum("ij,ij->i", d, d)
    safe = np.where(length2 > 0, length2, 1.0)
    s = np.clip(np.einsum("ij,ij->i", p - a, d) / safe, 0.0, 1.0)
    nearest = a + s[:, None] * d
    return _lengths(p - nearest)


def _most_urgent(
    samples: _Samples, candidates: NDArray[np.int64], count: int
) -> NDArray[np.int64]:
    """Pick the ``count`` candidates that matter most when the budget is short.

    Args:
        samples: The samples.
        candidates: Indices of open segments.
        count: How many to pick.

    Returns:
        The longest segments on screen (undefined ends first).
    """
    a = samples.points[candidates]
    b = samples.points[candidates + 1]
    length = _lengths(b - a)
    urgency = np.where(np.isfinite(length), length, np.inf)
    return candidates[np.argsort(-urgency, kind="stable")[:count]]


def _find_edges(samples: _Samples, curve: ScreenCurve, box: Box, budget: int) -> None:
    """Close in on the edges of the curve's domain, so its ends are exact.

    Each segment with one defined and one undefined end on screen is
    bisected, always keeping the edge inside, until the defined end moves
    less than :data:`_EDGE_SETTLED_PX` per step (or misses
    :data:`_EDGE_MISSES` times in a row). Only the final defined point of each
    edge is added to the samples.

    Args:
        samples: The samples (changed in place).
        curve: The curve in screen coordinates.
        box: The visible screen area.
        budget: Evaluations still allowed.
    """
    finite = np.isfinite(samples.points).all(axis=1)
    edge = np.flatnonzero(finite[:-1] != finite[1:])
    if edge.size == 0:
        return
    defined = np.where(finite[edge], edge, edge + 1)
    x_min, y_min, x_max, y_max = box
    point = samples.points[defined]
    on_screen = (
        (point[:, 0] >= x_min)
        & (point[:, 0] <= x_max)
        & (point[:, 1] >= y_min)
        & (point[:, 1] <= y_max)
    )
    edge, defined = edge[on_screen], defined[on_screen]
    if edge.size == 0:
        return
    t_in = samples.t[defined].copy()
    t_out = np.where(defined == edge, samples.t[edge + 1], samples.t[edge])
    p_in = samples.points[defined].copy()
    active = np.ones(edge.size, dtype=bool)
    misses = np.zeros(edge.size, dtype=np.int64)
    for _ in range(_EDGE_STEPS):
        rows = np.flatnonzero(active)
        if rows.size == 0 or rows.size > budget:
            break
        budget -= rows.size
        t_mid = 0.5 * (t_in[rows] + t_out[rows])
        mid = _evaluate(curve, t_mid)
        inside = np.isfinite(mid).all(axis=1)
        moved = _lengths(mid - p_in[rows])
        grow, shrink = rows[inside], rows[~inside]
        t_in[grow], p_in[grow] = t_mid[inside], mid[inside]
        t_out[shrink] = t_mid[~inside]
        active[grow[moved[inside] < _EDGE_SETTLED_PX]] = False
        misses[grow] = 0
        misses[shrink] += 1
        active[misses >= _EDGE_MISSES] = False
        active[(t_in == t_out) | (0.5 * (t_in + t_out) == t_in)] = False
    moved_any = t_in != samples.t[defined]
    if not moved_any.any():
        return
    # Insert each edge's final defined point into its segment.
    order = np.argsort(edge[moved_any])
    segments = edge[moved_any][order]
    samples.split(
        segments,
        t_in[moved_any][order],
        p_in[moved_any][order],
        np.zeros(segments.size, dtype=bool),
        fresh=True,
    )


def _insert(
    array: NDArray[ScalarT], positions: NDArray[np.int64], values: NDArray[ScalarT]
) -> NDArray[ScalarT]:
    """Return ``array`` with ``values`` inserted so they end up at ``positions``.

    A lean :func:`numpy.insert` for the sampler's hot loop.

    Args:
        array: The array (rows are inserted along the first axis).
        positions: Final, increasing indices of the inserted rows.
        values: The rows to insert.

    Returns:
        The longer array.
    """
    result = np.empty((array.shape[0] + positions.size, *array.shape[1:]), array.dtype)
    new = np.zeros(result.shape[0], dtype=bool)
    new[positions] = True
    result[new] = values
    result[~new] = array
    return result


def _lengths(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the length of each row of an ``(N, 2)`` array of vectors.

    Args:
        vectors: Vectors in pixels.

    Returns:
        Their lengths.
    """
    return np.asarray(np.hypot(vectors[:, 0], vectors[:, 1]), dtype=np.float64)


def _evaluate(curve: ScreenCurve, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the curve, returning points with NaN rows where undefined.

    Args:
        curve: The curve in screen coordinates.
        t: Parameter values.

    Returns:
        Screen points, shape ``(N, 2)``. Infinite coordinates become NaN.
    """
    px, py = curve(t)
    points = np.column_stack(
        (np.asarray(px, dtype=np.float64), np.asarray(py, dtype=np.float64))
    )
    points[~np.isfinite(points).all(axis=1)] = np.nan
    return points
