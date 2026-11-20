"""Turning a function ``y = f(x)`` into polylines that look right on screen.

The pipeline, run once per frame for every plotted function:

1. **Robust evaluation** (see :mod:`pygridplot.core.evaluation`). Points
   where the function raises, or has no real value, become NaN, so a user
   function that throws can never break the frame.
2. **World-anchored samples.** Samples start one per pixel column at
   ``x = k * step`` for whole numbers ``k``. Panning the view therefore
   reuses exactly the same world points, and curves move rigidly instead of
   shimmering.
3. **Adaptive refinement** (vectorised, a few rounds). Segments whose
   midpoint lies more than :attr:`SamplingConfig.tolerance_px` above or
   below the chord are halved again. Segments with an undefined end (a domain edge, as
   ``sqrt(x)`` has at 0) are bisected so the curve reaches the edge.
   Segments that are entirely above or below the screen are left alone.
4. **Discontinuities.** A segment that is still far from straight after
   the last round, rising more than :attr:`SamplingConfig.jump_px` pixels,
   gets one last midpoint test. If its rise sits entirely in one half (a
   step of ``floor(x)``) or the midpoint overshoots both ends (the pole of
   ``1/x`` or ``tan(x)``), the curve is broken there instead of being joined
   by a vertical line. Steep but continuous branches, like the sides of a
   pole, stay joined, so they run all the way to the edge of the plane.

All tolerances are measured in screen pixels, which is why the sampler
needs the axis scales.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.evaluation import Function, _evaluate_quietly

_STEEP_PX: Final = 2.0
"""Pixel rise per sample above which a segment is always checked."""

_SHARED_RISE: Final = 0.1
"""Smallest fraction of a segment's rise each half must have to count as smooth."""

_CHASE_STEPS: Final = 40
"""Most bisection steps spent closing in on one pole or domain edge."""


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """How functions are sampled. The defaults suit almost every function.

    Attributes:
        samples_per_px: Initial samples per pixel column.
        tolerance_px: Largest allowed distance, in pixels, between the curve
            and the polyline drawn for it.
        max_rounds: Most halvings of any one segment. Each halves its width, so
            8 rounds pin jumps down to 1/256 of a pixel.
        max_samples: Total budget of function evaluations per frame. When it
            runs out, the segments that change fastest are refined first.
        jump_px: Smallest remaining jump, in pixels, that is treated as a
            discontinuity once a segment cannot be refined any further.
    """

    samples_per_px: float = 1.0
    tolerance_px: float = 0.25
    max_rounds: int = 8
    max_samples: int = 16_384
    jump_px: float = 1.0

    def __post_init__(self) -> None:
        """Validate the settings."""
        if not (0.05 <= self.samples_per_px <= 16.0):
            msg = "samples_per_px must be within [0.05, 16]"
            raise ValueError(msg)
        if not self.tolerance_px > 0 or not self.jump_px > 0:
            msg = "tolerance_px and jump_px must be positive"
            raise ValueError(msg)
        if self.max_rounds < 0 or self.max_samples < 16:
            msg = "max_rounds must be non-negative and max_samples at least 16"
            raise ValueError(msg)


DEFAULT_SAMPLING: Final = SamplingConfig()
"""The sampling settings used when none are given."""


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def sample_function(
    func: Function,
    x_range: tuple[float, float],
    scale: tuple[float, float],
    y_band: tuple[float, float],
    config: SamplingConfig = DEFAULT_SAMPLING,
) -> list[NDArray[np.float64]]:
    """Sample ``func`` over ``x_range`` into polylines that look right on screen.

    Args:
        func: The function to plot.
        x_range: ``(x_min, x_max)`` world interval to sample (inclusive).
        scale: ``(scale_x, scale_y)`` in pixels per world unit. Tolerances are
            measured in these pixels.
        y_band: ``(y_min, y_max)`` world interval that is on screen (including
            any margin). Parts of the curve outside it are not refined.
        config: Sampling settings.

    Returns:
        The curve as a list of polylines in world coordinates, each an array of
        shape ``(N, 2)`` with ``N >= 2``. The curve is broken wherever the
        function is undefined or jumps.
    """
    lo, hi = x_range
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        return []
    xs = _initial_samples(lo, hi, scale[0], config)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ys = _evaluate_quietly(func, xs)
        xs, ys, broken = _refine(func, xs, ys, scale, y_band, config)
    return _polylines(xs, ys, broken, y_band)


def _initial_samples(
    lo: float, hi: float, scale_x: float, config: SamplingConfig
) -> NDArray[np.float64]:
    """Return evenly spaced, world-anchored sample positions covering ``[lo, hi]``.

    Args:
        lo: Start of the interval.
        hi: End of the interval.
        scale_x: Horizontal pixels per world unit.
        config: Sampling settings.

    Returns:
        Sorted, distinct x values: ``lo``, every multiple of the step inside
        the interval, and ``hi``.
    """
    step = 1.0 / (scale_x * config.samples_per_px)
    first, last = math.ceil(lo / step), math.floor(hi / step)
    count = last - first + 1
    if 0 < count <= config.max_samples:
        inner = np.arange(first, last + 1, dtype=np.float64) * step
    else:  # far too many: fall back to an unanchored even spread
        inner = np.linspace(lo, hi, min(max(count, 2), config.max_samples))
    return np.unique(np.concatenate(([lo], inner, [hi])))


def _refine(
    func: Function,
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    scale: tuple[float, float],
    band: tuple[float, float],
    config: SamplingConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Adaptively add samples where the polyline would visibly misrepresent f.

    Only the segments still being refined are tracked, as explicit endpoint
    pairs, so each round costs little once most of the curve is settled. The
    new samples are merged into the curve once at the end.

    Args:
        func: The function.
        xs: Initial sample positions (sorted).
        ys: Values at ``xs``.
        scale: Pixels per world unit along x and y.
        band: Visible y interval in world units.
        config: Sampling settings.

    Returns:
        ``(xs, ys, broken)``, where ``broken[i]`` says the curve must not be
        drawn between samples ``i`` and ``i + 1``.
    """
    budget = config.max_samples - xs.shape[0]
    seg = _initial_candidates(xs, ys, scale, band, config)
    xa, ya, xb, yb = xs[seg], ys[seg], xs[seg + 1], ys[seg + 1]
    added_x: list[NDArray[np.float64]] = [xs]
    added_y: list[NDArray[np.float64]] = [ys]
    stuck: list[NDArray[np.float64]] = []  # starts of segments that cannot be halved
    rounds_run = 0
    out_of_budget = False
    for _ in range(config.max_rounds):
        if xa.shape[0] == 0:
            break
        if budget <= 0:
            out_of_budget = True
            break
        rounds_run += 1
        if xa.shape[0] > budget:
            keep = _most_urgent(ya, yb, budget)
            xa, ya, xb, yb = xa[keep], ya[keep], xb[keep], yb[keep]
        xm = 0.5 * (xa + xb)
        splittable = (xm > xa) & (xm < xb)
        stuck.append(xa[~splittable])
        xa, ya, xb, yb, xm = (v[splittable] for v in (xa, ya, xb, yb, xm))
        if xm.shape[0] == 0:
            break
        ym = _evaluate_quietly(func, xm)
        budget -= xm.shape[0]
        added_x.append(xm)
        added_y.append(ym)
        left, right = _continue_halves(ya, ym, yb, scale[1], config.tolerance_px)
        xa, ya, xb, yb = (
            np.concatenate((xa[left], xm[right])),
            np.concatenate((ya[left], ym[right])),
            np.concatenate((xm[left], xb[right])),
            np.concatenate((ym[left], yb[right])),
        )
        visible = _segment_visible(ya, yb, band)
        xa, ya, xb, yb = xa[visible], ya[visible], xb[visible], yb[visible]
    # Segments still pending were only confirmed as non-straight if at least
    # one round ran and refinement was not cut short by the budget.
    # Classifying the pending segments needs one evaluation each; without the
    # budget for that, they are treated as if refinement had been cut short.
    confirmed = rounds_run > 0 and not out_of_budget and xa.shape[0] <= budget
    pending = xa
    if confirmed and xa.shape[0]:
        defined_a, defined_b = np.isfinite(ya), np.isfinite(yb)
        both = defined_a & defined_b
        edge = defined_a ^ defined_b
        # Jumps and poles between two defined values.
        xa_b, ya_b, xb_b, yb_b = xa[both], ya[both], xb[both], yb[both]
        jump, pole = _classify_jumps(func, xa_b, ya_b, xb_b, yb_b)
        budget -= xa_b.shape[0]
        chase = pole & (np.sign(ya_b) * np.sign(yb_b) < 0)
        pole_x, pole_y, pole_starts = _chase(
            func,
            (xa_b[chase], ya_b[chase], xb_b[chase], yb_b[chase]),
            band,
            scale[0],
            mode="pole",
            budget=budget,
        )
        budget -= pole_x.shape[0]
        # Domain edges: follow the curve right up to where it stops (or leaves).
        edge_x, edge_y, _ = _chase(
            func,
            (xa[edge], ya[edge], xb[edge], yb[edge]),
            band,
            scale[0],
            mode="edge",
            budget=budget,
        )
        added_x += [edge_x, pole_x]
        added_y += [edge_y, pole_y]
        pending = np.concatenate((xa_b[jump & ~chase], pole_starts))
    all_x = np.concatenate(added_x)
    order = np.argsort(all_x, kind="stable")
    all_x = all_x[order]
    all_y = np.concatenate(added_y)[order]
    stuck_starts = np.concatenate(stuck) if stuck else xa[:0]
    broken = _breaks(
        all_x, all_y, pending, stuck_starts, confirmed, scale, band, config
    )
    return all_x, all_y, broken


def _classify_jumps(
    func: Function,
    xa: NDArray[np.float64],
    ya: NDArray[np.float64],
    xb: NDArray[np.float64],
    yb: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Tell real discontinuities apart from steep, continuous pieces of curve.

    These segments are still far from straight after the last round, so one
    more midpoint decides. Across a step (as in ``floor``) the midpoint takes
    one end's value, so the whole rise sits in one half. Across a pole (as in
    ``1/x``) the midpoint overshoots both ends. Along a steep but continuous
    branch, such as the side of a pole, the midpoint lies strictly between the
    ends and both halves share the rise.

    Args:
        func: The function.
        xa: Segment starts.
        ya: Values at the starts.
        xb: Segment ends.
        yb: Values at the ends.

    Returns:
        ``(jump, pole)``: which segments are discontinuities (segments too
        narrow to split count as discontinuities), and which of those look
        like poles (the midpoint overshoots both ends).
    """
    xm = 0.5 * (xa + xb)
    splittable = (xm > xa) & (xm < xb)
    jump = np.ones(xa.shape[0], dtype=bool)
    pole = np.zeros(xa.shape[0], dtype=bool)
    ym = _evaluate_quietly(func, xm[splittable])
    a, b = ya[splittable], yb[splittable]
    low, high = np.minimum(a, b), np.maximum(a, b)
    between = (ym > low) & (ym < high)
    shared = np.minimum(np.abs(ym - a), np.abs(b - ym)) > _SHARED_RISE * (high - low)
    jump[splittable] = ~(between & shared)
    pole[splittable] = np.isfinite(ym) & (np.abs(ym) > np.maximum(np.abs(a), np.abs(b)))
    return jump, pole


def _chase(
    func: Function,
    segments: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    band: tuple[float, float],
    scale_x: float,
    *,
    mode: str,
    budget: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Close in on poles and domain edges, so the curve's ends are exact.

    Some features are so narrow that ordinary refinement stops before the
    curve has left the screen. Examples are the poles of the gamma function
    at large negative x, and its undefined values exactly at negative
    integers. Each such segment is bisected like a root search, always
    keeping the feature inside:

    * ``"pole"``: the ends have opposite signs, and the end with the same
      sign as the midpoint moves. Both ends climb a branch towards infinity.
    * ``"edge"``: one end is undefined, and the end that agrees with the
      midpoint (defined or undefined) moves, so the defined end approaches
      the edge of the function's domain.

    Bisection stops once the curve has left the screen (no defined end is
    visible any more), or the interval is narrower than a millionth of a pixel
    (a billionth for poles, whose branches can stay on screen until extremely
    close to the pole).

    Args:
        func: The function.
        segments: ``(xa, ya, xb, yb)``: segment starts and ends with their
            values.
        band: Visible y interval in world units.
        scale_x: Horizontal pixels per world unit.
        mode: ``"pole"`` or ``"edge"``.
        budget: Most function evaluations to spend.

    Returns:
        ``(x, y, starts)``: the new samples, and the start of each final,
        narrow segment.
    """
    xa, ya, xb, yb = (values.copy() for values in segments)
    lo, hi = band
    finest = (1e-9 if mode == "pole" else 1e-6) / scale_x
    new_x: list[NDArray[np.float64]] = []
    new_y: list[NDArray[np.float64]] = []
    for _ in range(_CHASE_STEPS):
        with np.errstate(invalid="ignore"):
            visible_a = (ya >= lo) & (ya <= hi)
            visible_b = (yb >= lo) & (yb <= hi)
        xm = 0.5 * (xa + xb)
        wide = (xb - xa > finest) & (xm > xa) & (xm < xb)
        active = np.flatnonzero((visible_a | visible_b) & wide)
        if active.shape[0] == 0 or active.shape[0] > budget:
            break
        budget -= active.shape[0]
        ym = _evaluate_quietly(func, xm[active])
        new_x.append(xm[active])
        new_y.append(ym)
        if mode == "pole":
            usable = np.isfinite(ym)
            move_a = usable & (np.sign(ym) == np.sign(ya[active]))
            move_b = usable & ~move_a
            # An undefined midpoint ends the chase for that pole.
            ya[active[~usable]] = yb[active[~usable]] = np.inf
        else:
            move_a = np.isfinite(ym) == np.isfinite(ya[active])
            move_b = ~move_a
        xa[active[move_a]], ya[active[move_a]] = xm[active][move_a], ym[move_a]
        xb[active[move_b]], yb[active[move_b]] = xm[active][move_b], ym[move_b]
    empty = np.zeros(0)
    x = np.concatenate(new_x) if new_x else empty
    y = np.concatenate(new_y) if new_y else empty
    return x, y, xa


def _initial_candidates(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    scale: tuple[float, float],
    band: tuple[float, float],
    config: SamplingConfig,
) -> NDArray[np.intp]:
    """Return the segments worth refining in the first round.

    A segment is checked if it is visible and either curves noticeably (judged
    from the second differences of the existing samples, which costs no
    extra evaluations), rises steeply, or has one undefined end.

    Args:
        xs: Sample positions.
        ys: Sample values.
        scale: Pixels per world unit along x and y.
        band: Visible y interval in world units.
        config: Sampling settings.

    Returns:
        Sorted indices of candidate segments.
    """
    if xs.shape[0] < 2:
        return np.zeros(0, dtype=np.intp)
    with np.errstate(all="ignore"):
        y_px = ys * scale[1]
        rise = np.abs(np.diff(y_px))
        bend = np.full(xs.shape[0], np.inf)
        bend[1:-1] = np.abs(y_px[:-2] - 2.0 * y_px[1:-1] + y_px[2:])
        curved = np.maximum(bend[:-1], bend[1:]) > 8.0 * config.tolerance_px
        steep = rise > max(_STEEP_PX, 8.0 * config.tolerance_px)
    finite = np.isfinite(ys)
    both = finite[:-1] & finite[1:]
    one = finite[:-1] ^ finite[1:]
    candidates = (both & (curved | steep)) | one
    visible = _segment_visible(ys[:-1], ys[1:], band)
    return np.flatnonzero(candidates & visible)


def _continue_halves(
    ya: NDArray[np.float64],
    ym: NDArray[np.float64],
    yb: NDArray[np.float64],
    scale_y: float,
    tolerance_px: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Decide which halves of each just-split segment need more refinement.

    A half with exactly one defined end is always bisected further (to find
    a domain edge). A half with two defined ends continues only if its parent
    curved noticeably or straddled a domain edge.

    Args:
        ya: Values at the split segments' starts.
        ym: Values at their midpoints.
        yb: Values at their ends.
        scale_y: Vertical pixels per world unit.
        tolerance_px: Allowed deviation in pixels.

    Returns:
        ``(left, right)`` masks over the split segments.
    """
    fa, fm, fb = np.isfinite(ya), np.isfinite(ym), np.isfinite(yb)
    curved = fa & fb & fm & (_deviation(ya, ym, yb, scale_y) > tolerance_px)
    parent_flag = curved | (fa ^ fb)
    left = (fa | fm) & (~(fa & fm) | parent_flag)
    right = (fm | fb) & (~(fm & fb) | parent_flag)
    return left, right


def _deviation(
    ya: NDArray[np.float64],
    ym: NDArray[np.float64],
    yb: NDArray[np.float64],
    scale_y: float,
) -> NDArray[np.float64]:
    """Return how far each midpoint lies above or below its segment's chord.

    The distance is measured vertically, in pixels. For a smooth curve it
    shrinks quickly as segments are halved. Across a jump it stays at about
    half the jump however narrow the segment gets, which is what lets jumps
    be told apart from steep but continuous curves. (The perpendicular
    distance would not work: it shrinks with the segment's width even across
    a jump.)

    Args:
        ya: Values at segment starts.
        ym: Values at the midpoints.
        yb: Values at segment ends.
        scale_y: Vertical pixels per world unit.

    Returns:
        Vertical distances in pixels (``inf`` where not computable).
    """
    with np.errstate(all="ignore"):
        dist = np.abs(ym - 0.5 * (ya + yb)) * scale_y
    return np.where(np.isfinite(dist), dist, np.inf)


def _most_urgent(
    ya: NDArray[np.float64], yb: NDArray[np.float64], budget: int
) -> NDArray[np.intp]:
    """Pick the ``budget`` segments with the biggest rise.

    Big rises are where discontinuities hide, so they are resolved first.
    Segments with an undefined end rank highest.

    Args:
        ya: Values at segment starts.
        yb: Values at segment ends.
        budget: How many segments may be refined.

    Returns:
        Sorted indices of the segments to refine.
    """
    with np.errstate(all="ignore"):
        rise = np.abs(yb - ya)
    rise = np.where(np.isfinite(rise), rise, np.inf)
    return np.sort(np.argpartition(-rise, budget - 1)[:budget])


def _segment_visible(
    ya: NDArray[np.float64], yb: NDArray[np.float64], band: tuple[float, float]
) -> NDArray[np.bool_]:
    """Return which segments might be on screen.

    Args:
        ya: Values at segment starts.
        yb: Values at segment ends.
        band: Visible y interval in world units.

    Returns:
        ``False`` for segments entirely above or entirely below the band, or
        with no defined end. A segment with one defined end counts as visible
        if that end is inside the band.
    """
    lo, hi = band
    fa, fb = np.isfinite(ya), np.isfinite(yb)
    with np.errstate(invalid="ignore"):
        above = (ya > hi) & (yb > hi)
        below = (ya < lo) & (yb < lo)
        defined = np.where(fa, ya, yb)
        end_inside = (defined >= lo) & (defined <= hi)
    return np.where(fa & fb, ~(above | below), (fa ^ fb) & end_inside)


def _breaks(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    pending: NDArray[np.float64],
    stuck: NDArray[np.float64],
    confirmed: bool,
    scale: tuple[float, float],
    band: tuple[float, float],
    config: SamplingConfig,
) -> NDArray[np.bool_]:
    """Decide where the curve must not be joined.

    Args:
        xs: Final sample positions (sorted).
        ys: Final sample values.
        pending: Start positions of segments that still needed refinement
            when it stopped.
        stuck: Start positions of segments that could not be halved further.
        confirmed: Whether the pending segments were confirmed as not straight
            by a full last round. If not (no rounds allowed, or the budget ran
            out), only rises too big to belong to a curve are broken.
        scale: Pixels per world unit along x and y.
        band: Visible y interval in world units.
        config: Sampling settings.

    Returns:
        ``broken[i]`` is ``True`` if samples ``i`` and ``i + 1`` must not be
        connected.
    """
    finite = np.isfinite(ys)
    broken = ~(finite[:-1] & finite[1:])
    with np.errstate(all="ignore"):
        rise = np.abs(np.diff(ys)) * scale[1]
    last = xs.shape[0] - 2
    unresolved = np.zeros_like(broken)
    unresolved[np.minimum(np.searchsorted(xs, stuck), last)] = True
    pending_at = np.minimum(np.searchsorted(xs, pending), last)
    if confirmed:
        unresolved[pending_at] = True
    broken |= unresolved & (rise > config.jump_px)
    if not confirmed:
        # Refinement was cut short: only break rises too big to be a curve.
        band_px = (band[1] - band[0]) * scale[1]
        cut_short = np.zeros_like(broken)
        cut_short[pending_at] = True
        broken |= cut_short & (rise > band_px)
    return broken


def _polylines(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    broken: NDArray[np.bool_],
    band: tuple[float, float],
) -> list[NDArray[np.float64]]:
    """Cut the samples into polylines at every break.

    Args:
        xs: Sample positions.
        ys: Sample values.
        broken: Where consecutive samples must not be joined.
        band: Visible y interval (polylines entirely outside it are dropped).

    Returns:
        Polylines with at least two points, as ``(N, 2)`` arrays.
    """
    starts = np.flatnonzero(np.concatenate(([True], broken)))
    ends = np.concatenate((starts[1:], [xs.shape[0]]))
    lo, hi = band
    lines = []
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        if end - start < 2 or not np.isfinite(ys[start]):
            continue
        y = ys[start:end]
        if np.max(y) < lo or np.min(y) > hi:
            continue
        lines.append(np.column_stack((xs[start:end], y)))
    return lines
