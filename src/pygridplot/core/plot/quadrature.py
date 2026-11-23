"""Adaptive Gauss-Kronrod quadrature, for many integrals at once.

:func:`integrate` computes several definite integrals of one function
together (the *pieces*: one for :func:`~pygridplot.core.plot.integrals.integral`,
one per gap between query points for an antiderivative), each starting
from one or more intervals. Every round evaluates the function once, at
the nodes of all the intervals still being refined, so a NumPy function is
called a few times in all however many pieces there are, and the work per
round is vectorised.

**The rule.** Each interval is integrated with the 21-point Gauss-Kronrod
rule, whose 10 Gauss nodes give a second, less accurate estimate at no
extra cost. The difference between the two estimates is turned into an
error estimate as QUADPACK does (``qk21``): scaled by how much the function
varies over the interval, sharply reduced when the two estimates agree
closely (the Kronrod estimate is then far better than the Gauss one), and
never below the rounding error of the sum.

Neither rule samples the last 0.43% at either end of an interval, so a
jump or a kink there would be invisible to both, and the estimate would
be confidently wrong. So each interval is also probed once near each end
(:data:`PROBE`), and the difference between the probe and the nodes'
interpolating polynomial extrapolated to it, times the sliver's width, is
added to the error: it bounds what the sliver hides. The probes are just
inside the ends, not at them, because a jump exactly at an end (``floor``
integrated up to 1) is a single point that changes nothing, yet seen from
the ends alone it looks the same as a jump just inside.

**Refining.** A piece is done when its total error is at most
:data:`TOLERANCE` times the integral of ``|f|`` over it: the error is
relative to the size of what is being added up, so an integral that
cancels to zero (``sin`` over a period) still converges, and the result
does not depend on the units. Until then, the intervals of the piece whose
error exceeds an equal share of its tolerance are cut: graded towards
their ends if they are one or two (:data:`GRADED`), into eighths if they
are few (:data:`FEW`), else in half. The criterion is global within a
piece rather than per interval, which is what lets integrable
singularities converge: at ``1/sqrt(x)`` the interval touching 0 always
has a large error *relative to its own size*, but its share of the whole
shrinks with every cut.

**Undefined points and divergence.** Robustness follows the rest of the
library: nothing raises, and a result is NaN when no honest number exists.

- An interval with one undefined node (``sin(x) / x`` at 0, or a pole
  that a node lands on) has an infinite error, so it is cut: the point
  becomes an end of the parts, where no node lies. An isolated undefined
  point does not change an integral.
- An interval with two or more undefined nodes (or probes) means the
  function is undefined on a stretch of the piece (``sqrt(x)`` left of
  0), so the piece is undefined (NaN) at once.
- A piece whose error has not halved in :data:`STALL_ROUNDS` rounds is
  given up: cutting no longer helps, as at a non-integrable singularity
  (``1/x`` at 0), where the error of the interval touching it never
  shrinks. Rounds and evaluations are limited too (:data:`MAX_ROUNDS`,
  :data:`MAX_NEW_INTERVALS`), for integrands that oscillate thousands of
  times.
- Intervals too narrow to cut in floating point are kept as they are.

A piece that could not reach :data:`TOLERANCE` keeps its estimate if its
error is within :data:`ACCEPTABLE` (relative to the integral of ``|f|``),
and is NaN otherwise. In practice the estimate falls short of the target
only at singularities at the end of an interval away from 0, such as
``1 / sqrt(1 - x²)`` at 1, where floating point cannot get closer to the
singularity than about ``1e-13`` (times its distance from 0); there it is
still good to about 8 digits near ±1, and is NaN beyond about ±10.
Divergent integrals are NaN.

The process is deterministic: the same function values give the same
intervals, evaluations and results, as the plot cache requires.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

Integrand = Callable[[NDArray[np.float64]], NDArray[np.float64]]
"""A vectorised, robust function: an array of points to an array of values.

It must never raise and return one float per point (NaN where undefined),
as the robust evaluation of :mod:`pygridplot.core.plot.evaluation` does.
"""

_KRONROD_NODES: Final = (
    0.995657163025808080735527280689003,
    0.973906528517171720077964012084452,
    0.930157491355708226001207180059508,
    0.865063366688984510732096688423493,
    0.780817726586416897063717578345042,
    0.679409568299024406234327365114874,
    0.562757134668604683339000099272694,
    0.433395394129247190799265943165784,
    0.294392862701460198131126603103866,
    0.148874338981631210884826001129720,
    0.0,
)
_KRONROD_WEIGHTS: Final = (
    0.011694638867371874278064396062192,
    0.032558162307964727478818972459390,
    0.054755896574351996031381300244580,
    0.075039674810919952767043140916190,
    0.093125454583697605535065465083366,
    0.109387158802297641899210590325805,
    0.123491976262065851077958109831074,
    0.134709217311473325928054001771707,
    0.142775938577060080797094273138717,
    0.147739104901338491374841515972068,
    0.149445554002916905664936468389821,
)
_GAUSS_WEIGHTS: Final = (
    0.066671344308688137593568809893332,
    0.149451349150580593145776339657697,
    0.219086362515982043995534934228163,
    0.269266719309996355091226921569469,
    0.295524224714752870173892994651338,
)


def _rule() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the 21 nodes on ``[-1, 1]`` and both rules' weights at them.

    QUADPACK lists the nodes of one half, outermost first, with the Gauss
    nodes at odd positions; the Gauss weights are zero at the other nodes.
    """
    half = np.array(_KRONROD_NODES[:10])
    nodes = np.concatenate([-half, [0.0], half[::-1]])
    kronrod = np.array(_KRONROD_WEIGHTS)
    weights = np.concatenate([kronrod[:10], kronrod[10:], kronrod[9::-1]])
    gauss = np.zeros(21)
    for j, i in enumerate(range(1, 10, 2)):
        gauss[i] = gauss[20 - i] = _GAUSS_WEIGHTS[j]
    return nodes, weights, gauss


NODES, KRONROD, GAUSS = _rule()
"""The 21 Gauss-Kronrod nodes on ``[-1, 1]``, and the Kronrod and Gauss weights.

The Kronrod rule integrates polynomials of degree 31 exactly, the Gauss
rule (10 of the nodes) those of degree 19."""

_ENDS_TO_NODES: Final = np.stack([(1.0 - NODES) / 2.0, (1.0 + NODES) / 2.0])
"""``(2, 21)``: an interval's ends ``(lo, hi)`` times this are its nodes."""

_ENDS_TO_HALF: Final = np.array([-0.5, 0.5])
"""An interval's ends times this are its half width (without overflow)."""

_BOTH_RULES: Final = np.column_stack([KRONROD, GAUSS])
"""``(21, 2)``: values at the nodes times this are both rules' sums."""


_SLIVER: Final = 1.0 - float(NODES[-1])
"""The part of each half of an interval beyond its outermost node (0.43%)."""

PROBE: Final = 1.0 - 1e-3 * _SLIVER
"""Where each interval is probed near its ends, on ``[-1, 1]``.

Neither rule samples the slivers beyond the outermost nodes, so a jump or
a kink there would go unnoticed. The function is evaluated once more near
each end, inside the sliver but not at the end itself: a jump exactly at
an end (``floor`` integrated up to 1) is a single point and changes
nothing, and would otherwise look like one just inside."""


def _extrapolation() -> NDArray[np.float64]:
    """Return the Lagrange weights of the 21 nodes at both probes."""
    weights = np.ones((21, 2))
    for j in range(21):
        others = np.delete(NODES, j)
        for side, x in enumerate((-PROBE, PROBE)):
            weights[j, side] = np.prod((x - others) / (NODES[j] - others))
    return weights


_TO_PROBES: Final = _extrapolation()
"""``(21, 2)``: values at the nodes times this are the interpolating polynomial
at both probes (well conditioned: the weights' sizes add up to about 4)."""

_ENDS_TO_POINTS: Final = np.column_stack(
    [
        np.array([(1.0 + PROBE) / 2.0, (1.0 - PROBE) / 2.0]),
        _ENDS_TO_NODES,
        np.array([(1.0 - PROBE) / 2.0, (1.0 + PROBE) / 2.0]),
    ]
)
"""``(2, 23)``: an interval's ends times this are its probes and its nodes."""


def _place(
    ends: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return points placed in intervals: ``lo * weights[0] + hi * weights[1]``.

    This is ``ends @ weights``, written out because a matrix product goes
    through BLAS, whose kernels may or may not fuse the multiply and the add
    (FMA) depending on the platform and the shapes: with Accelerate on
    macOS arm64, a product with 21 columns and one with 23 rounded the same
    node differently in the last bit. Two multiplies and an add round the
    same way everywhere, so the nodes :func:`integrate` evaluates and the
    ones :func:`nodes_of` returns are identical on every platform.

    Args:
        ends: ``(m, 2)``: the intervals' left and right ends.
        weights: ``(2, k)``: the weights of the ends for each point.

    Returns:
        ``(m, k)``: the points.
    """
    points: NDArray[np.float64] = ends[:, :1] * weights[0] + ends[:, 1:] * weights[1]
    return points


def _tail() -> NDArray[np.float64]:
    """Return the weights giving the two highest Legendre coefficients.

    The 21 values at the nodes determine a polynomial of degree 20; these
    weights give its coefficients of ``P₁₉`` and ``P₂₀`` (the nodes are well
    conditioned for interpolation: the Vandermonde matrix's condition
    number is about 8).
    """
    vandermonde = np.polynomial.legendre.legvander(NODES, 20)
    return np.asarray(np.linalg.inv(vandermonde)[19:21].T, dtype=np.float64)


_TO_TAIL: Final = _tail()
"""``(21, 2)``: values at the nodes times this are the interpolant's two
highest Legendre coefficients."""

TAIL_FACTOR: Final = 8.0
"""The error estimate is at least this times ``|(c₁₉, c₂₀)|`` times the half width.

QUADPACK's estimate rests on one difference, Kronrod minus Gauss, which
for an integrand that is not smooth (a kink, a jump, a cusp) crosses zero
at particular positions of the feature: there both rules are wrong by the
same amount and the estimate is tiny (about 1% of kink positions, in a
test of 40,000). The highest Legendre coefficients of the interpolant do
not decay at such features, and an odd and an even one do not vanish
together, so this bound catches every position in that test, with a
margin of 1.2 for the hardest (a square-root cusp). For smooth integrands
the coefficients are at rounding level by the time the rules agree, so
the bound costs nothing there."""

TOLERANCE: Final = 1e-10
"""Target error of each piece, relative to the integral of ``|f|`` over it."""

ACCEPTABLE: Final = 1e-6
"""Largest relative error for which a piece that missed the target keeps its value."""

STALL_ROUNDS: Final = 8
"""Rounds within which a piece's error must halve, or the piece is given up."""

MAX_ROUNDS: Final = 120
"""Most rounds of refinement."""

MAX_NEW_INTERVALS: Final = 8192
"""Most intervals evaluated after the first round, over all pieces."""

GRADED: Final = 2
"""Most intervals refined in a round for which the parts are graded.

When only one or two intervals need refining, the trouble is usually a
singularity or a steep stretch at one of their ends: at the lower limit of
an antiderivative, at ``1/sqrt(x)``'s 0, at a domain edge. Each is then cut
into a few uniform parts and parts that shrink geometrically towards both
ends (down to ``2⁻²⁴`` of the width), so an end singularity is closed in on
by 24 halvings per round instead of 3, while a jump in the middle is still
cut eightfold."""

FEW: Final = 16
"""Most intervals refined in a round for which each is cut into :data:`PARTS`.

Near jumps and kinks only a few intervals need refining, over many rounds;
cutting each into more parts then takes fewer rounds for about the same
number of evaluations (a third of the rounds of halving, for a third more
evaluations). When many intervals need refining, as for an integrand that
oscillates, halving wastes the least."""

PARTS: Final = 8
"""How many uniform parts each interval is cut into when only a few are refined."""

_CUTS: Final = {
    n: np.stack([1.0 - np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, 1.0, n + 1)])
    for n in (2, PARTS)
}
"""``(2, n + 1)``: an interval's ends times this are the ends of its ``n`` parts."""

_GRADED: Final = np.concatenate(
    [
        2.0 ** -np.arange(24, 0, -4),
        np.linspace(0.0, 1.0, PARTS + 1)[1:-1],
        1.0 - 2.0 ** -np.arange(4, 25, 4),
    ]
)
"""The inner cuts of a graded interval, as fractions of its width (increasing)."""

_EPS: Final = float(np.finfo(np.float64).eps)

_ROUNDING: Final = 50.0 * _EPS
"""Smallest error estimate, relative to the integral of ``|f|`` (as QUADPACK)."""

_RESOLUTION: Final = 512.0 * _EPS
"""Narrowest part, relative to its coordinates, whose 21 nodes are distinct.

Narrower, the outer nodes would round onto the part's ends."""

_NARROWEST: Final = PARTS * _RESOLUTION
"""Width, relative to the coordinates, below which an interval is not cut."""

_TINY: Final = 1e-280
"""Absolute width below which an interval is not cut (near zero)."""


@dataclass(frozen=True, slots=True)
class Quadrature:
    """The result of :func:`integrate`.

    Attributes:
        values: The integral over each piece, NaN where it is undefined,
            diverges or could not be computed accurately.
        errors: The estimated absolute error of each piece (infinite for
            undefined pieces).
        magnitudes: The integral of ``|f|`` over each piece, the scale of
            its tolerance.
        lo: The left ends of the final intervals (only with ``samples=True``,
            otherwise empty), in no particular order.
        hi: Their right ends.
        piece: The piece each final interval belongs to.
        samples: The function's values at each final interval's 21 nodes,
            shape ``(len(lo), 21)``, with 0 at undefined nodes.
    """

    values: NDArray[np.float64]
    errors: NDArray[np.float64]
    magnitudes: NDArray[np.float64]
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]
    piece: NDArray[np.intp]
    samples: NDArray[np.float64]


@dataclass(slots=True)
class _Intervals:
    """The intervals being refined, with their estimates (parallel arrays).

    Attributes:
        ends: ``(m, 2)``: each interval's left and right end.
        piece: The piece of each interval.
        estimates: ``(m, 3)``: the value, error and integral of ``|f|``.
        samples: ``(m, 21)`` values at the nodes, or ``(m, 0)`` if not kept.
    """

    ends: NDArray[np.float64]
    piece: NDArray[np.intp]
    estimates: NDArray[np.float64]
    samples: NDArray[np.float64]

    def take(self, which: NDArray[np.bool_] | NDArray[np.intp]) -> _Intervals:
        """Return the intervals chosen by a mask or by indices."""
        return _Intervals(
            self.ends[which],
            self.piece[which],
            self.estimates[which],
            self.samples[which],
        )


def integrate(
    func: Integrand,
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    piece: NDArray[np.intp],
    pieces: int,
    *,
    samples: bool = False,
) -> Quadrature:
    """Integrate a function over several pieces at once, adaptively.

    Args:
        func: The integrand, vectorised and robust (see :data:`Integrand`).
        lo: The left ends of the starting intervals (finite, ``lo < hi``).
        hi: Their right ends.
        piece: Which piece each starting interval belongs to, in
            ``[0, pieces)``. Every piece needs at least one interval.
        pieces: The number of pieces.
        samples: Also return the final intervals and the function's values
            at their nodes (for Fourier coefficients, which integrate many
            weights against the same samples).

    Returns:
        The integral over each piece, with error estimates.
    """
    values = np.full(pieces, np.nan)
    errors = np.full(pieces, np.inf)
    magnitudes = np.zeros(pieces)
    undefined = np.zeros(pieces, dtype=bool)
    kept: list[_Intervals] = []
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ends = np.column_stack(
            [np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)]
        )
        work = _evaluate(
            func, ends, np.asarray(piece, dtype=np.intp), undefined, samples
        )
        value, error, magnitude = _totals(work, pieces)
        if not undefined.any() and np.all(error <= TOLERANCE * magnitude):
            # Smooth integrands usually converge at once.
            return _result(value, error, magnitude, [work] if samples else None)
        open_ = np.ones(pieces, dtype=bool)
        history: list[NDArray[np.float64]] = []
        evaluated = 0
        for round_ in range(MAX_ROUNDS + 1):
            if round_:
                value, error, magnitude = _totals(work, pieces)
            target = TOLERANCE * magnitude
            history.append(error)
            converged = error <= target
            stalled = (
                ~(error < 0.5 * history[-1 - STALL_ROUNDS])
                if round_ >= STALL_ROUNDS
                else np.zeros(pieces, dtype=bool)
            )
            counts = np.bincount(work.piece, minlength=pieces)
            share = target / (2.0 * np.maximum(counts, 1))
            candidate = np.flatnonzero(work.estimates[:, 1] > share[work.piece])
            candidate = candidate[_cuttable(work.ends[candidate])]
            cuttable = np.zeros(pieces, dtype=bool)
            cuttable[work.piece[candidate]] = True
            closing = open_ & (undefined | converged | stalled | ~cuttable)
            if round_ == MAX_ROUNDS or evaluated >= MAX_NEW_INTERVALS:
                closing = open_.copy()

            # Close the pieces that are finished, keeping their results.
            if closing.any():
                good = closing & ~undefined
                accurate = good & (converged | (error <= ACCEPTABLE * magnitude))
                values[accurate] = value[accurate]
                errors[good] = error[good]
                magnitudes[good] = magnitude[good]
                open_ &= ~closing
                leaving = closing[work.piece]
                if samples:
                    kept.append(work.take(leaving))
                candidate = candidate[~leaving[candidate]]
                candidate = np.cumsum(~leaving)[candidate] - 1  # new indices
                work = work.take(~leaving)
            if not open_.any():
                break

            # Cut the intervals with more than their share of the error: graded
            # if they are very few, into eight if few, else in half; the worst
            # first if the budget runs out.
            chosen = candidate
            parts = PARTS if chosen.size <= FEW else 2
            room = max(1, (MAX_NEW_INTERVALS - evaluated) // parts)
            if chosen.size > room:
                worst = np.argpartition(-work.estimates[chosen, 1], room - 1)[:room]
                chosen = np.sort(chosen[worst])
            parents = work.ends[chosen]
            if chosen.size <= GRADED:
                cuts = [_graded(ends) for ends in parents]
                piece_of = np.repeat(work.piece[chosen], [c.size - 1 for c in cuts])
                ends = np.column_stack(
                    [
                        np.concatenate([c[:-1] for c in cuts]),
                        np.concatenate([c[1:] for c in cuts]),
                    ]
                )
            else:
                grid = parents @ _CUTS[parts]  # (len(chosen), parts + 1)
                grid[:, 0], grid[:, -1] = parents[:, 0], parents[:, 1]  # exactly
                piece_of = np.repeat(work.piece[chosen], parts)
                ends = np.column_stack([grid[:, :-1].ravel(), grid[:, 1:].ravel()])
            children = _evaluate(func, ends, piece_of, undefined, samples)
            evaluated += children.piece.size
            remaining = np.ones(work.piece.size, dtype=bool)
            remaining[chosen] = False
            work = _concatenate(work.take(remaining), children)
    return _result(values, errors, magnitudes, kept if samples else None)


def _totals(
    work: _Intervals, pieces: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return each piece's total value, error and integral of ``|f|``."""
    return (
        np.bincount(work.piece, work.estimates[:, 0], pieces),
        np.bincount(work.piece, work.estimates[:, 1], pieces),
        np.bincount(work.piece, work.estimates[:, 2], pieces),
    )


def _result(
    values: NDArray[np.float64],
    errors: NDArray[np.float64],
    magnitudes: NDArray[np.float64],
    kept: list[_Intervals] | None,
) -> Quadrature:
    """Assemble a :class:`Quadrature` (values that are not finite become NaN)."""
    values = np.where(np.isfinite(values), values, np.nan)
    if kept is None:
        empty = np.zeros(0)
        return Quadrature(
            values, errors, magnitudes, empty, empty, np.zeros(0, np.intp),
            np.zeros((0, 21)),
        )  # fmt: skip
    final = _concatenate(*kept)
    return Quadrature(
        values,
        errors,
        magnitudes,
        final.ends[:, 0].copy(),
        final.ends[:, 1].copy(),
        final.piece,
        final.samples,
    )


def _evaluate(
    func: Integrand,
    ends: NDArray[np.float64],
    piece: NDArray[np.intp],
    undefined: NDArray[np.bool_],
    keep: bool,
) -> _Intervals:
    """Apply the Gauss-Kronrod rule to every interval, with one call of ``func``.

    Args:
        func: The integrand.
        ends: ``(m, 2)``: the intervals' left and right ends.
        piece: Their pieces.
        undefined: Per piece, whether the function is undefined on part of
            it; set here for the pieces of intervals with two or more
            undefined nodes (counting the probes near the ends).
        keep: Keep the values at the nodes (otherwise an empty array per
            interval, so that refining does not copy them round after round).

    Returns:
        The intervals with their estimates. An interval with an undefined
        node or probe has value 0 and an infinite error.
    """
    points = _place(ends, _ENDS_TO_POINTS)
    values = np.asarray(func(points.ravel()), dtype=np.float64).reshape(points.shape)
    f = values[:, 1:-1]
    probes = values[:, [0, -1]]
    # A probe that rounds onto its end (a very narrow interval away from 0)
    # is the end: it tells nothing, and may be a singularity at a limit.
    inside = points[:, [0, -1]] != ends
    finite = np.isfinite(f)
    probed = np.isfinite(probes) | ~inside
    bad = None
    if not (finite.all() and probed.all()):
        missing = np.count_nonzero(~finite, axis=1) + np.count_nonzero(~probed, axis=1)
        undefined[piece[missing >= 2]] = True
        bad = missing > 0
        f = np.where(finite, f, 0.0)
    half = ends @ _ENDS_TO_HALF
    sums = f @ _BOTH_RULES
    kronrod = sums[:, 0]
    estimates = np.empty((ends.shape[0], 3))
    estimates[:, 0] = kronrod * half
    estimates[:, 2] = (np.abs(f) @ KRONROD) * half
    error = np.abs(kronrod - sums[:, 1]) * half
    spread = (np.abs(f - 0.5 * kronrod[:, np.newaxis]) @ KRONROD) * half  # ∫|f - mean|
    # QUADPACK's estimate: close agreement means the Kronrod value is far
    # better than the difference suggests; never below rounding (fmax also
    # replaces the 0 / 0 of a constant integrand).
    error = spread * np.minimum(1.0, (200.0 * error / spread) ** 1.5)
    # The tail bound (see TAIL_FACTOR) only where the interval can still be
    # cut: at the floating-point limit next to a singularity, node positions
    # are rounded, which it would read as roughness it cannot refine away.
    tail = f @ _TO_TAIL
    bound = TAIL_FACTOR * half * np.hypot(tail[:, 0], tail[:, 1])
    error = np.where(_cuttable(ends), np.fmax(error, bound), error)
    # The probes against the nodes' polynomial extrapolated to them bound
    # what the slivers beyond the outermost nodes hide (see PROBE).
    hidden = np.abs(probes - f @ _TO_PROBES)
    hidden = np.where(inside & np.isfinite(hidden), hidden, 0.0)
    hidden = hidden.sum(axis=1) * (_SLIVER * half)
    estimates[:, 1] = np.fmax(error, _ROUNDING * estimates[:, 2]) + hidden
    if bad is not None:
        estimates[bad] = (0.0, np.inf, 0.0)
    return _Intervals(ends, piece, estimates, f if keep else f[:, :0])


def nodes_of(lo: NDArray[np.float64], hi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the 21 nodes of each interval, exactly as :func:`integrate` places them.

    Args:
        lo: The intervals' left ends.
        hi: Their right ends.

    Returns:
        ``(len(lo), 21)``: the nodes, matching :attr:`Quadrature.samples`.
    """
    return _place(np.column_stack([lo, hi]), _ENDS_TO_NODES)


def _graded(ends: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the cuts of one interval into graded parts (see :data:`GRADED`).

    Cuts that would leave a part too narrow for its nodes to be distinct
    (next to an end away from zero) are left out.

    Args:
        ends: The interval's ``(lo, hi)``.

    Returns:
        The cut points, increasing, starting at ``lo`` and ending at ``hi``.
    """
    lo, hi = float(ends[0]), float(ends[1])
    width = hi - lo
    smallest = max(_RESOLUTION * max(abs(lo), abs(hi)), _TINY)
    fractions = _GRADED[np.minimum(_GRADED, 1.0 - _GRADED) * width > smallest]
    inner = lo * (1.0 - fractions) + hi * fractions
    return np.concatenate([[lo], inner, [hi]])


def _cuttable(ends: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return whether intervals are wide enough to be cut into parts.

    Args:
        ends: ``(m, 2)``: the intervals' ends.

    Returns:
        For each, whether its width is above :data:`_NARROWEST` relative to
        its coordinates (and :data:`_TINY`).
    """
    width = ends[:, 1] - ends[:, 0]
    return width > np.maximum(_NARROWEST * np.abs(ends).max(axis=1), _TINY)


def _concatenate(*parts: _Intervals) -> _Intervals:
    """Join sets of intervals."""
    return _Intervals(
        np.concatenate([p.ends for p in parts]),
        np.concatenate([p.piece for p in parts]),
        np.concatenate([p.estimates for p in parts]),
        np.concatenate([p.samples for p in parts]),
    )
