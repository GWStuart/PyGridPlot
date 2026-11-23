"""Sequences: which terms to draw, and how faded.

:meth:`~pygridplot.Grid.sequence` draws the terms ``aₙ`` of a sequence
at the integers ``n``. A finite sequence (an array) is drawn in full. An
infinite one, a function of ``n``, has a term at every integer, and a view
zoomed far out (or a log x axis, where the integers crowd together towards
the right) can show millions of them, far more than there are pixels.
:func:`sequence_lattice` chooses the terms to draw, so that the work stays
bounded and the picture still looks right:

* **Every term is drawn** while neighbouring terms are at least
  :data:`MIN_SPACING_PX` (a quarter of a pixel) apart on screen: closely
  enough to show the envelope of the terms faithfully, as the stems of
  several terms in one pixel column are drawn as one (see
  :func:`~pygridplot.core.plot.builders.sequence_primitives`).
* **Closer than that, terms are thinned** along nested powers of three, the
  way the arrows of vector fields are thinned along powers of two (see
  :mod:`pygridplot.core.plot.fields`). With ``sp(n)`` the distance on
  screen from term ``n`` to the next (``scale × T'(n)``, constant on a
  linear axis, ``scale / (n ln 10)`` on a log one), term ``n`` is drawn if
  it is a multiple of ``step(n) = 3^e(n)``, the power of three that puts
  the drawn terms between a third of the spacing and the spacing apart.
  Powers of three rather than two, because thinning to the even terms
  would make an alternating sequence (``(-1)ⁿ / n``) look one-sided, and
  hide the odd harmonics of a square wave's spectrum; a multiple of
  ``3^e`` is as often odd as even. Only patterns repeating every 3, 9, ...
  terms can be misrepresented, and only below a quarter pixel apart.
* **Nothing pops, and panning is rigid.** The terms at multiples of
  ``step(n)`` but not of ``3 step(n)``, the next to go, fade out over the
  second half of each power of three of zoom
  (:data:`~pygridplot.core.plot.fields.FADE_START`), and the rule depends
  on each term alone, so the same terms are drawn every frame.
* **The terms are found without visiting the others.** ``e(n)`` never
  decreases as ``n`` grows (it is constant on a linear axis), so the terms
  drawn at each level of thinning lie in a stretch starting at the first
  visible term, found by bisection; the multiples of the coarsest step
  cover the rest.

On a linear axis all the fading terms fade together, and are drawn as one
primitive. On a log axis every term fades at its own zoom level, so their
fades are rounded up to eighths (:data:`LOG_FADE_LEVELS`), which keeps the
number of primitives small at the price of steps of 1/8 in the opacity of
single terms less than a quarter pixel apart.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pygridplot.core.plot.fields import FADE_START
from pygridplot.core.view import View

MIN_SPACING_PX: Final = 0.25
"""Terms of an infinite sequence closer than this on screen are thinned.

At most about ``3 / MIN_SPACING_PX`` terms per pixel are evaluated."""

BASE: Final = 3
"""Terms are thinned to multiples of powers of this (see the module notes)."""

LOG_FADE_LEVELS: Final = 8
"""On a log x axis, fades are rounded up to multiples of one over this."""

MAX_INDEX: Final = 2**52
"""Largest index whose term can be placed exactly (floats hold integers up to it)."""

_COARSEST: Final = 34
"""The largest exponent of a step needed: ``3³⁴ > 2⁵²``, so 0 is its only multiple."""


@dataclass(frozen=True, slots=True, eq=False)
class SequenceLattice:
    """The terms of an infinite sequence to draw in a view.

    Attributes:
        n: The indices, increasing.
        fade: How far each term has faded in, in ``(0, 1]``.
    """

    n: NDArray[np.int64]
    fade: NDArray[np.float64]


def check_start(start: object) -> int:
    """Validate the index of a sequence's first term.

    Args:
        start: The index.

    Returns:
        It, as an int.

    Raises:
        ValueError: If it is not a whole number within ``±2⁵²``.
    """
    if (
        isinstance(start, bool)
        or not isinstance(start, (int, np.integer))
        or not -MAX_INDEX <= start <= MAX_INDEX
    ):
        msg = f"start must be a whole number (the first index), got {start!r}"
        raise ValueError(msg)
    return int(start)


def check_baseline(baseline: object) -> float:
    """Validate where a sequence's stems start.

    Args:
        baseline: The y value.

    Returns:
        It, as a float.

    Raises:
        ValueError: If it is not a finite number.
    """
    if (
        isinstance(baseline, bool)
        or not isinstance(baseline, numbers.Real)
        or not math.isfinite(baseline)
    ):
        msg = f"baseline must be a finite number, got {baseline!r}"
        raise ValueError(msg)
    return float(baseline)


def as_terms(terms: ArrayLike) -> NDArray[np.float64]:
    """Convert the terms of a finite sequence to a float array.

    NaN terms are allowed (they are not drawn).

    Args:
        terms: A sequence or 1-D array of real numbers.

    Returns:
        The terms, a new 1-D float array.

    Raises:
        ValueError: If they are not a 1-D sequence of real numbers.
    """
    try:
        array = np.asarray(terms)
    except (TypeError, ValueError):
        array = np.asarray(None)
    if np.iscomplexobj(array):
        msg = "sequence terms must be real: take np.abs(...) or .real of complex ones"
        raise ValueError(msg)
    if array.ndim != 1 or not (
        np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_
    ):
        msg = "terms must be a function of n or a 1-D sequence of numbers"
        raise ValueError(msg)
    return np.array(array, dtype=np.float64)


def integer_arguments(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap a term function so that single indices reach it as Python ints.

    Terms are first asked for all at once, with a float array of whole
    numbers (NumPy arithmetic on floats cannot overflow, as ``2**n`` on
    64-bit integers would). A function that only works one index at a time
    is then called with each index alone (see
    :mod:`pygridplot.core.plot.evaluation`), and receives a Python ``int``,
    so ``math.factorial``, ``math.comb`` and list indexing work.

    Args:
        func: The term ``aₙ`` as a function of ``n``.

    Returns:
        The wrapped function.
    """

    def term(n: Any) -> Any:
        if isinstance(n, float) and n.is_integer():
            return func(int(n))
        return func(n)

    return term


def visible_indices(view: View, start: int, margin: float) -> tuple[int, int] | None:
    """Return the first and last indices ``n >= start`` whose terms can show.

    Args:
        view: The view being drawn.
        start: The first index of the sequence.
        margin: Pixels beyond the plane's left and right edges to cover, so
            that a term just outside still shows its part inside.

    Returns:
        ``(first, last)``, or ``None`` if no index is visible.
    """
    if view.rect.is_empty:
        return None
    scale = view.scale_x
    bounds = view.transformed_bounds
    transform = view.transform_x
    low = float(transform.inverse(bounds.x_min - margin / scale))
    high = float(transform.inverse(bounds.x_max + margin / scale))
    if math.isnan(low) or math.isnan(high):
        return None
    first = max(start, -MAX_INDEX, math.ceil(low) if math.isfinite(low) else -MAX_INDEX)
    last = min(MAX_INDEX, math.floor(high) if math.isfinite(high) else MAX_INDEX)
    if transform.is_log:
        first = max(first, 1)  # only positive indices can be shown
    return (first, last) if first <= last else None


def sequence_lattice(view: View, start: int, margin: float) -> SequenceLattice | None:
    """Return the terms of an infinite sequence to draw (see the module notes).

    Args:
        view: The view being drawn.
        start: The first index of the sequence.
        margin: Pixels beyond the plane's left and right edges to cover.

    Returns:
        The terms, or ``None`` if none are visible.
    """
    visible = visible_indices(view, start, margin)
    if visible is None:
        return None
    first, last = visible
    at_first, at_last = (float(e) for e in _exponents(view, np.array([first, last])))
    # Beyond 3³⁴ no step has a multiple in range but 0, and 3³⁴ keeps it.
    coarsest = int(min(max(at_first, at_last, 0.0), _COARSEST))
    finest = int(min(max(min(at_first, at_last), 0.0), _COARSEST))
    parts = [_multiples(first, last, coarsest, exact=False)]
    if finest < coarsest:  # a log axis: finer levels nearer the first term
        levels = np.arange(finest, coarsest)
        ends = _level_ends(view, first, last, levels)
        for k, end in zip(levels, ends, strict=True):
            parts.append(_multiples(first, int(end), int(k), exact=True))
    n = np.concatenate([part[0] for part in parts])
    found = np.concatenate([part[1] for part in parts])
    exact = np.concatenate([part[2] for part in parts])
    order = np.argsort(n)
    n, found, exact = n[order], found[order], exact[order]
    fade = _fades(view, n, found, exact)
    shown = fade > 0
    return SequenceLattice(n[shown], fade[shown])


def _spacing(view: View, n: NDArray[np.int64]) -> NDArray[np.float64]:
    """The distance on screen from each term to the next, in pixels."""
    derivative = view.transform_x.derivative(n.astype(np.float64))
    return np.asarray(view.scale_x * derivative, dtype=np.float64)


def _exponents(view: View, n: NDArray[np.int64]) -> NDArray[np.float64]:
    """``e(n)``: the exponent of the power of three the terms near ``n`` are thinned to.

    Below 0 (and so step 1, every term) where terms are at least
    :data:`MIN_SPACING_PX` apart.
    """
    with np.errstate(divide="ignore"):
        ratio = MIN_SPACING_PX / (BASE * _spacing(view, n))
        return np.ceil(np.log(ratio) / math.log(BASE))


def _multiples(
    first: int, last: int, level: int, *, exact: bool
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
    """Return the multiples of ``3^level`` in ``[first, last]``.

    Args:
        first: The first index.
        last: The last index.
        level: The exponent.
        exact: Only the multiples that are not also multiples of
            ``3^(level + 1)`` (those of exactly this level).

    Returns:
        The indices, their level, and whether each is of exactly that level
        (not a multiple of the next power).
    """
    step = BASE**level
    low, high = -((-first) // step), last // step  # ceil and floor
    if high < low:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=bool)
    multiples = np.arange(low, high + 1, dtype=np.int64)
    only = multiples % BASE != 0
    if exact:
        multiples, only = multiples[only], only[only]
    return multiples * step, np.full(multiples.shape, level, dtype=np.int64), only


def _level_ends(
    view: View, first: int, last: int, levels: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Return, for each level ``k``, the last index in range with ``e(n) <= k``.

    ``e`` never decreases as ``n`` grows, so the indices with ``e(n) <= k``
    are a stretch starting at ``first``; all levels are bisected at once.

    Args:
        view: The view being drawn.
        first: The first visible index (``e(first) <= k`` for every level).
        last: The last visible index.
        levels: The levels.

    Returns:
        The last index of each level's stretch.
    """
    low = np.full(levels.shape, first, dtype=np.int64)  # e(low) <= k
    high = np.full(levels.shape, last + 1, dtype=np.int64)  # beyond the stretch
    while np.any(high - low > 1):
        middle = low + (high - low) // 2
        inside = _exponents(view, middle) <= levels
        low = np.where(inside, middle, low)
        high = np.where(inside, high, middle)
    return low


def _fades(
    view: View,
    n: NDArray[np.int64],
    level: NDArray[np.int64],
    exact: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return how far each candidate term has faded in.

    A term at a multiple of its own step but not of three steps (the next
    to go when zooming out) fades with how far the zoom is through its power
    of three; every other term is fully drawn.

    Args:
        view: The view being drawn.
        n: The indices.
        level: The level each was found at.
        exact: Whether each is of exactly that level.

    Returns:
        The fades, in ``[0, 1]``.
    """
    spacing = _spacing(view, n)
    exponent = _exponents(view, n)
    fading = (exponent >= 0) & exact & (level == exponent)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        steps = np.log(BASE * spacing / MIN_SPACING_PX) / math.log(BASE) + exponent
    t = np.clip((steps - FADE_START) / (1.0 - FADE_START), 0.0, 1.0)
    fade = np.where(fading, t * t * (3.0 - 2.0 * t), 1.0)
    if view.transform_x.is_log:
        fade = np.ceil(fade * LOG_FADE_LEVELS) / LOG_FADE_LEVELS
    return np.asarray(fade, dtype=np.float64)
