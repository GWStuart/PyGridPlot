"""Reusing plots between frames, exactly.

Drawing is immediate mode: every frame, every plot is sampled and drawn from
scratch. Yet most frames show the same plots in the same view as the frame
before (nothing moved), or change only some of them (one point is dragged).
:class:`PlotCache` reuses the work for every plot that has not changed,
without asking anything of the caller and without ever showing a stale
plot.

The difficulty is knowing that a plot has not changed. A function passed
in a frame may be a new lambda with the same meaning, or the same function
whose result has changed because it reads a variable that changed
(``lambda x: a.y * np.sin(x)``). Neither the function's identity nor its
code says whether it would draw the same thing, so neither is used.

Instead, the cache relies on the samplers being **deterministic**: given
the same view and settings, a sampler asks for the function's values at the
same points, in the same order, every time, *as long as the values it gets
back are the same*. So while a plot is built, every value the function
returns is recorded (:class:`Recorder`). In the next frame, the same call
(matched by its position among the frame's plotting calls) is reused only if
its kind, view, style and other inputs are equal, and the function, called
once on all the recorded points, returns exactly the recorded values (bit
for bit, NaN included). Then sampling again would retrace the same steps
and draw exactly the same thing, so the previous result is that result.

Checking costs one vectorised call over a few thousand points, far less
than sampling and rasterising again. A function that changes every frame
(an animation) is caught by a first check of a few points
(:data:`PROBE_POINTS`), so the miss costs almost nothing.

The one assumption is that a function is *pointwise*: its value at ``x``
does not depend on which other points it is evaluated with. Functions that
violate it (normalising by the maximum of their input, say) are not well
defined as plots anyway; at worst they are drawn again every frame.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.plot.evaluation import evaluate_quietly, evaluate_xy_quietly

PROBE_POINTS: Final = 32
"""Recorded points re-evaluated first, to reject a changed function cheaply."""

MAX_ENTRIES: Final = 512
"""Most plotting calls remembered per frame; beyond, calls are not cached.

It also bounds memory if a program never starts a new frame."""


class Recorder:
    """A user function, evaluated robustly, that remembers every evaluation.

    Builders call it in place of the user's function. It evaluates like
    :func:`~pygridplot.core.plot.evaluation.evaluate_quietly` (so the
    builders' own robust evaluation passes its results straight through) and
    keeps a copy of the inputs and results, for :meth:`unchanged`.

    Args:
        func: The user's function.
        arity: 1 for functions of ``x``, 2 for functions of ``(x, y)``.
    """

    __slots__ = ("_calls", "arity", "func")

    def __init__(self, func: Callable[..., Any], arity: int = 1) -> None:
        self.func = func
        self.arity = arity
        self._calls: list[
            tuple[tuple[NDArray[np.float64], ...], NDArray[np.float64]]
        ] = []

    def __call__(self, *args: Any) -> NDArray[np.float64]:
        """Evaluate the function and record the call.

        Args:
            *args: Arrays of argument values (one per argument), or numbers.

        Returns:
            The values, NaN where undefined.
        """
        arrays = tuple(np.array(a, dtype=np.float64, ndmin=1) for a in args)
        values = self._evaluate(arrays)
        self._calls.append((arrays, values.copy()))
        return values

    def unchanged(self) -> bool:
        """Return whether the function still gives every recorded value.

        A few points spread over the record are checked first, then all.

        Returns:
            ``True`` if every value is bit for bit what was recorded.
        """
        if not self._calls:
            return True
        inputs = tuple(
            np.concatenate([call[0][k] for call in self._calls])
            for k in range(self.arity)
        )
        recorded = np.concatenate([call[1] for call in self._calls])
        n = recorded.shape[0]
        probe = np.unique(np.linspace(0, n - 1, min(PROBE_POINTS, n)).astype(np.intp))
        if not _identical(
            self._quiet(tuple(a[probe] for a in inputs)), recorded[probe]
        ):
            return False
        return _identical(self._quiet(inputs), recorded)

    @property
    def evaluations(self) -> int:
        """How many points have been recorded."""
        return sum(call[1].shape[0] for call in self._calls)

    def _evaluate(self, arrays: tuple[NDArray[np.float64], ...]) -> NDArray[np.float64]:
        if self.arity == 1:
            return evaluate_quietly(self.func, arrays[0])
        return evaluate_xy_quietly(self.func, arrays[0], arrays[1])

    def _quiet(self, arrays: tuple[NDArray[np.float64], ...]) -> NDArray[np.float64]:
        """Evaluate with NumPy's warnings silenced, as the samplers do."""
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._evaluate(arrays)


def _identical(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    """Return whether two float arrays are equal bit for bit (NaN equals NaN).

    The sign of zero counts too: a sampler may branch on it.
    """
    if a.shape != b.shape:
        return False
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    return bool(np.all(same & (np.signbit(a) == np.signbit(b))))


def same_inputs(a: object, b: object) -> bool:
    """Return whether two cache keys are equal, including arrays inside them.

    Args:
        a: One key (nested tuples of numbers, strings, arrays and frozen
            dataclasses).
        b: The other.

    Returns:
        Whether they are exactly equal (arrays elementwise, NaN equal NaN).
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        if a.dtype.kind == "f" and b.dtype.kind == "f":
            return _identical(a, b)
        return a.shape == b.shape and bool(np.array_equal(a, b))
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(
            same_inputs(x, y) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True  # NaN
    try:
        return bool(type(a) is type(b) and a == b)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class CacheStats:
    """How often plotting calls were reused.

    Attributes:
        hits: Calls whose previous result was reused.
        misses: Calls that were built again.
    """

    hits: int = 0
    misses: int = 0


@dataclass(slots=True, eq=False)
class _Entry:
    """One plotting call's result, and how to tell whether it still holds."""

    kind: str
    key: object
    recorders: list[Recorder]
    result: Any


@dataclass(slots=True)
class PlotCache:
    """Remembers each plotting call of a frame, to reuse it in the next.

    Calls are matched by their position among the frame's calls (the third
    plotting call with the third), like the palette's colours. A program
    that skips a plot in some frame loses the reuse of the calls after it
    for that frame only.

    Attributes:
        enabled: Whether results are reused at all.
    """

    enabled: bool = True
    _previous: list[_Entry] = field(default_factory=list)
    _current: list[_Entry] = field(default_factory=list)
    _hits: int = 0
    _misses: int = 0

    def begin_frame(self) -> None:
        """Start a new frame: this frame's calls become the ones to match."""
        self._previous, self._current = self._current, []

    def clear(self) -> None:
        """Forget every remembered result and the counts."""
        self._previous, self._current = [], []
        self._hits = self._misses = 0

    @property
    def stats(self) -> CacheStats:
        """Hits and misses since the cache was created or cleared."""
        return CacheStats(self._hits, self._misses)

    def build(
        self,
        kind: str,
        key: object,
        functions: Sequence[object],
        make: Callable[[list[Any]], Any],
        *,
        arity: int = 1,
    ) -> Any:
        """Return a plotting call's result, reused if nothing changed.

        Args:
            kind: The kind of plot (``"plot"``, ``"implicit"``, ...).
            key: Everything else the result depends on, exactly: the view's
                fingerprint, styles, colours, ranges, data arrays (copied).
            functions: The call's user functions; anything not callable (a
                constant curve component) is passed to ``make`` unchanged.
            make: Builds the result from the functions (wrapped so that
                their evaluations are recorded).
            arity: The number of arguments of the functions.

        Returns:
            The result: the previous frame's own object when reused.
        """
        if not self.enabled:
            return make(list(functions))
        # Constant arguments (a number in place of a curve component) are
        # inputs like any other.
        key = (key, tuple(f for f in functions if not callable(f)))
        slot = len(self._current)
        if slot >= MAX_ENTRIES:
            return make(list(functions))
        previous = self._previous[slot] if slot < len(self._previous) else None
        if (
            previous is not None
            and previous.kind == kind
            and same_inputs(previous.key, key)
            and _same_functions(previous.recorders, functions)
            and all(recorder.unchanged() for recorder in previous.recorders)
        ):
            self._hits += 1
            self._current.append(previous)
            return previous.result
        recorders = [Recorder(f, arity) for f in functions if callable(f)]
        wrapped = iter(recorders)
        arguments = [next(wrapped) if callable(f) else f for f in functions]
        result = make(arguments)
        self._misses += 1
        self._current.append(_Entry(kind, key, recorders, result))
        return result


def _same_functions(recorders: list[Recorder], functions: Sequence[object]) -> bool:
    """Point the remembered recorders at this frame's functions, if they match.

    The recorders are re-checked against the *current* functions: a new
    lambda that computes the same values reuses the old result. The number
    of functions (and which arguments are constants) must match.
    """
    callables = [f for f in functions if callable(f)]
    if len(callables) != len(recorders):
        return False
    for recorder, func in zip(recorders, callables, strict=True):
        recorder.func = func
    return True
