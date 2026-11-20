"""Grid line positions along one axis, computed from exact integer indices.

Line positions are never accumulated (``x += step`` drifts). Every candidate
line is the ``n``-th multiple of a **base step**, its position on screen is
computed directly from ``n`` (see
:meth:`~pygridplot.core.view.View.multiples_to_screen`), and whether it is a
major line, a minor line or not drawn at all is decided by integer
divisibility of ``n``. Float modulo is never used.

While cross-fading, two grid *levels* (major/minor configurations) are drawn
at once. The base step is then the largest step that divides every major and
minor step involved, so a single set of candidate lines can be classified
under both levels.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.coords import Axis
from pygridplot.core.nice import NiceStep, common_step
from pygridplot.core.spacing import SpacingChoice
from pygridplot.core.view import View

MAX_LINES_PER_AXIS: Final = 2000
"""Safety cap on candidate lines per axis.

With the default spacing targets a 4K display needs a few hundred lines. If
the cap is ever exceeded (for example with a tiny ``target_major_px``), minor
lines are dropped first and then the axis is skipped entirely.
"""


class LineClass(enum.IntEnum):
    """How a candidate line is drawn within one grid level."""

    NONE = 0
    """Not part of this level."""

    MINOR = 1
    """A minor line of this level."""

    MAJOR = 2
    """A major line of this level."""


@dataclass(frozen=True, slots=True)
class GridLevel:
    """One complete grid configuration together with its current opacity.

    Attributes:
        major: Spacing of major lines (the ones that get labels).
        minor: Spacing of minor lines, or ``None`` if they are hidden.
        weight: Opacity of this level in ``(0, 1]``. During a cross-fade two
            levels are active and their weights sum to 1.
    """

    major: NiceStep
    minor: NiceStep | None
    weight: float


@dataclass(frozen=True, slots=True, eq=False)
class AxisTicks:
    """All candidate lines along one axis, classified under each active level.

    Attributes:
        axis: ``"x"`` (vertical lines at constant x) or ``"y"``.
        base: Step whose multiples are the candidate lines.
        indices: Integer multipliers ``n`` of :attr:`base`, in increasing
            order.
        positions: Continuous screen coordinate of each candidate line.
        levels: The active grid levels.
        classes: One array per level giving the :class:`LineClass` of each
            candidate line.
    """

    axis: Axis
    base: NiceStep
    indices: NDArray[np.int64]
    positions: NDArray[np.float64]
    levels: tuple[GridLevel, ...]
    classes: tuple[NDArray[np.int8], ...]

    def __len__(self) -> int:
        return int(self.indices.shape[0])


def grid_levels(choice: SpacingChoice, *, show_minor: bool) -> tuple[GridLevel, ...]:
    """Expand a spacing choice into the grid levels that should be drawn.

    Args:
        choice: The (possibly blended) major spacing.
        show_minor: Whether minor lines are wanted.

    Returns:
        One level, or two levels during a cross-fade (finer level first).
    """
    return tuple(
        GridLevel(step, step.minor if show_minor else None, weight)
        for step, weight in choice.weighted_steps()
    )


def base_step(levels: tuple[GridLevel, ...]) -> NiceStep:
    """Return the largest step that divides every step used by ``levels``.

    Args:
        levels: The active grid levels (at least one).

    Returns:
        The common base step for the candidate lines.

    Raises:
        ValueError: If ``levels`` is empty.
    """
    steps = [s for level in levels for s in (level.major, level.minor) if s is not None]
    if not steps:
        msg = "at least one grid level is required"
        raise ValueError(msg)
    base = steps[0]
    for step in steps[1:]:
        base = common_step(base, step)
    return base


def index_range(low: float, high: float, step: float) -> tuple[int, int]:
    """Return the inclusive range of ``n`` with ``low <= n * step <= high``.

    Args:
        low: Lower end of the world interval.
        high: Upper end of the world interval.
        step: Positive spacing in world units.

    Returns:
        ``(first, last)``. ``first > last`` means no multiples fall inside.
    """
    return (math.ceil(low / step), math.floor(high / step))


def classify(
    indices: NDArray[np.int64], base: NiceStep, level: GridLevel
) -> NDArray[np.int8]:
    """Classify candidate lines under one grid level using integer arithmetic.

    Args:
        indices: Multipliers of ``base``.
        base: The base step, which must divide the level's steps.
        level: The grid level.

    Returns:
        The :class:`LineClass` value of each index.
    """
    result = np.zeros(indices.shape, dtype=np.int8)
    if level.minor is not None:
        result[indices % level.minor.multiple_of(base) == 0] = LineClass.MINOR
    result[indices % level.major.multiple_of(base) == 0] = LineClass.MAJOR
    return result


def compute_axis_ticks(
    view: View,
    axis: Axis,
    choice: SpacingChoice,
    *,
    show_minor: bool,
    margin_px: float = 4.0,
    max_lines: int = MAX_LINES_PER_AXIS,
) -> AxisTicks:
    """Compute and classify every candidate grid line along one axis.

    Args:
        view: The view being drawn.
        axis: ``"x"`` for vertical lines, ``"y"`` for horizontal lines.
        choice: The major spacing for this axis.
        show_minor: Whether minor lines are wanted.
        margin_px: Extra distance outside the plane, in pixels, within which
            lines are still generated (so thick lines at the edge stay whole).
        max_lines: Safety cap on the number of candidate lines.

    Returns:
        The classified candidate lines. They may be empty if the plane has no
        area or the cap could not be met.
    """
    levels = grid_levels(choice, show_minor=show_minor)
    low, high = _visible_range(view, axis, margin_px)
    for attempt_levels in (levels, _without_minor(levels)):
        base = base_step(attempt_levels)
        first, last = index_range(low, high, base.value)
        count = last - first + 1
        if count <= max_lines:
            indices = np.arange(first, last + 1, dtype=np.int64)
            positions = view.multiples_to_screen(axis, base.value, indices)
            classes = tuple(classify(indices, base, lv) for lv in attempt_levels)
            return AxisTicks(axis, base, indices, positions, attempt_levels, classes)
    empty = np.zeros(0, dtype=np.int64)
    return AxisTicks(
        axis,
        base,
        empty,
        np.zeros(0),
        levels,
        tuple(np.zeros(0, dtype=np.int8) for _ in levels),
    )


def _visible_range(view: View, axis: Axis, margin_px: float) -> tuple[float, float]:
    """Return the world interval covered by the plane along an axis, plus a margin.

    Args:
        view: The view being drawn.
        axis: ``"x"`` or ``"y"``.
        margin_px: Margin to add on both sides, in pixels.

    Returns:
        ``(low, high)`` in world units.
    """
    low, high = view.bounds.range(axis)
    pad = margin_px / view.axis_scale(axis)
    return (low - pad, high + pad)


def _without_minor(levels: tuple[GridLevel, ...]) -> tuple[GridLevel, ...]:
    """Return ``levels`` with their minor lines removed.

    Args:
        levels: Grid levels.

    Returns:
        The same levels with ``minor`` set to ``None``.
    """
    return tuple(GridLevel(lv.major, None, lv.weight) for lv in levels)
