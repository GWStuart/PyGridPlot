"""Grid lines of logarithmic axes.

On a log axis equal distances on screen are equal *ratios*, so its grid is
not one evenly spaced set of lines. It is built in three tiers, each fading
in as zooming makes room for it:

1. **Decades** (``…, 0.1, 1, 10, 100, …``). They are evenly spaced in the
   exponent, so the linear grid machinery (:mod:`pygridplot.core.grid.spacing` and
   :mod:`pygridplot.core.grid.ticks`) is simply applied to the exponent. When
   decades crowd together, only every 2nd, 5th, 10th, … decade is a major
   line, cross-faded exactly like the spacings of a linear axis.
2. **Integer multiples** ``2·10ⁿ, …, 9·10ⁿ``, the familiar lines of log
   paper. They fade in together as minor lines once the most crowded pair
   (9 and 10) is far enough apart. Each one then becomes a labelled major
   line on its own once the gap to its neighbours is wide enough: 2 and 5
   first, then 3, 4, 6 and so on.
3. **Finer subdivisions** between consecutive integers (``1.1, 1.2, …``).
   Over short distances a log axis is nearly linear, so each subdivision
   takes the 1-2-5 level that a linear axis with the *local* scale would
   use, cross-faded near switching points just as a linear axis cross-fades
   while zooming. Every such step divides 1, so subdivisions always line up
   with the integers, and the local scale changes smoothly (even across
   integers and decades), so a deeply zoomed log axis looks just like a
   linear one.

Tiers 2 and 3 only appear while every decade is a major line. Every opacity
is a continuous function of the zoom (or switches outright in ``"snap"``
mode), so lines and labels fade in and out instead of popping, and all
positions follow the view rigidly while panning.

:func:`compute_log_ticks` builds the lines and returns them with their tiers
as :class:`LogTicks`, which :mod:`pygridplot.core.grid.loglabels` then labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.coords import Axis
from pygridplot.core.grid.gridlines import AxisLines, linear_axis_lines
from pygridplot.core.grid.nice import (
    NiceStep,
    common_step,
    level_boundary,
    select_level,
)
from pygridplot.core.grid.spacing import SpacingChoice
from pygridplot.core.grid.theme import GridTheme
from pygridplot.core.grid.ticks import (
    MAX_LINES_PER_AXIS,
    AxisTicks,
    GridLevel,
    ticks_for_levels,
)
from pygridplot.core.view import View

INTEGER_MANTISSAS: Final = (2, 3, 4, 5, 6, 7, 8, 9)
"""The integer multiples of each decade that make up the log-paper lines."""

PAPER_DENSITY: Final = 1 / 25
"""When the log-paper lines appear, relative to ``target_major_px``.

They are half faded in once the gap between 9 and 10 (the most crowded pair)
is this fraction of the target major spacing: 4 pixels by default.
"""

_PAPER_GAP: Final = math.log10(10 / 9)
"""The gap between 9 and 10, in decades."""

_INTEGER_GAPS: Final = np.array(
    [
        math.log10(2) if m in (2, 5) else math.log10((m + 1) / m)
        for m in INTEGER_MANTISSAS
    ]
)
"""The gap in decades from each integer line to its nearest neighbour.

2 and 5 are spaced among ``1, 2, 5, 10`` (they become major lines first);
the others among all the integers.
"""

_LN10: Final = math.log(10.0)

_QUANTUM: Final = 1 / 1024
"""Line opacities are rounded to this, so lines group into few batches."""


def major_threshold_px(theme: GridTheme) -> float:
    """Return the gap at which a log-paper line becomes a major line.

    Args:
        theme: Supplies the target spacing and rounding rule.

    Returns:
        ``target_major_px`` for the ``"ceil"`` rule, and the smallest major
        spacing a linear axis uses (``target / √2.5``) for ``"nearest"``.
    """
    target = theme.target_major_px
    return target if theme.spacing_rule == "ceil" else target / math.sqrt(2.5)


def decade_choice(choice: SpacingChoice) -> SpacingChoice:
    """Limit the major spacing of a log axis to whole decades.

    Args:
        choice: A spacing chosen from the axis' scale in pixels per decade.

    Returns:
        ``choice`` itself, or one decade if it asked for less (finer lines
        are handled by the other tiers).
    """
    if choice.fine.level >= 0:
        return choice
    return SpacingChoice(NiceStep(0))


def decade_levels(choice: SpacingChoice, *, show_minor: bool) -> tuple[GridLevel, ...]:
    """Expand a decade spacing choice into grid levels.

    Minor steps follow the usual rule (two ladder steps finer) but never go
    below one decade: for a major step of 2 decades every decade is a minor
    line, and for one decade the log-paper lines take over.

    Args:
        choice: The decade spacing (see :func:`decade_choice`).
        show_minor: Whether minor lines are wanted.

    Returns:
        One level, or two during a cross-fade (finer level first).
    """
    return tuple(
        GridLevel(step, _decade_minor(step) if show_minor else None, weight)
        for step, weight in choice.weighted_steps()
    )


def compute_log_ticks(
    view: View,
    axis: Axis,
    choice: SpacingChoice,
    theme: GridTheme,
    *,
    margin_px: float,
) -> LogTicks:
    """Build every grid line of one logarithmic axis, tier by tier.

    Args:
        view: The view being drawn.
        axis: ``"x"`` or ``"y"`` (it must be a log axis of ``view``).
        choice: The decade spacing (see :func:`decade_choice`).
        theme: The grid's appearance.
        margin_px: Extra distance outside the plane within which lines are
            still generated (labels slide in from there).

    Returns:
        The lines with their opacities, and the tiers they came from (which
        :mod:`pygridplot.core.grid.loglabels` labels).
    """
    context = LogContext.create(view, axis, theme, margin_px)
    levels = decade_levels(choice, show_minor=theme.show_minor)
    decades = ticks_for_levels(view, axis, levels, margin_px=margin_px)
    every_decade = sum(level.weight for level in levels if level.major.level == 0)
    parts = [linear_axis_lines(decades, skip_axis=False)]
    integers = None
    fine: list[FineTier] = []
    if every_decade > 0:
        integers = _integer_tier(context, every_decade)
        parts.append(integers.lines)
        gate = every_decade * integers.paper
        if gate > 0:
            fine = _fine_tier(context, gate)
            parts.extend(part.lines for part in fine)
    lines = AxisLines(
        axis,
        np.concatenate([part.positions for part in parts]),
        np.concatenate([part.total for part in parts]),
        np.concatenate([part.major for part in parts]),
    )
    return LogTicks(context, lines, decades, integers, fine)


@dataclass(frozen=True, slots=True, eq=False)
class LogTicks:
    """All grid lines of one logarithmic axis, and the tiers they came from.

    Attributes:
        context: The axis being drawn.
        lines: Every line with its opacities, ready to paint.
        decades: Tier 1, the decade lines (classified like a linear axis'
            lines, in decades).
        integers: Tier 2, the log-paper lines, if every decade is a major
            line.
        fine: Tier 3, the subdivisions of each decade on screen.
    """

    context: LogContext
    lines: AxisLines
    decades: AxisTicks
    integers: IntegerTier | None
    fine: list[FineTier]


@dataclass(frozen=True, slots=True)
class LogContext:
    """What every tier needs to know about the axis being drawn.

    Attributes:
        view: The view being drawn.
        axis: The log axis.
        theme: The grid's appearance.
        scale: Pixels per decade.
        low: Lowest visible exponent, including the margin.
        high: Highest visible exponent, including the margin.
        band: Half-width of per-line fades in ``log10`` units (0: snap).
        major_px: Gap at which log-paper lines become major lines.
    """

    view: View
    axis: Axis
    theme: GridTheme
    scale: float
    low: float
    high: float
    band: float
    major_px: float

    @classmethod
    def create(
        cls, view: View, axis: Axis, theme: GridTheme, margin_px: float
    ) -> LogContext:
        """Gather the context for one axis.

        Args:
            view: The view being drawn.
            axis: The log axis.
            theme: The grid's appearance.
            margin_px: Margin around the plane, in pixels.

        Returns:
            The context.
        """
        scale = view.axis_scale(axis)
        low, high = view.transformed_bounds.range(axis)
        pad = margin_px / scale
        band = theme.fade_band if theme.transition == "fade" else 0.0
        return cls(
            view,
            axis,
            theme,
            scale,
            low - pad,
            high + pad,
            band,
            major_threshold_px(theme),
        )

    @property
    def decades(self) -> range:
        """The exponents ``n`` of the decades ``[10ⁿ, 10ⁿ⁺¹)`` on screen."""
        return range(math.floor(self.low), math.floor(self.high) + 1)

    def ramp(self, gaps: NDArray[np.float64], threshold: float) -> NDArray[np.float64]:
        """Return how far lines with the given gaps have faded in.

        Args:
            gaps: The gap from each line to its neighbours, in pixels.
            threshold: The gap at which a line is half faded in.

        Returns:
            Opacity weights in ``[0, 1]``: a smoothstep over ``±band`` decades
            of zoom around the threshold, or a hard switch in snap mode.
        """
        with np.errstate(divide="ignore"):
            ratio = np.log10(np.maximum(gaps, 0.0) / threshold)
        if self.band <= 0.0:
            return (ratio >= 0.0).astype(np.float64)
        t = np.clip((ratio + self.band) / (2.0 * self.band), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True, slots=True, eq=False)
class IntegerTier:
    """The log-paper lines ``2·10ⁿ … 9·10ⁿ`` on screen.

    Attributes:
        lines: The lines with their opacities.
        paper: How far the log-paper lines as a whole have faded in.
        exponents: The decade ``n`` of each line.
        mantissas: The integer multiple (2 to 9) of each line.
        gaps_px: The gap from each line to its nearest neighbour, in pixels.
    """

    lines: AxisLines
    paper: float
    exponents: NDArray[np.int64]
    mantissas: NDArray[np.int64]
    gaps_px: NDArray[np.float64]


def _integer_tier(context: LogContext, every_decade: float) -> IntegerTier:
    """Build the log-paper lines of every decade on screen.

    Args:
        context: The axis being drawn.
        every_decade: The weight of the level that makes every decade a major
            line (the tier fades with it).

    Returns:
        The tier.
    """
    exponents = np.arange(context.decades.start, context.decades.stop)
    count = len(INTEGER_MANTISSAS)
    offsets = np.log10(np.array(INTEGER_MANTISSAS, dtype=np.float64))
    u = (exponents[:, None] + offsets[None, :]).ravel()
    visible = (u >= context.low) & (u <= context.high)
    mantissas = np.tile(np.array(INTEGER_MANTISSAS), exponents.size)[visible]
    gaps = np.tile(_INTEGER_GAPS, exponents.size)[visible] * context.scale
    theme = context.theme
    paper_gap = np.array([context.scale * _PAPER_GAP])
    paper = float(context.ramp(paper_gap, theme.target_major_px * PAPER_DENSITY)[0])
    major = context.ramp(gaps, context.major_px) * every_decade
    total = np.maximum(major, paper * every_decade) if theme.show_minor else major
    positions = context.view.axis_to_screen(context.axis, u[visible])
    lines = AxisLines(context.axis, positions, _quantise(total), _quantise(major))
    return IntegerTier(
        lines, paper, np.repeat(exponents, count)[visible], mantissas, gaps
    )


@dataclass(frozen=True, slots=True, eq=False)
class FineTier:
    """The subdivisions between the integer lines of one decade on screen.

    Every line chooses its own 1-2-5 level from the scale at its own value
    (see :func:`_fine_tier`), cross-faded between a ``fine`` and a ``coarse``
    level like a linear axis.

    Attributes:
        lines: The lines with their opacities.
        base: The step that :attr:`indices` count, in units of the mantissa.
        indices: Each line's mantissa as a multiple of :attr:`base`.
        exponent: The decade ``n``.
        fine: Each line's finer ladder level.
        weight: The weight of the coarser level (``fine + 1``) of each line.
        major_fine: Whether each line is a major line of its finer level.
        major_coarse: Whether each line is a major line of its coarser level.
        gate: How far the whole tier has faded in.
    """

    lines: AxisLines
    base: NiceStep
    indices: NDArray[np.int64]
    exponent: int
    fine: NDArray[np.int64]
    weight: NDArray[np.float64]
    major_fine: NDArray[np.bool_]
    major_coarse: NDArray[np.bool_]
    gate: float


def _fine_tier(context: LogContext, gate: float) -> list[FineTier]:
    """Build the subdivisions between the integer lines on screen.

    A log axis is nearly linear over short distances, with
    ``scale / (m ln 10)`` pixels per unit of the mantissa ``m``. Each
    subdivision takes the 1-2-5 level a linear axis with that local scale
    would use, so the spacing on screen stays near the target everywhere and
    changes smoothly (even across integers and decades, where the local
    scale is continuous). A line's level depends only on its value and the
    zoom, so panning moves every line rigidly. Wherever the local scale is
    near a switching point, the two levels are cross-faded, exactly as a
    linear axis cross-fades while zooming.

    Args:
        context: The axis being drawn.
        gate: How far the tier has faded in (it needs every decade to be a
            major line and the log-paper lines to be visible).

    Returns:
        One part per decade that has subdivisions on screen.
    """
    parts = []
    budget = MAX_LINES_PER_AXIS
    # Every whole decade on screen has the same pattern of subdivisions (only
    # the partly visible ones at the edges differ), so each pattern is worked
    # out once per frame and shifted to each decade.
    patterns: dict[tuple[float, float], _Pattern | None] = {}
    for n in context.decades:
        low = max(1.0, 10.0 ** (context.low - n))
        high = min(10.0, 10.0 ** (context.high - n))
        if low >= high:
            continue
        key = (low, high)
        if key not in patterns:
            patterns[key] = _decade_pattern(context, key)
        pattern = patterns[key]
        if pattern is None or pattern.candidates > budget:
            continue
        part = _decade_subdivisions(context, n, pattern, gate)
        if part is not None:
            budget -= part.indices.size
            parts.append(part)
    return parts


@dataclass(frozen=True, slots=True, eq=False)
class _Pattern:
    """The subdivisions of a visible range of mantissas, in any decade.

    Attributes:
        candidates: How many multiples of :attr:`base` were considered.
        base: The step that :attr:`indices` count, in units of the mantissa.
        indices: Each line's mantissa as a multiple of ``base``.
        values: Each line's mantissa.
        fine: Each line's finer ladder level.
        weight: The weight of the coarser level of each line.
        major_fine: Whether each line is major at its finer level.
        major_coarse: Whether each line is major at its coarser level.
        total: Each line's opacity before the tier's gate.
        major: Each line's major weight before the gate.
    """

    candidates: int
    base: NiceStep
    indices: NDArray[np.int64]
    values: NDArray[np.float64]
    fine: NDArray[np.int64]
    weight: NDArray[np.float64]
    major_fine: NDArray[np.bool_]
    major_coarse: NDArray[np.bool_]
    total: NDArray[np.float64]
    major: NDArray[np.float64]


def _decade_pattern(
    context: LogContext, mantissas: tuple[float, float]
) -> _Pattern | None:
    """Work out the subdivisions of a range of mantissas within a decade.

    The candidate lines are multiples of the finest step needed in the
    visible part of the decade, so the work stays proportional to what is on
    screen. Nothing here depends on which decade it is: a line's level
    depends only on its mantissa and the zoom.

    Args:
        context: The axis being drawn.
        mantissas: The visible range of the mantissa, within ``[1, 10]``.

    Returns:
        The pattern (only lines with some opacity), or ``None`` if there are
        no subdivisions, or more candidates than any axis may draw.
    """
    theme = context.theme
    # log10 of the raw spacing (in mantissa units) a line at mantissa m
    # would ideally have: offset + log10(m).
    offset = math.log10(theme.target_major_px * _LN10 / context.scale)
    first, last = (offset + math.log10(m) for m in mantissas)
    low = select_level(10.0**first, theme.spacing_rule) - 1
    high = select_level(10.0**last, theme.spacing_rule) + 1
    lowest = low - 2 if theme.show_minor else low  # finest step that is drawn
    if lowest >= 0:
        return None  # every step in use is a whole number: nothing in between
    base = common_step(NiceStep(lowest), NiceStep(lowest + 1))
    per_unit = NiceStep(0).multiple_of(base)
    start = math.ceil(mantissas[0] * per_unit)
    stop = math.floor(mantissas[1] * per_unit)
    candidates = stop - start + 1
    if not 0 < candidates <= MAX_LINES_PER_AXIS:
        return None
    indices = np.arange(start, stop + 1, dtype=np.int64)
    indices = indices[indices % per_unit != 0]  # the integers are tier 2
    values = indices / per_unit
    fine, weight = _local_levels(offset + np.log10(values), low, high, context)
    divisors = np.array(
        [NiceStep(level).multiple_of(base) for level in range(lowest, high + 1)]
    )

    def drawn(levels: NDArray[np.int64]) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        major = indices % divisors[levels - lowest] == 0
        if not theme.show_minor:
            return major, major
        return major, major | (indices % divisors[levels - 2 - lowest] == 0)

    major_fine, any_fine = drawn(fine)
    major_coarse, any_coarse = drawn(fine + 1)
    total = (1.0 - weight) * any_fine + weight * any_coarse
    major = (1.0 - weight) * major_fine + weight * major_coarse
    keep = total > 0
    return _Pattern(
        candidates,
        base,
        indices[keep],
        values[keep],
        fine[keep],
        weight[keep],
        major_fine[keep],
        major_coarse[keep],
        total[keep],
        major[keep],
    )


def _decade_subdivisions(
    context: LogContext, n: int, pattern: _Pattern, gate: float
) -> FineTier | None:
    """Place a pattern of subdivisions in the decade ``[10ⁿ, 10ⁿ⁺¹)``.

    Args:
        context: The axis being drawn.
        n: The decade.
        pattern: Its subdivisions (see :func:`_decade_pattern`).
        gate: How far the tier has faded in.

    Returns:
        The decade's subdivisions, or ``None`` if there are none.
    """
    if pattern.indices.size == 0:
        return None
    u = n + np.log10(pattern.values)
    lines = AxisLines(
        context.axis,
        context.view.axis_to_screen(context.axis, u),
        _quantise(pattern.total * gate),
        _quantise(pattern.major * gate),
    )
    return FineTier(
        lines,
        pattern.base,
        pattern.indices,
        n,
        pattern.fine,
        pattern.weight,
        pattern.major_fine,
        pattern.major_coarse,
        gate,
    )


def _local_levels(
    raw_log: NDArray[np.float64], low: int, high: int, context: LogContext
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Choose the ladder level of each line from its own raw spacing.

    This is :func:`~pygridplot.core.grid.spacing.choose_faded`, vectorised over
    lines whose raw spacings lie within the levels ``low`` to ``high``.

    Args:
        raw_log: ``log10`` of the spacing each line would ideally have.
        low: A level at or below the finest one needed.
        high: A level at or above the coarsest one needed.
        context: The axis being drawn (for the rounding rule and fade band).

    Returns:
        ``(fine, weight)``: each line's finer level and the weight of the
        coarser level ``fine + 1`` (0 outside the fade bands).
    """
    rule = context.theme.spacing_rule
    levels = np.arange(low - 1, high + 1)
    boundaries = np.array([level_boundary(int(level), rule) for level in levels])
    # The level whose interval (boundary(L-1), boundary(L)] holds raw_log.
    chosen = levels[np.clip(np.searchsorted(boundaries, raw_log), 0, levels.size - 1)]
    band = context.band
    if band <= 0.0:
        return chosen, np.zeros(raw_log.shape)
    upper = boundaries[chosen - (low - 1)]
    lower = boundaries[np.maximum(chosen - 1 - (low - 1), 0)]
    rising = raw_log > upper - band  # fading towards the next coarser level
    falling = raw_log <= lower + band  # just left the next finer level
    fine = np.where(falling & ~rising, chosen - 1, chosen)
    edge = np.where(falling & ~rising, lower, upper)
    t = np.clip((raw_log - (edge - band)) / (2.0 * band), 0.0, 1.0)
    weight = np.where(rising | falling, t * t * (3.0 - 2.0 * t), 0.0)
    return fine, weight


def _quantise(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Round opacities to :data:`_QUANTUM` so lines group into few batches.

    Args:
        values: Opacities in ``[0, 1]``.

    Returns:
        The rounded opacities.
    """
    return np.round(values / _QUANTUM) * _QUANTUM


def _decade_minor(step: NiceStep) -> NiceStep | None:
    """Return the minor step of a decade spacing.

    Args:
        step: The major step, in decades (at least one).

    Returns:
        Two ladder steps finer, but at least one decade; ``None`` for a major
        step of one decade.
    """
    if step.level >= 2:
        return NiceStep(step.level - 2)
    if step.level == 1:
        return NiceStep(0)
    return None
