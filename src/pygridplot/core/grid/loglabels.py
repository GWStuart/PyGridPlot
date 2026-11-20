"""Tick labels for logarithmic axes.

The lines of a log axis come in three tiers (see
:mod:`pygridplot.core.grid.logticks`), and so do its labels:

* **Decades** are thinned (every 2nd, 5th, 10th, … decade) and cross-faded
  exactly like the labels of a linear axis, by reusing
  :mod:`pygridplot.core.grid.ticklabels` on the exponents.
* **Log-paper lines** are labelled once they are major lines, and fade out
  as the gap to their neighbours gets too small for the widest label.
* **Subdivisions** are thinned like a linear axis' labels, but each label
  is judged by the scale *at its own position*, because the density of a
  log axis changes along it.

Labels choose their notation one by one
(:func:`pygridplot.core.grid.formatting.format_log_value`), so their text
never changes while panning. :func:`log_label_candidates` is the entry
point.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.grid.formatting import NumberFormat, format_log_value
from pygridplot.core.grid.layout import LabelCandidate
from pygridplot.core.grid.logticks import FineTier, IntegerTier, LogContext, LogTicks
from pygridplot.core.grid.nice import NiceStep
from pygridplot.core.grid.ticklabels import (
    MIN_LABEL_GAP_PX,
    combine_level_opacities,
    label_length,
    level_label_opacity,
    next_label_step,
    thinning_opacity,
)
from pygridplot.core.grid.ticks import AxisTicks
from pygridplot.core.primitives import (
    FULL_OPACITY,
    FontSpec,
    LabelText,
    TextMeasurer,
    TextSize,
)

_LN10: Final = math.log(10.0)


def log_label_candidates(
    ticks: LogTicks, measurer: TextMeasurer, font: FontSpec
) -> list[LabelCandidate]:
    """Choose the labels of a log axis and how opaque each one is.

    Args:
        ticks: The axis' lines and tiers.
        measurer: Reports label sizes.
        font: The label font.

    Returns:
        Label candidates in increasing position order.
    """
    return _labels(
        ticks.context, ticks.decades, ticks.integers, ticks.fine, measurer, font
    )


@dataclass(slots=True)
class _Label:
    """A label one of the tiers would like to draw.

    Attributes:
        position: Screen coordinate of its tick.
        text: The formatted label.
        size: Its measured size.
        opacity: Its opacity, once known.
    """

    position: float
    text: LabelText
    size: TextSize
    opacity: float = 0.0


_LabelMaker: TypeAlias = Callable[[float, Decimal], _Label]
"""Builds a label from its tick position and exact value."""


def _labels(
    context: LogContext,
    decades: AxisTicks,
    integers: IntegerTier | None,
    fine: list[FineTier],
    measurer: TextMeasurer,
    font: FontSpec,
) -> list[LabelCandidate]:
    """Choose the labels of a log axis and how opaque each one is.

    Args:
        context: The axis being drawn.
        decades: The decade lines.
        integers: The log-paper lines, if shown.
        fine: The subdivisions, if shown.
        measurer: Reports label sizes.
        font: The label font.

    Returns:
        Label candidates in increasing position order.
    """
    fmt = context.theme.number_format

    def make(position: float, value: Decimal) -> _Label:
        text = _format(value, fmt)
        return _Label(position, text, measurer.measure_text(text, font))

    decade_labels = _decade_labels(decades, make)
    integer_labels = []
    if integers is not None:
        for i in np.flatnonzero(integers.lines.major > 0).tolist():
            value = Decimal(int(integers.mantissas[i])).scaleb(
                int(integers.exponents[i])
            )
            integer_labels.append((i, make(float(integers.lines.positions[i]), value)))
    fine_labels = [_fine_labels(part, make) for part in fine]

    every = [
        *decade_labels.values(),
        *(label for _, label in integer_labels),
        *(label for labels in fine_labels for label in labels.values()),
    ]
    entries = [(label.position, label.size) for label in every]
    length = label_length(context.axis, entries, context.view.rect)
    min_spacing = length + MIN_LABEL_GAP_PX
    comfortable = length + max(context.theme.label_min_separation, MIN_LABEL_GAP_PX)

    _apply(decade_labels, _decade_opacity(decades, context, min_spacing, comfortable))
    if integers is not None:
        for i, label in integer_labels:
            crowding = thinning_opacity(
                float(integers.gaps_px[i]), min_spacing, comfortable
            )
            label.opacity = float(integers.lines.major[i]) * crowding
    for part, labels in zip(fine, fine_labels, strict=True):
        if labels:
            opacity = _fine_label_opacity(part, context.scale, min_spacing, comfortable)
            _apply(labels, opacity)

    candidates = [
        LabelCandidate(label.position, label.text, label.size, _clean(label.opacity))
        for label in every
        if label.opacity > 0
    ]
    candidates.sort(key=lambda candidate: candidate.position)
    return candidates


def _decade_labels(decades: AxisTicks, make: _LabelMaker) -> dict[int, _Label]:
    """Create a label for every decade line that is major in some level.

    Args:
        decades: The decade lines.
        make: Builds a label from a position and a value.

    Returns:
        Labels keyed by candidate line number.
    """
    majors = np.zeros(len(decades), dtype=bool)
    for level in decades.levels:
        majors |= decades.indices % level.major.multiple_of(decades.base) == 0
    labels = {}
    step = round(decades.base.value)  # decade steps are whole numbers
    for i in np.flatnonzero(majors).tolist():
        exponent = int(decades.indices[i]) * step
        labels[i] = make(float(decades.positions[i]), Decimal(1).scaleb(exponent))
    return labels


def _decade_opacity(
    decades: AxisTicks, context: LogContext, min_spacing: float, comfortable: float
) -> NDArray[np.float64]:
    """Return the label opacity of every decade line.

    Decade labels are thinned out (every 2nd, 5th, 10th, … decade) and
    cross-faded exactly like the labels of a linear axis.

    Args:
        decades: The decade lines.
        context: The axis being drawn.
        min_spacing: Label spacing in pixels below which labels would collide.
        comfortable: Label spacing in pixels from which labels are opaque.

    Returns:
        Opacity per decade line.
    """
    if len(decades) == 0:
        return np.zeros(0)
    per_level = [
        level_label_opacity(
            decades,
            level.major,
            context.scale,
            min_spacing,
            comfortable,
            skip_zero=False,
        )
        for level in decades.levels
    ]
    return combine_level_opacities(decades, per_level)


def _fine_labels(fine: FineTier, make: _LabelMaker) -> dict[int, _Label]:
    """Create a label for every subdivision that is major in one of its levels.

    Args:
        fine: The subdivisions.
        make: Builds a label from a position and a value.

    Returns:
        Labels keyed by line number.
    """
    base = fine.base
    labels = {}
    for i in np.flatnonzero(fine.major_fine | fine.major_coarse).tolist():
        units = int(fine.indices[i]) * base.mantissa
        value = Decimal(units).scaleb(base.exponent + fine.exponent)
        labels[i] = make(float(fine.lines.positions[i]), value)
    return labels


def _fine_label_opacity(
    fine: FineTier, scale: float, min_spacing: float, comfortable: float
) -> NDArray[np.float64]:
    """Return the label opacity of each subdivision.

    Within each of a line's two levels, labels are thinned like a linear
    axis' labels (keeping every 2nd, 5th, 10th, … label, see
    :func:`~pygridplot.core.grid.ticklabels.next_label_step`), judged by the scale
    at the label's own position. A label's own labelling step is the
    coarsest of the nested steps it is a multiple of, and it is shown once
    that step is wide enough on screen. The nested steps divide each other,
    so the labels kept never crowd the ones in between. The two levels are
    then combined like a linear axis' cross-fade: shared labels stay, the
    others fade sequentially (see :func:`~pygridplot.core.grid.ticklabels.label_fade`).

    Args:
        fine: The subdivisions.
        scale: Pixels per decade.
        min_spacing: Label spacing in pixels below which labels would collide.
        comfortable: Label spacing in pixels from which labels are opaque.

    Returns:
        Opacity per subdivision line (0 for lines without a label).
    """
    per_unit = NiceStep(0).multiple_of(fine.base)
    local = scale / (_LN10 * (fine.indices / per_unit))  # px per mantissa unit
    shown = []
    for levels, major in (
        (fine.fine, fine.major_fine),
        (fine.fine + 1, fine.major_coarse),
    ):
        own = np.zeros(fine.indices.shape)
        for level in np.unique(levels[major]).tolist():
            step = NiceStep(level)
            rows = major & (levels == level)
            own[rows] = step.value
            step = next_label_step(step)
            while step.level < 0:  # coarser steps are whole numbers
                own[rows & (fine.indices % step.multiple_of(fine.base) == 0)] = (
                    step.value
                )
                step = next_label_step(step)
        spacing = own * local
        if comfortable <= min_spacing:
            fade = (spacing >= min_spacing).astype(np.float64)
        else:
            t = np.clip((spacing - min_spacing) / (comfortable - min_spacing), 0.0, 1.0)
            fade = t * t * (3.0 - 2.0 * t)
        shown.append(np.where(major, fade, 0.0))
    fine_shown, coarse_shown = shown
    weight = fine.weight
    faded = _label_fade(1.0 - weight) * fine_shown + _label_fade(weight) * coarse_shown
    shared = (fine_shown > 0) & (coarse_shown > 0)
    opacity = np.where(shared, np.maximum(fine_shown, coarse_shown), faded)
    return opacity * fine.gate


def _label_fade(weight: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vectorised :func:`~pygridplot.core.grid.ticklabels.label_fade`.

    Args:
        weight: Level weights in ``[0, 1]``.

    Returns:
        ``clamp(2 * weight - 1, 0, 1)``.
    """
    return np.clip(2.0 * weight - 1.0, 0.0, 1.0)


def _apply(labels: dict[int, _Label], opacity: NDArray[np.float64]) -> None:
    """Set the opacity of labels keyed by candidate line number.

    Args:
        labels: The labels.
        opacity: Opacity per candidate line.
    """
    for i, label in labels.items():
        label.opacity = float(opacity[i])


def _clean(opacity: float) -> float:
    """Clamp an opacity to ``(0, 1]``, absorbing float round-off near 1.

    Args:
        opacity: A positive opacity.

    Returns:
        The opacity, or exactly 1 if it is within round-off of it.
    """
    return 1.0 if opacity > FULL_OPACITY else min(opacity, 1.0)


@functools.lru_cache(maxsize=4096)
def _format(value: Decimal, fmt: NumberFormat) -> LabelText:
    """Format a log-axis label, with caching.

    Args:
        value: The exact value.
        fmt: Formatting options.

    Returns:
        The formatted label.
    """
    return format_log_value(value, fmt)
