"""Choosing which tick labels to show, and how opaque each one is.

Every grid level labels its major lines. This module decides which of those
labels are actually shown and with what opacity, so that labels never crowd
each other and never pop in or out while zooming:

* **Thinning:** when labels would be too close, only every 2nd, 5th, 10th, ...
  one is kept (:func:`label_step`). Labels about to be thinned out fade away
  first (:func:`thinning_opacity`).
* **Cross-fades:** while two grid levels are blended, each level's own labels
  fade sequentially (:func:`label_fade`) and shared labels stay opaque.
* **Notation:** each level writes its numbers consistently (see
  :func:`pygridplot.core.grid.formatting.choose_notation`). If the two levels of a
  cross-fade disagree, their labels are faded as separate sets.

The result is a list of :class:`~pygridplot.core.grid.layout.LabelCandidate`
objects, ready to be positioned by :mod:`pygridplot.core.grid.layout`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

import numpy as np

from pygridplot.core.coords import Axis, ScreenRect
from pygridplot.core.grid.formatting import (
    Notation,
    NumberFormat,
    choose_notation,
    format_tick,
    tick_value,
)
from pygridplot.core.grid.layout import LabelCandidate
from pygridplot.core.grid.nice import NiceStep
from pygridplot.core.grid.spacing import smoothstep
from pygridplot.core.grid.theme import GridTheme
from pygridplot.core.grid.ticks import AxisTicks
from pygridplot.core.primitives import (
    FULL_OPACITY,
    FontSpec,
    LabelText,
    TextMeasurer,
    TextSize,
)
from pygridplot.core.view import View

LABEL_MARGIN_PX: Final = 96.0
"""How far outside the plane (in pixels) ticks are still generated.

Labels centred on ticks just outside the plane are still partly visible, so
they need to exist for labels to slide smoothly in from the edges instead of
popping in.
"""

MIN_LABEL_GAP_PX: Final = 4
"""Hard minimum gap between neighbouring labels on one axis.

Labels are fully opaque while their gap is at least the theme's
``label_min_separation``, fade out as the gap shrinks towards this minimum,
and are thinned out once it would be crossed.
"""


def label_candidates(
    ticks: AxisTicks,
    view: View,
    theme: GridTheme,
    measurer: TextMeasurer,
    font: FontSpec,
) -> list[LabelCandidate]:
    """Return the labels one axis would like to draw, with their opacities.

    Each active level labels its major lines, thinned out to every 2nd, 5th,
    10th, ... line when labels would otherwise be too close together (see
    :func:`label_step`). Thinning is continuous: labels the next thinning
    step would remove fade out as they approach that point (see
    :func:`thinning_opacity`). During a cross-fade, labels shared by both
    levels stay opaque, while the others fade *sequentially* (see
    :func:`label_fade`): the outgoing labels are gone before the incoming ones
    appear, so the two sets never crowd each other.

    Args:
        ticks: The axis' classified candidate lines.
        view: The view being drawn (for the axis scale).
        theme: Supplies label settings.
        measurer: Reports label sizes.
        font: The label font.

    Returns:
        Candidate labels in increasing position order. Zero is never included.
    """
    if len(ticks) == 0:
        return []
    largest = _largest_label_value(ticks)
    levels = [
        _level_labels(ticks, level.major, largest, view, theme, measurer, font)
        for level in ticks.levels
    ]
    if len(levels) == 2 and levels[0].notation != levels[1].notation:
        # The levels write numbers differently: let each level's labels fade
        # on their own, so the notation change is cross-faded too.
        return [
            *_candidates(
                ticks, levels[0], label_fade(ticks.levels[0].weight) * levels[0].opacity
            ),
            *_candidates(
                ticks, levels[1], label_fade(ticks.levels[1].weight) * levels[1].opacity
            ),
        ]
    opacity = combine_level_opacities(ticks, [lv.opacity for lv in levels])
    return _candidates(ticks, levels[-1], opacity, fallback=levels[0])


@dataclass(slots=True)
class _LevelLabels:
    """The labels one grid level would draw along an axis.

    Attributes:
        notation: How the level writes its numbers.
        texts: Text and size of each major line's label, by index.
        opacity: Opacity per candidate line (0 where unlabelled).
    """

    notation: Notation
    texts: dict[int, tuple[LabelText, TextSize]]
    opacity: np.ndarray


def _level_labels(
    ticks: AxisTicks,
    major: NiceStep,
    largest: Decimal,
    view: View,
    theme: GridTheme,
    measurer: TextMeasurer,
    font: FontSpec,
) -> _LevelLabels:
    """Format, measure and thin out the labels of one grid level.

    Args:
        ticks: The axis' candidate lines.
        major: The level's major step.
        largest: The largest magnitude among the axis' labels.
        view: The view being drawn (for the axis scale).
        theme: Supplies label settings.
        measurer: Reports label sizes.
        font: The label font.

    Returns:
        The level's labels.
    """
    fmt = theme.number_format
    notation = choose_notation(largest, major, fmt)
    indices = ticks.indices
    major_mask = (indices != 0) & (indices % major.multiple_of(ticks.base) == 0)
    texts: dict[int, tuple[LabelText, TextSize]] = {}
    for n in indices[major_mask].tolist():
        text = _format(n, ticks.base, fmt, notation)
        texts[n] = (text, measurer.measure_text(text, font))
    positions = dict(zip(indices.tolist(), ticks.positions.tolist(), strict=True))
    entries = [(positions[n], size) for n, (_, size) in texts.items()]
    length = label_length(ticks.axis, entries, view.rect)
    comfortable = length + max(theme.label_min_separation, MIN_LABEL_GAP_PX)
    scale = view.axis_scale(ticks.axis)
    opacity = level_label_opacity(
        ticks, major, scale, length + MIN_LABEL_GAP_PX, comfortable
    )
    return _LevelLabels(notation, texts, opacity)


def _candidates(
    ticks: AxisTicks,
    labels: _LevelLabels,
    opacity: np.ndarray,
    fallback: _LevelLabels | None = None,
) -> list[LabelCandidate]:
    """Turn per-line opacities into label candidates.

    Args:
        ticks: The axis' candidate lines.
        labels: Where the label texts come from.
        opacity: Opacity per candidate line.
        fallback: Second source of texts for lines ``labels`` has no text for
            (the other level of a cross-fade, when both write numbers alike).

    Returns:
        One candidate per line with a non-zero opacity.
    """
    candidates = []
    for i in np.flatnonzero(opacity > 0).tolist():
        n = int(ticks.indices[i])
        entry = labels.texts.get(n)
        if entry is None and fallback is not None:
            entry = fallback.texts.get(n)
        if entry is None:
            continue
        text, size = entry
        alpha = float(min(opacity[i], 1.0))
        alpha = 1.0 if alpha > FULL_OPACITY else alpha
        candidates.append(LabelCandidate(float(ticks.positions[i]), text, size, alpha))
    return candidates


def level_label_opacity(
    ticks: AxisTicks,
    major: NiceStep,
    scale: float,
    min_spacing: float,
    comfortable_spacing: float,
    *,
    skip_zero: bool = True,
) -> np.ndarray:
    """Return each candidate line's label opacity within one grid level.

    Args:
        ticks: The axis' candidate lines.
        major: The level's major step.
        scale: Pixels per unit of the steps' values along the axis (per world
            unit on a linear axis).
        min_spacing: Tick spacing in pixels below which labels would collide.
        comfortable_spacing: Tick spacing in pixels from which labels are
            fully opaque.
        skip_zero: Leave index 0 unlabelled (zero gets its own label on a
            linear axis; on a log axis index 0 is a real tick, ``10⁰``).

    Returns:
        Opacity in ``[0, 1]`` per candidate line (0 for unlabelled lines).
    """
    indices = ticks.indices
    step = label_step(major, scale, min_spacing)
    labelled = indices % step.multiple_of(ticks.base) == 0
    if skip_zero:
        labelled &= indices != 0
    survivors = indices % next_label_step(step).multiple_of(ticks.base) == 0
    fading = thinning_opacity(step.value * scale, min_spacing, comfortable_spacing)
    return np.where(labelled, np.where(survivors, 1.0, fading), 0.0)


def combine_level_opacities(
    ticks: AxisTicks, per_level: list[np.ndarray]
) -> np.ndarray:
    """Merge the label opacities of the active levels of a cross-fade.

    Args:
        ticks: The axis' candidate lines (their levels carry the weights).
        per_level: One opacity array per level, as from
            :func:`_level_label_opacity`.

    Returns:
        The final opacity of each candidate line's label.
    """
    if len(per_level) == 1:
        return per_level[0]
    total = np.zeros(len(ticks))
    for level, opacity in zip(ticks.levels, per_level, strict=True):
        total += label_fade(level.weight) * opacity
    stacked = np.stack(per_level)
    shared = np.all(stacked > 0, axis=0)
    total[shared] = np.max(stacked, axis=0)[shared]
    return total


def _largest_label_value(ticks: AxisTicks) -> Decimal:
    """Return the largest magnitude among the axis' major-line labels.

    Args:
        ticks: The axis' candidate lines (finest level first).

    Returns:
        The magnitude as an exact decimal (zero if there are no labels).
    """
    finest = ticks.levels[0].major
    majors = ticks.indices[ticks.indices % finest.multiple_of(ticks.base) == 0]
    if majors.size == 0:
        return Decimal(0)
    return tick_value(int(np.abs(majors).max()), ticks.base)


def label_fade(weight: float) -> float:
    """Return the opacity of a level's own labels from the level's weight.

    The label fades are compressed into one half of the cross-fade each:
    a level's labels are invisible while its weight is below 0.5, then
    become fully opaque by the time its weight reaches 1. At the midpoint
    only labels shared by both levels are shown.

    Args:
        weight: The level's weight in ``[0, 1]``.

    Returns:
        ``clamp(2 * weight - 1, 0, 1)``.
    """
    return min(max(2.0 * weight - 1.0, 0.0), 1.0)


def label_length(
    axis: Axis, entries: list[tuple[float, TextSize]], rect: ScreenRect
) -> float:
    """Return the length the largest label takes up along its axis.

    Labels outside the plane count less the further away they are, fading
    to nothing at :data:`LABEL_MARGIN_PX`. A wide label sliding in from the
    edge therefore raises the requirement gradually, and thinning (see
    :func:`thinning_opacity`) reacts smoothly instead of all at once.

    Args:
        axis: ``"x"`` (labels side by side: widths) or ``"y"`` (labels
            stacked: heights).
        entries: ``(tick position, size)`` of each label.
        rect: The plane's screen rectangle.

    Returns:
        The (weighted) largest label length in pixels.
    """
    low, high = (rect.left, rect.right) if axis == "x" else (rect.top, rect.bottom)
    longest = 0.0
    for position, size in entries:
        length = size.width if axis == "x" else size.height + size.overflow_top
        outside = max(low - position, position - (high - 1), 0.0) - length / 2
        weight = min(max(1.0 - outside / LABEL_MARGIN_PX, 0.0), 1.0)
        longest = max(longest, length * weight)
    return longest


def next_label_step(step: NiceStep) -> NiceStep:
    """Return the next coarser ladder step that is a whole multiple of ``step``.

    Following this repeatedly gives a *nested* sequence of labelling steps,
    for example 5, 10, 20, 100, 200, ... Every label of a coarser step is
    also a label of the finer one, so thinning only ever removes labels and
    never introduces new ones.

    Args:
        step: The current labelling step.

    Returns:
        The next coarser labelling step.
    """
    candidate = step.coarser()
    while not candidate.is_multiple_of(step):
        candidate = candidate.coarser()
    return candidate


def label_step(major: NiceStep, scale: float, min_spacing_px: float) -> NiceStep:
    """Return the finest labelling step that keeps labels ``min_spacing_px`` apart.

    The candidates are ``major`` followed by :func:`next_label_step`
    repeatedly, so labels always sit on major lines and keep "nice" values.
    For example from a major step of 2 they are 2, 10, 20, 100, ...

    Args:
        major: The major step of the level.
        scale: Pixels per world unit along the axis.
        min_spacing_px: Required distance between labelled ticks.

    Returns:
        The labelling step.
    """
    step = major
    for _ in range(24):  # 8 decades: far more than any real label needs
        if step.value * scale >= min_spacing_px:
            return step
        step = next_label_step(step)
    return step


def thinning_opacity(
    spacing_px: float, min_spacing_px: float, comfortable_px: float
) -> float:
    """Return the opacity of labels that the next thinning step would remove.

    As zooming squeezes labels together, the labels that are about to be
    thinned out stay fully opaque until the gap between labels drops below
    the comfortable gap, then fade away. By the time thinning kicks in (at
    ``min_spacing_px``) they are invisible, and zooming back in fades them
    in again.

    Args:
        spacing_px: Current on-screen spacing of the labelled ticks.
        min_spacing_px: Spacing at which the labels would collide.
        comfortable_px: Spacing from which labels are fully opaque.

    Returns:
        Opacity in ``[0, 1]``.
    """
    if comfortable_px <= min_spacing_px:
        return 1.0 if spacing_px >= min_spacing_px else 0.0
    return smoothstep((spacing_px - min_spacing_px) / (comfortable_px - min_spacing_px))


@functools.lru_cache(maxsize=8192)
def _format(
    index: int, base: NiceStep, fmt: NumberFormat, notation: Notation
) -> LabelText:
    """Format a tick label, with caching (formatting is exact but not free).

    Args:
        index: The tick's multiplier of ``base``.
        base: The base step.
        fmt: Formatting options.
        notation: The axis' notation.

    Returns:
        The formatted label.
    """
    return format_tick(index, base, fmt, notation)
