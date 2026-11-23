"""Category labels: names in place of numbers along an axis.

A bar chart of categories (days, groups, countries) stands its bars at
0, 1, 2, ...; ``grid.draw(x_labels=["Mon", "Tue", ...])`` names those
positions on the axis instead of numbering it, and the names behave like
tick labels: they sit in the tick labels' row, are pushed and pinned to the
edge with their axis, fade where they meet a title or the other axis'
labels, and never overlap.

Names are thinned like numbers when the axis is zoomed out: only every
2nd, 4th, 8th, ... name is kept (counted from the first category, so the
names kept do not change while panning), and the names the next thinning
would remove fade out as they close in, exactly as numeric labels do (see
:func:`~pygridplot.core.grid.ticklabels.thinning_opacity`), so nothing pops.

A categorical axis has no shared ``0`` at the origin: the other axis, if
numeric, labels its own zero.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import TypeAlias

import numpy as np

from pygridplot.core.coords import Axis
from pygridplot.core.grid.layout import LabelCandidate
from pygridplot.core.grid.theme import GridTheme
from pygridplot.core.grid.ticklabels import (
    LABEL_MARGIN_PX,
    MIN_LABEL_GAP_PX,
    label_length,
    thinning_opacity,
)
from pygridplot.core.primitives import FontSpec, LabelText, TextMeasurer, TextRun
from pygridplot.core.view import View

Categories: TypeAlias = tuple[tuple[float, str], ...]
"""Named positions along an axis, in increasing order of position."""

LARGEST_STRIDE = 2**30
"""The coarsest thinning: every this many categories (a guard, never reached)."""


def check_categories(labels: object, name: str) -> Categories | None:
    """Return category labels as named positions.

    Args:
        labels: ``None`` (numbers, as usual), a sequence of names (for the
            positions 0, 1, 2, ...) or a mapping from positions to names.
        name: The argument's name, for error messages.

    Returns:
        ``(position, name)`` pairs in increasing order of position, or
        ``None``.

    Raises:
        TypeError: If a name is not a string, or ``labels`` is neither a
            sequence nor a mapping.
        ValueError: If a position is not a finite number.
    """
    if labels is None:
        return None
    if isinstance(labels, Mapping):
        pairs = list(labels.items())
    elif isinstance(labels, str):
        msg = f"{name} must be a sequence of names or a mapping, not a string"
        raise TypeError(msg)
    else:
        try:
            pairs = list(enumerate(labels))  # type: ignore[arg-type]
        except TypeError as error:
            msg = f"{name} must be a sequence of names or a mapping from positions"
            raise TypeError(msg) from error
    checked = []
    for position, text in pairs:
        if not isinstance(text, str):
            msg = f"{name} names must be strings, got {text!r}"
            raise TypeError(msg)
        if (
            isinstance(position, bool)
            or not isinstance(position, numbers.Real)
            or not math.isfinite(position)
        ):
            msg = f"{name} positions must be finite numbers, got {position!r}"
            raise ValueError(msg)
        checked.append((float(position), text))
    checked.sort(key=lambda pair: pair[0])
    return tuple(checked)


def category_candidates(
    axis: Axis,
    categories: Categories,
    view: View,
    theme: GridTheme,
    measurer: TextMeasurer,
    font: FontSpec,
) -> list[LabelCandidate]:
    """Return the category names an axis would like to draw, with opacities.

    Args:
        axis: ``"x"`` or ``"y"``.
        categories: The named positions, in increasing order.
        view: The view being drawn.
        theme: Supplies the comfortable gap between labels.
        measurer: Measures the names.
        font: The label font.

    Returns:
        Candidates for the names near the plane, thinned (see the module
        docstring), in increasing order of position.
    """
    if not categories:
        return []
    transform = view.axis_transform(axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        where = np.asarray(
            view.axis_to_screen(
                axis, transform.forward(np.array([p for p, _ in categories]))
            ),
            dtype=np.float64,
        )
    rect = view.rect
    low, high = (rect.left, rect.right) if axis == "x" else (rect.top, rect.bottom)
    near = (
        np.isfinite(where)
        & (where >= low - LABEL_MARGIN_PX)
        & (where <= high + LABEL_MARGIN_PX)
    )
    chosen = np.flatnonzero(near)
    if chosen.size == 0:
        return []
    texts = {i: LabelText((TextRun(categories[i][1]),)) for i in chosen.tolist()}
    sizes = {i: measurer.measure_text(text, font) for i, text in texts.items()}
    length = label_length(axis, [(float(where[i]), sizes[i]) for i in texts], rect)
    gaps = np.abs(np.diff(where[chosen]))
    gaps = gaps[gaps > 0]
    gap = float(np.min(gaps)) if gaps.size else math.inf
    needed = length + MIN_LABEL_GAP_PX
    comfortable = length + theme.label_min_separation
    stride = 1
    while stride * gap < needed and stride < LARGEST_STRIDE:
        stride *= 2
    fading = thinning_opacity(stride * gap, needed, comfortable)
    candidates = []
    for i in chosen.tolist():
        if i % stride:
            continue
        opacity = 1.0 if i % (2 * stride) == 0 else fading
        if opacity > 0:
            candidates.append(
                LabelCandidate(float(where[i]), texts[i], sizes[i], opacity)
            )
    return candidates
