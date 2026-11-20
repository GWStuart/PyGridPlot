"""Placement of tick labels around the axes.

The rules follow the conventions of graphing calculators such as Desmos:

* x-axis labels sit just **below** the x-axis, centred on their tick.
* y-axis labels sit just **left** of the y-axis, right-aligned, and are
  vertically centred on their tick.
* When an axis gets close to the plane's edge, its labels are **pushed**
  along so they stay inside the plane, and once the axis is off-screen they
  stay **pinned** to that edge. The scale is therefore always readable, and
  because pushing is continuous, labels never jump while panning. While an
  axis passes through its own pinned labels, their halo keeps them
  readable.
* Zero is labelled once, at the bottom-left of the origin. The label stays
  while at least one axis is on screen (then it labels that axis line).
* Labels fade out as they come close to labels with a higher priority
  (x labels, then the ``0``, then y labels) instead of overlapping them,
  which matters in corners when both axes are pinned.

All positions are snapped with :mod:`pygridplot.core.pixels`, using the
same rules the renderer uses to draw lines. A label therefore sits a fixed
number of pixels from the line it belongs to and never wobbles while
zooming.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from pygridplot.core.coords import ScreenRect
from pygridplot.core.labels import LabelText
from pygridplot.core.pixels import line_start, snap
from pygridplot.core.primitives import TextSize


@dataclass(frozen=True, slots=True)
class LabelCandidate:
    """A label that would like to be drawn next to a tick.

    Attributes:
        position: Continuous screen coordinate of the tick along its axis
            (x for x-axis labels, y for y-axis labels).
        text: The label text.
        size: The measured size of the label.
        opacity: Opacity in ``(0, 1]``.
    """

    position: float
    text: LabelText
    size: TextSize
    opacity: float = 1.0


@dataclass(frozen=True, slots=True)
class PlacedLabel:
    """A label with its final pixel position.

    Attributes:
        text: The label text.
        left: Screen x of the left edge of the digit box.
        top: Screen y of the top edge of the digit box.
        size: The measured size of the label.
        opacity: Opacity in ``(0, 1]``.
    """

    text: LabelText
    left: int
    top: int
    size: TextSize
    opacity: float = 1.0

    @property
    def box(self) -> tuple[int, int, int, int]:
        """The occupied pixels as ``(left, top, right, bottom)``, exclusive ends.

        The box includes any superscript overflow above the digit box.
        """
        return (
            self.left,
            self.top - self.size.overflow_top,
            self.left + self.size.width,
            self.top + self.size.height,
        )

    def overlaps(self, other: PlacedLabel, padding: int = 0) -> bool:
        """Return whether two labels' boxes overlap.

        Args:
            other: The other label.
            padding: Extra clearance in pixels required between the boxes.

        Returns:
            ``True`` if the boxes, grown by ``padding``, intersect.
        """
        return separation(self, other) < padding


@dataclass(frozen=True, slots=True)
class LabelSpacing:
    """The distances used when placing labels.

    Attributes:
        gap: Pixels between an axis line and its labels.
        edge_margin: Smallest distance in pixels between a label and the
            plane's edge when the label is pushed or pinned.
        axis_width: Thickness of the axis lines in pixels.
    """

    gap: int
    edge_margin: int
    axis_width: int


def x_labels_top(
    axis_y: float,
    rect: ScreenRect,
    height: int,
    overflow_top: int,
    spacing: LabelSpacing,
) -> int:
    """Return the top of the row of x-axis labels' digit boxes.

    Args:
        axis_y: Continuous screen y of the x-axis (may be off-screen).
        rect: The plane's screen rectangle.
        height: Digit box height of the labels.
        overflow_top: Largest superscript overflow among the labels.
        spacing: Gap, margin and axis-width settings.

    Returns:
        Screen y of the top of the digit boxes: just below the axis, pushed
        or pinned so the row always stays inside the plane.
    """
    first_row = line_start(axis_y, spacing.axis_width)
    below = first_row + spacing.axis_width + spacing.gap
    highest = rect.top + spacing.edge_margin + overflow_top
    lowest = rect.bottom - spacing.edge_margin - height
    return _clamp(below, highest, lowest)


def place_x_labels(
    candidates: list[LabelCandidate], top: int, *, hang_minus: bool
) -> list[PlacedLabel]:
    """Centre each x-axis label horizontally on its tick.

    Args:
        candidates: The labels to place.
        top: Screen y of the row of digit boxes (see :func:`x_labels_top`).
        hang_minus: If ``True``, centre the digits only and let a leading
            minus sign hang to the left.

    Returns:
        The placed labels.
    """
    placed = []
    for label in candidates:
        lead = label.size.sign_width if hang_minus else 0
        body = label.size.width - lead
        left = snap(label.position) - lead - (body - 1) // 2
        placed.append(PlacedLabel(label.text, left, top, label.size, label.opacity))
    return placed


def y_label_left(
    axis_x: float, rect: ScreenRect, width: int, spacing: LabelSpacing
) -> int:
    """Return the left edge of a y-axis label of the given width.

    Args:
        axis_x: Continuous screen x of the y-axis (may be off-screen).
        rect: The plane's screen rectangle.
        width: Width of the label.
        spacing: Gap, margin and axis-width settings.

    Returns:
        Screen x of the label's left edge: right-aligned just left of the
        axis, pushed or pinned so the label always stays inside the plane.
    """
    first_col = line_start(axis_x, spacing.axis_width)
    right_aligned = first_col - spacing.gap - width
    leftmost = rect.left + spacing.edge_margin
    rightmost = rect.right - spacing.edge_margin - width
    return _clamp(right_aligned, leftmost, rightmost)


def place_y_labels(
    candidates: list[LabelCandidate],
    axis_x: float,
    rect: ScreenRect,
    spacing: LabelSpacing,
) -> list[PlacedLabel]:
    """Place y-axis labels beside the axis, centred on their ticks.

    Args:
        candidates: The labels to place.
        axis_x: Continuous screen x of the y-axis (may be off-screen).
        rect: The plane's screen rectangle.
        spacing: Gap, margin and axis-width settings.

    Returns:
        The placed labels.
    """
    placed = []
    for label in candidates:
        left = y_label_left(axis_x, rect, label.size.width, spacing)
        top = snap(label.position) - (label.size.height - 1) // 2
        placed.append(PlacedLabel(label.text, left, top, label.size, label.opacity))
    return placed


def place_origin_label(
    text: LabelText,
    size: TextSize,
    axis_x: float,
    rect: ScreenRect,
    top: int,
    spacing: LabelSpacing,
) -> PlacedLabel:
    """Place the single ``0`` label next to the origin.

    It shares the x-axis labels' row and the y-axis labels' alignment, so it
    sits at the bottom-left of the origin, pushed along with the other labels
    near the edges.

    Args:
        text: The zero label.
        size: Its measured size.
        axis_x: Continuous screen x of the y-axis.
        rect: The plane's screen rectangle.
        top: The x labels' row (see :func:`x_labels_top`).
        spacing: Gap, margin and axis-width settings.

    Returns:
        The placed label.
    """
    return PlacedLabel(text, y_label_left(axis_x, rect, size.width, spacing), top, size)


def separation(a: PlacedLabel, b: PlacedLabel) -> int:
    """Return how far apart two labels' boxes are.

    Args:
        a: One label.
        b: The other label.

    Returns:
        The gap in pixels along whichever axis separates the boxes (the larger
        of the horizontal and vertical gaps). Zero or negative means the boxes
        touch or overlap.
    """
    l1, t1, r1, b1 = a.box
    l2, t2, r2, b2 = b.box
    return max(l2 - r1, l1 - r2, t2 - b1, t1 - b2)


def resolve_overlaps(labels: list[PlacedLabel], padding: int) -> list[PlacedLabel]:
    """Remove labels that overlap a more opaque label on the same axis.

    Labels of one grid level never overlap (they are thinned out first), but
    while two levels are cross-faded their label sets interleave. Wide labels
    can then collide. The more opaque label wins, so at any moment the
    dominant level's labels are the ones that stay readable.

    Args:
        labels: Placed labels of one axis.
        padding: Clearance in pixels required between kept labels.

    Returns:
        The surviving labels, in their original order.
    """
    order = sorted(range(len(labels)), key=lambda i: -labels[i].opacity)
    kept: list[int] = []
    for i in order:
        if not any(labels[i].overlaps(labels[j], padding) for j in kept):
            kept.append(i)
    return [labels[i] for i in sorted(kept)]


def fade_collisions(
    fixed: list[PlacedLabel],
    candidates: list[PlacedLabel],
    padding: int,
    fade_px: float,
) -> list[PlacedLabel]:
    """Fade out candidates as they approach a fixed label, dropping overlaps.

    A candidate is pushed out completely once it comes within ``padding``
    pixels of a fixed label and is unaffected from ``padding + fade_px``
    pixels away, with a linear fade in between. The push is weighted by the
    fixed label's own opacity, so a label that is itself fading in pushes
    others out gradually. Labels sliding towards each other (for example a
    y label towards the row of pinned x labels while panning) therefore fade
    out smoothly instead of popping.

    Args:
        fixed: Labels that take priority (they are never changed).
        candidates: Labels to fade or drop.
        padding: Clearance in pixels at which a candidate is fully pushed out.
        fade_px: Distance over which the push fades.

    Returns:
        The surviving candidates, with reduced opacity where they are close
        to a fixed label.
    """
    if not fixed:
        return candidates
    reach = padding + fade_px
    band_top = min(label.box[1] for label in fixed) - reach
    band_bottom = max(label.box[3] for label in fixed) + reach
    kept = []
    for label in candidates:
        _, top, _, bottom = label.box
        factor = 1.0
        if top < band_bottom and band_top < bottom:
            for other in fixed:
                closeness = _closeness(separation(label, other), padding, fade_px)
                factor = min(factor, 1.0 - other.opacity * closeness)
        if factor <= 1e-3:
            continue
        if factor < 1.0:
            label = replace(label, opacity=label.opacity * factor)
        kept.append(label)
    return kept


def _closeness(gap: int, padding: int, fade_px: float) -> float:
    """Return how strongly a label at distance ``gap`` is pushed out.

    Args:
        gap: Separation between two label boxes (see :func:`separation`).
        padding: Gap at or below which the push is total.
        fade_px: Distance beyond ``padding`` over which the push fades out.

    Returns:
        1 at or below ``padding``, 0 beyond ``padding + fade_px``, linear between.
    """
    if gap <= padding:
        return 1.0
    if fade_px <= 0.0:
        return 0.0
    return max(0.0, 1.0 - (gap - padding) / fade_px)


def _clamp(value: int, low: int, high: int) -> int:
    """Clamp ``value`` into ``[low, high]``, preferring ``low`` if they cross.

    Args:
        value: The preferred position.
        low: The smallest allowed position.
        high: The largest allowed position.

    Returns:
        The clamped position (``low`` when the plane is too small for both).
    """
    return max(low, min(value, high))
