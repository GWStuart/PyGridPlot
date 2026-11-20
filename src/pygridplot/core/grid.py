"""Building the complete grid (lines, axes and labels) for one frame.

:func:`build_grid` is the single entry point. It is a pure function of the
view, the theme and a small :class:`GridState` (only used for hysteresis in
``"snap"`` mode), and it returns a :class:`GridScene` of backend-neutral
primitives for a renderer to draw.

Pipeline for each axis:

1. Choose the major spacing (:mod:`pygridplot.core.spacing`), possibly as a
   cross-fade between two ladder steps.
2. Generate and classify candidate lines (:mod:`pygridplot.core.ticks`).
3. Paint them as line batches (:mod:`pygridplot.core.gridlines`).
4. Choose which labels to show and how opaque they are
   (:mod:`pygridplot.core.ticklabels`), then place them
   (:mod:`pygridplot.core.layout`).

Painter's order: minor-style lines, then major-style lines, then the axes,
then labels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from pygridplot.core.coords import Axis, ColorLike, ScreenRect
from pygridplot.core.gridlines import axis_batches, grid_line_batches
from pygridplot.core.labels import ZERO_LABEL
from pygridplot.core.layout import (
    LabelSpacing,
    PlacedLabel,
    fade_collisions,
    place_origin_label,
    place_x_labels,
    place_y_labels,
    resolve_overlaps,
    x_labels_top,
)
from pygridplot.core.primitives import FontSpec, LineBatch, TextItem, TextMeasurer
from pygridplot.core.spacing import SnapTracker, SpacingChoice, choose_faded
from pygridplot.core.theme import GridTheme
from pygridplot.core.ticklabels import LABEL_MARGIN_PX, label_candidates
from pygridplot.core.ticks import AxisTicks, compute_axis_ticks
from pygridplot.core.view import View

COLLISION_PADDING_PX: Final = 2
"""Clearance required between a y label and any x label."""

COLLISION_FADE_PX: Final = 8.0
"""Distance over which a y label fades out as it approaches an x label."""

ORIGIN_FADE_PX: Final = 12.0
"""Distance from the edge over which the ``0`` label fades as the last axis leaves."""

FADE_OVERLAP_PADDING_PX: Final = 4
"""Clearance required between labels of two cross-fading levels on one axis."""


@dataclass(frozen=True, slots=True)
class AxisLabels:
    """The tick labels of one frame, grouped by what they label.

    Attributes:
        x: Labels of the x axis.
        zero: The single ``0`` label, if it is shown.
        y: Labels of the y axis.
    """

    x: tuple[TextItem, ...] = ()
    zero: tuple[TextItem, ...] = ()
    y: tuple[TextItem, ...] = ()

    def all(self) -> tuple[TextItem, ...]:
        """Return every label in drawing order (x, then zero, then y).

        Returns:
            All labels.
        """
        return (*self.x, *self.zero, *self.y)


@dataclass(frozen=True, slots=True, eq=False)
class GridScene:
    """Everything needed to draw the grid for one frame.

    Attributes:
        rect: The screen rectangle the grid covers.
        background: Fill colour for the plane, or ``None`` for no fill.
        lines: Line batches in painter's order (axes last).
        axis_labels: Positioned tick labels grouped by axis.
        x_spacing: The major spacing chosen for the x axis.
        y_spacing: The major spacing chosen for the y axis.
    """

    rect: ScreenRect
    background: ColorLike | None
    lines: tuple[LineBatch, ...]
    axis_labels: AxisLabels
    x_spacing: SpacingChoice
    y_spacing: SpacingChoice

    @property
    def labels(self) -> tuple[TextItem, ...]:
        """All tick labels in drawing order (drawn after all lines)."""
        return self.axis_labels.all()


class GridState:
    """Per-plane memory the grid keeps between frames.

    Only ``"snap"`` transitions need it: they remember the last chosen
    spacing for each axis so hysteresis can work. ``"fade"`` transitions are
    stateless.
    """

    def __init__(self) -> None:
        self._trackers: dict[Axis, SnapTracker] = {
            "x": SnapTracker(),
            "y": SnapTracker(),
        }

    def tracker(self, axis: Axis) -> SnapTracker:
        """Return the hysteresis tracker for one axis.

        Args:
            axis: ``"x"`` or ``"y"``.

        Returns:
            The axis' :class:`~pygridplot.core.spacing.SnapTracker`.
        """
        return self._trackers[axis]

    def reset(self) -> None:
        """Forget all remembered choices (for example after a theme change)."""
        for tracker in self._trackers.values():
            tracker.reset()


def build_grid(
    view: View,
    theme: GridTheme,
    measurer: TextMeasurer,
    state: GridState | None = None,
) -> GridScene:
    """Build the grid, axes and labels for the current view.

    Args:
        view: The view to draw.
        theme: The grid's appearance.
        measurer: Reports label sizes (normally the renderer).
        state: Memory for ``"snap"`` hysteresis. Without it, snapping has no
            hysteresis.

    Returns:
        The primitives to draw, in painter's order.
    """
    rect = view.rect
    x_choice = choose_axis_spacing(view, "x", theme, state)
    y_choice = choose_axis_spacing(view, "y", theme, state)
    if rect.is_empty:
        return GridScene(rect, theme.background, (), AxisLabels(), x_choice, y_choice)

    margin = LABEL_MARGIN_PX if theme.show_labels else theme.axis_width + 2.0
    x_ticks = compute_axis_ticks(
        view, "x", x_choice, show_minor=theme.show_minor, margin_px=margin
    )
    y_ticks = compute_axis_ticks(
        view, "y", y_choice, show_minor=theme.show_minor, margin_px=margin
    )

    ranked = [
        *grid_line_batches(x_ticks, rect, theme),
        *grid_line_batches(y_ticks, rect, theme),
    ]
    ranked.sort(key=lambda item: item[0])  # stable: x before y at equal rank
    lines = [batch for _, batch in ranked]
    lines.extend(axis_batches(view, theme))

    labels = AxisLabels()
    if theme.show_labels:
        labels = _build_labels(view, theme, measurer, x_ticks, y_ticks)
    return GridScene(rect, theme.background, tuple(lines), labels, x_choice, y_choice)


def choose_axis_spacing(
    view: View, axis: Axis, theme: GridTheme, state: GridState | None
) -> SpacingChoice:
    """Choose the major spacing for one axis according to the theme.

    Args:
        view: The view being drawn.
        axis: ``"x"`` or ``"y"``.
        theme: Supplies the target spacing, rounding rule and transition mode.
        state: Hysteresis memory for ``"snap"`` mode (optional).

    Returns:
        The axis' spacing choice.
    """
    scale = view.axis_scale(axis)
    if theme.transition == "fade":
        return choose_faded(
            scale, theme.target_major_px, theme.spacing_rule, theme.fade_band
        )
    tracker = state.tracker(axis) if state is not None else SnapTracker()
    return tracker.choose(
        scale, theme.target_major_px, theme.spacing_rule, theme.hysteresis
    )


def _build_labels(
    view: View,
    theme: GridTheme,
    measurer: TextMeasurer,
    x_ticks: AxisTicks,
    y_ticks: AxisTicks,
) -> AxisLabels:
    """Choose, format and place all tick labels.

    Args:
        view: The view being drawn.
        theme: Supplies label settings.
        measurer: Reports label sizes.
        x_ticks: The x axis' candidate lines.
        y_ticks: The y axis' candidate lines.

    Returns:
        The positioned labels. Priority is x labels, then the ``0`` label, then
        y labels: each group fades out where it comes close to a group before
        it.

    The single ``0`` label stays while at least one axis is on screen. With
    the other axis pinned it labels the visible axis line, and it only fades
    out as the last axis leaves the plane.
    """
    rect = view.rect
    font = theme.label_font_spec
    spacing = LabelSpacing(theme.label_gap, theme.label_edge_margin, theme.axis_width)
    zero = measurer.measure_text(ZERO_LABEL, font)
    axis_x = view.to_screen_x(0.0)
    axis_y = view.to_screen_y(0.0)

    x_candidates = label_candidates(x_ticks, view, theme, measurer, font)
    y_candidates = label_candidates(y_ticks, view, theme, measurer, font)

    overflow = max((c.size.overflow_top for c in x_candidates), default=0)
    top = x_labels_top(axis_y, rect, zero.height, overflow, spacing)
    fixed = place_x_labels(x_candidates, top, hang_minus=theme.hang_minus)
    if len(x_ticks.levels) > 1:
        fixed = resolve_overlaps(fixed, FADE_OVERLAP_PADDING_PX)

    y_placed = place_y_labels(y_candidates, axis_x, rect, spacing)
    if len(y_ticks.levels) > 1:
        y_placed = resolve_overlaps(y_placed, FADE_OVERLAP_PADDING_PX)

    origins: list[PlacedLabel] = []
    presence = max(
        _axis_presence(axis_x, rect.left, rect.right),
        _axis_presence(axis_y, rect.top, rect.bottom),
    )
    if presence > 0.0:
        origin = place_origin_label(ZERO_LABEL, zero, axis_x, rect, top, spacing)
        origins = [replace(origin, opacity=presence)]
    pad, fade = COLLISION_PADDING_PX, COLLISION_FADE_PX
    # Priority: x labels, then the 0 label, then y labels.
    origins = fade_collisions(fixed, origins, pad, fade)
    y_placed = fade_collisions([*fixed, *origins], y_placed, pad, fade)
    return AxisLabels(
        x=tuple(_text_item(label, theme, font) for label in fixed),
        zero=tuple(_text_item(label, theme, font) for label in origins),
        y=tuple(_text_item(label, theme, font) for label in y_placed),
    )


def _axis_presence(position: float, low: int, high: int) -> float:
    """Return how firmly an axis line is on screen, for fading the ``0`` label.

    Args:
        position: Continuous screen coordinate of the axis line.
        low: First pixel of the plane along that direction.
        high: One past the last pixel of the plane along that direction.

    Returns:
        1 when the axis is at least :data:`ORIGIN_FADE_PX` inside the plane,
        falling linearly to 0 as it reaches the edge (and 0 beyond it).
    """
    inside = min(position - (low - 0.5), (high - 0.5) - position)
    return min(max(inside / ORIGIN_FADE_PX, 0.0), 1.0)


def _text_item(label: PlacedLabel, theme: GridTheme, font: FontSpec) -> TextItem:
    """Turn a placed label into a drawable :class:`TextItem`.

    Args:
        label: The placed label.
        theme: Supplies colours and the halo settings.
        font: The label font.

    Returns:
        The text primitive.
    """
    halo = theme.resolved_halo_color
    return TextItem(
        text=label.text,
        left=label.left,
        top=label.top,
        font=font,
        color=theme.label_color,
        opacity=label.opacity,
        halo_color=halo,
        halo_width=theme.label_halo_width if halo is not None else 0,
    )
