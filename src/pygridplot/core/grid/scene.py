"""Building the complete grid (lines, axes and labels) for one frame.

:func:`build_grid` is the single entry point. It is a pure function of the
view, the theme and a small :class:`GridState` (only used for hysteresis in
``"snap"`` mode), and it returns a :class:`GridScene` of backend-neutral
primitives for a renderer to draw.

Pipeline for each axis (see :func:`axis_grid`):

1. Choose the major spacing (:func:`choose_axis_spacing`, using
   :mod:`pygridplot.core.grid.spacing`), possibly as a cross-fade between
   two ladder steps. On a log axis this is the spacing of the decades.
2. Generate the lines with their opacities: classified candidate lines on a
   linear axis (:mod:`pygridplot.core.grid.ticks`), or the three tiers of
   decades, log-paper lines and subdivisions on a log axis
   (:mod:`pygridplot.core.grid.logticks`). Both end up as
   :class:`~pygridplot.core.grid.gridlines.AxisLines`.
3. Choose which labels to show and how opaque they are
   (:mod:`pygridplot.core.grid.ticklabels` or
   :mod:`pygridplot.core.grid.loglabels`).

Then, for both axes together: paint the lines as line batches
(:mod:`pygridplot.core.grid.gridlines`), add the axes, and place the labels
(:mod:`pygridplot.core.grid.layout`) with their collision rules (x labels
win over the ``0``, which wins over y labels).

Painter's order: minor-style lines, then major-style lines, then the axes,
then labels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final, Literal

from pygridplot.core.coords import Axis, ColorLike, ScreenRect
from pygridplot.core.grid.formatting import ZERO_LABEL
from pygridplot.core.grid.gridlines import (
    AxisLines,
    axis_batches,
    line_batches,
    linear_axis_lines,
)
from pygridplot.core.grid.layout import (
    LabelCandidate,
    LabelSpacing,
    PlacedLabel,
    fade_collisions,
    place_origin_label,
    place_x_labels,
    place_y_labels,
    resolve_overlaps,
    soften_overlaps,
    x_labels_top,
)
from pygridplot.core.grid.loglabels import log_label_candidates
from pygridplot.core.grid.logticks import compute_log_ticks, decade_choice
from pygridplot.core.grid.spacing import SnapTracker, SpacingChoice, choose_faded
from pygridplot.core.grid.theme import GridTheme
from pygridplot.core.grid.ticklabels import LABEL_MARGIN_PX, label_candidates
from pygridplot.core.grid.ticks import compute_axis_ticks
from pygridplot.core.primitives import FontSpec, LineBatch, TextItem, TextMeasurer
from pygridplot.core.view import View

COLLISION_PADDING_PX: Final = 2
"""Clearance required between a y label and any x label."""

COLLISION_FADE_PX: Final = 8.0
"""Distance over which a y label fades out as it approaches an x label."""

ORIGIN_FADE_PX: Final = 12.0
"""Distance from the edge over which the ``0`` label fades as the last axis leaves."""

FADE_OVERLAP_PADDING_PX: Final = 4
"""Clearance required between labels of two cross-fading levels on one axis."""

OVERLAP_FADE_PX: Final = 8.0
"""Distance over which neighbouring labels of a log axis fade as they meet.

See :func:`~pygridplot.core.grid.layout.soften_overlaps`.
"""

OFF_SCREEN_PX: Final = 1e6
"""How far off the plane an axis line that is not on screen is placed.

Labels of the other axis pin themselves to the plane's edge whenever the
axis line is off screen. A log axis has no zero, so the perpendicular axis
line is infinitely far away; it is placed this far off the edge instead.
"""


@dataclass(frozen=True, slots=True, eq=False)
class AxisGrid:
    """The lines and label candidates of one axis, before label placement.

    Attributes:
        lines: Every grid line of the axis with its opacities.
        labels: The labels the axis would like to draw.
        overlaps: How labels that come close after placement are treated:
            ``None`` when they cannot (a single linear level), ``"resolve"``
            to keep the more opaque one (a linear cross-fade, whose levels
            fade sequentially), or ``"soften"`` to fade both continuously (a
            log axis, whose levels meet side by side; see
            :func:`~pygridplot.core.grid.layout.soften_overlaps`).
    """

    lines: AxisLines
    labels: list[LabelCandidate]
    overlaps: Literal["resolve", "soften"] | None


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
            The axis' :class:`~pygridplot.core.grid.spacing.SnapTracker`.
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

    x_grid = axis_grid(view, "x", x_choice, theme, measurer)
    y_grid = axis_grid(view, "y", y_choice, theme, measurer)
    ranked = [
        *line_batches(x_grid.lines, rect, theme),
        *line_batches(y_grid.lines, rect, theme),
    ]
    ranked.sort(key=lambda item: item[0])  # stable: x before y at equal rank
    lines = [batch for _, batch in ranked]
    lines.extend(axis_batches(view, theme))

    labels = AxisLabels()
    if theme.show_labels:
        labels = _build_labels(view, theme, measurer, x_grid, y_grid)
    return GridScene(rect, theme.background, tuple(lines), labels, x_choice, y_choice)


def axis_grid(
    view: View,
    axis: Axis,
    choice: SpacingChoice,
    theme: GridTheme,
    measurer: TextMeasurer,
) -> AxisGrid:
    """Build the lines and label candidates of one axis.

    Args:
        view: The view being drawn.
        axis: ``"x"`` or ``"y"``.
        choice: The axis' major spacing (see :func:`choose_axis_spacing`).
        theme: The grid's appearance.
        measurer: Reports label sizes.

    Returns:
        The axis' lines and labels.
    """
    margin = LABEL_MARGIN_PX if theme.show_labels else theme.axis_width + 2.0
    font = theme.label_font_spec
    if view.axis_transform(axis).is_log:
        log_ticks = compute_log_ticks(view, axis, choice, theme, margin_px=margin)
        log_labels = []
        if theme.show_labels:
            log_labels = log_label_candidates(log_ticks, measurer, font)
        return AxisGrid(log_ticks.lines, log_labels, overlaps="soften")
    ticks = compute_axis_ticks(
        view, axis, choice, show_minor=theme.show_minor, margin_px=margin
    )
    lines = linear_axis_lines(ticks, skip_axis=theme.show_axes)
    labels = []
    if theme.show_labels:
        labels = label_candidates(ticks, view, theme, measurer, font)
    return AxisGrid(lines, labels, "resolve" if len(ticks.levels) > 1 else None)


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
        The axis' spacing choice. On a log axis it is the spacing of the
        decades (in decades, never less than one).
    """
    scale = view.axis_scale(axis)
    if theme.transition == "fade":
        choice = choose_faded(
            scale, theme.target_major_px, theme.spacing_rule, theme.fade_band
        )
    else:
        tracker = state.tracker(axis) if state is not None else SnapTracker()
        choice = tracker.choose(
            scale, theme.target_major_px, theme.spacing_rule, theme.hysteresis
        )
    return decade_choice(choice) if view.axis_transform(axis).is_log else choice


def _build_labels(
    view: View,
    theme: GridTheme,
    measurer: TextMeasurer,
    x_grid: AxisGrid,
    y_grid: AxisGrid,
) -> AxisLabels:
    """Place all tick labels.

    Args:
        view: The view being drawn.
        theme: Supplies label settings.
        measurer: Reports label sizes.
        x_grid: The x axis' lines and label candidates.
        y_grid: The y axis' lines and label candidates.

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
    axis_x = _axis_line_position(view, "x")
    axis_y = _axis_line_position(view, "y")

    overflow = max((c.size.overflow_top for c in x_grid.labels), default=0)
    top = x_labels_top(axis_y, rect, zero.height, overflow, spacing)
    fixed = _settle_overlaps(
        place_x_labels(x_grid.labels, top, hang_minus=theme.hang_minus), x_grid
    )
    y_placed = _settle_overlaps(
        place_y_labels(y_grid.labels, axis_x, rect, spacing), y_grid
    )

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


def _settle_overlaps(placed: list[PlacedLabel], grid: AxisGrid) -> list[PlacedLabel]:
    """Treat labels of one axis that come close, as the axis asks.

    Args:
        placed: The axis' placed labels.
        grid: The axis' lines and labels (see :attr:`AxisGrid.overlaps`).

    Returns:
        The labels to draw.
    """
    if grid.overlaps == "resolve":
        return resolve_overlaps(placed, FADE_OVERLAP_PADDING_PX)
    if grid.overlaps == "soften":
        return soften_overlaps(placed, FADE_OVERLAP_PADDING_PX, OVERLAP_FADE_PX)
    return placed


def _axis_line_position(view: View, axis: Axis) -> float:
    """Return the screen coordinate of the line where ``axis`` is zero.

    That is the y axis line (``x = 0``) for ``"x"`` and the x axis line
    (``y = 0``) for ``"y"``. Labels sit beside these lines.

    Args:
        view: The view being drawn.
        axis: ``"x"`` or ``"y"``.

    Returns:
        The coordinate, limited to :data:`OFF_SCREEN_PX` beyond the plane's
        edges. On a log axis zero is infinitely far away (left of the plane
        for x, below it for y), so the other axis' labels stay pinned to
        that edge.
    """
    rect = view.rect
    if axis == "x":
        position = view.to_screen_x(0.0)
        low, high = rect.left - OFF_SCREEN_PX, rect.right + OFF_SCREEN_PX
    else:
        position = view.to_screen_y(0.0)
        low, high = rect.top - OFF_SCREEN_PX, rect.bottom + OFF_SCREEN_PX
    return min(max(position, low), high)


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
