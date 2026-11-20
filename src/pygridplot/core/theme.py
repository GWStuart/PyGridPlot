"""Grid appearance: the :class:`GridTheme` dataclass and built-in presets.

Themes are frozen dataclasses. To customise one, derive a new theme from a
preset with :func:`dataclasses.replace`::

    from dataclasses import replace
    import pygridplot as pgp

    theme = replace(pgp.LIGHT, axis_width=3, target_major_px=120)

Nothing about the grid's appearance is hard-coded anywhere else: every
colour, width, spacing target, font and visibility switch lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from pygridplot.core.coords import ColorLike
from pygridplot.core.labels import NumberFormat
from pygridplot.core.nice import SpacingRule
from pygridplot.core.primitives import FontSpec
from pygridplot.core.spacing import MAX_FADE_BAND, TransitionMode


@dataclass(frozen=True, slots=True)
class GridTheme:
    """Everything that controls how the grid, axes and tick labels look.

    Attributes:
        background: Colour the plane is filled with before the grid is drawn,
            or ``None`` to draw the grid over whatever is already on the
            surface.
        minor_color: Colour of minor grid lines.
        major_color: Colour of major grid lines.
        axis_color: Colour of the two axes.
        label_color: Colour of tick labels.
        label_halo_color: Colour of the outline drawn around label glyphs.
            ``None`` uses :attr:`background` (and no halo if that is ``None``
            too).
        minor_width: Thickness of minor lines in pixels.
        major_width: Thickness of major lines in pixels.
        axis_width: Thickness of the axes in pixels.
        target_major_px: Preferred on-screen distance between major lines.
        spacing_rule: How the ideal spacing is rounded to the 1-2-5 ladder.
        transition: ``"fade"`` cross-fades between spacings while zooming;
            ``"snap"`` switches instantly (with hysteresis).
        fade_band: Half-width (in ``log10`` units of zoom) of the region where
            two spacings are cross-faded. ``0.05`` is about ±12% of zoom.
        hysteresis: Extra zoom (in ``log10`` units) needed before a ``"snap"``
            transition switches back, so the grid cannot flicker.
        show_minor: Whether to draw minor grid lines.
        show_major: Whether to draw major grid lines. If minor lines are
            shown but major lines are not, major positions use the minor style
            so the grid has no gaps.
        show_axes: Whether to draw the axes.
        show_labels: Whether to draw tick labels.
        label_font: Font for labels: ``None`` for the bundled DejaVu Sans,
            a path to a font file, or a system font name.
        label_size: Label font size in points.
        label_halo_width: Thickness of the label halo in pixels (0 disables).
        label_gap: Distance in pixels between an axis and its labels.
        label_edge_margin: Smallest distance in pixels between a label pinned
            to the plane's edge and that edge.
        label_min_separation: Comfortable gap in pixels between neighbouring
            labels on the same axis. When zooming squeezes labels closer
            than this, every other (or every fifth) label fades out and is
            then thinned out.
        hang_minus: If ``True``, x-axis labels centre their digits on the tick
            and let a minus sign hang to the left, as in typeset maths.
        number_format: How tick values are written.
    """

    background: ColorLike | None = (255, 255, 255)
    minor_color: ColorLike = (234, 234, 234)
    major_color: ColorLike = (200, 200, 200)
    axis_color: ColorLike = (32, 32, 32)
    label_color: ColorLike = (32, 32, 32)
    label_halo_color: ColorLike | None = None

    minor_width: int = 1
    major_width: int = 1
    axis_width: int = 2

    target_major_px: float = 100.0
    spacing_rule: SpacingRule = "nearest"
    transition: TransitionMode = "fade"
    fade_band: float = 0.05
    hysteresis: float = 0.04

    show_minor: bool = True
    show_major: bool = True
    show_axes: bool = True
    show_labels: bool = True

    label_font: str | None = None
    label_size: int = 12
    label_halo_width: int = 2
    label_gap: int = 4
    label_edge_margin: int = 4
    label_min_separation: int = 14
    hang_minus: bool = True
    number_format: NumberFormat = field(default_factory=NumberFormat)

    def __post_init__(self) -> None:
        """Validate numeric settings so mistakes fail early and clearly."""
        for name in ("minor_width", "major_width", "axis_width"):
            if getattr(self, name) < 1:
                msg = f"{name} must be at least 1 pixel"
                raise ValueError(msg)
        if not self.target_major_px >= 8:
            msg = "target_major_px must be at least 8 pixels"
            raise ValueError(msg)
        if not 0.0 <= self.fade_band <= MAX_FADE_BAND:
            msg = f"fade_band must be within [0, {MAX_FADE_BAND}]"
            raise ValueError(msg)
        if self.hysteresis < 0:
            msg = "hysteresis must be non-negative"
            raise ValueError(msg)
        if self.label_size < 1 or self.label_halo_width < 0:
            msg = "label_size must be positive and label_halo_width non-negative"
            raise ValueError(msg)

    @property
    def label_font_spec(self) -> FontSpec:
        """The label font as a :class:`~pygridplot.core.primitives.FontSpec`."""
        return FontSpec(self.label_font, self.label_size)

    @property
    def resolved_halo_color(self) -> ColorLike | None:
        """The halo colour actually used: explicit, else the background."""
        if self.label_halo_width == 0:
            return None
        if self.label_halo_color is not None:
            return self.label_halo_color
        return self.background


LIGHT = GridTheme()
"""The default light theme: white paper with grey lines and dark axes."""

DARK = replace(
    LIGHT,
    background=(24, 26, 31),
    minor_color=(38, 41, 47),
    major_color=(62, 66, 75),
    axis_color=(214, 217, 224),
    label_color=(204, 208, 216),
)
"""A dark theme with the same layout as :data:`LIGHT`."""
