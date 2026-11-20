"""Backend-neutral drawing primitives: the contract between core and renderers.

The maths layer never draws anything. It describes what to draw as small
immutable primitives in **screen space** (continuous pixel coordinates), and
a renderer (see :mod:`pygridplot.render`) turns them into actual drawing
calls. Keeping this boundary narrow is what makes other backends (OpenGL,
SVG export, ...) realistic later.

Current primitives:

* :class:`LineBatch`: many parallel, axis-aligned lines that share a paint
  (grid lines and axes).
* :class:`TextItem`: a positioned label.
* :class:`Stroke`: polylines drawn with a round pen (function curves,
  polylines and polygon outlines).
* :class:`Fill`: filled polygons (closed rings).
* :class:`Markers`: points drawn as fixed-size markers.

and the vocabulary they are described in: :class:`LabelText` (a label made
of typeset :class:`TextRun` pieces), :data:`MarkerShape` and
:data:`FillRule`.

All coordinates are continuous screen pixels (integer values are pixel
centres; see :mod:`pygridplot.core.pixels`).

This module also defines :class:`TextMeasurer`, the one thing the core needs
*from* a renderer: label layout has to know how big each label will be.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.coords import ColorLike

MarkerShape: TypeAlias = Literal["circle", "square", "diamond", "cross", "plus"]
"""The shape drawn for each point."""

FillRule: TypeAlias = Literal["nonzero", "evenodd"]
"""How overlapping parts of a polygon are filled (as in SVG and the HTML canvas).

``"nonzero"`` fills everything enclosed by the outline, and ``"evenodd"``
leaves areas that are enclosed an even number of times empty (for example
the middle of a five-pointed star).
"""

MARKER_SHAPES: tuple[MarkerShape, ...] = (
    "circle",
    "square",
    "diamond",
    "cross",
    "plus",
)
"""All supported marker shapes."""

FILL_RULES: tuple[FillRule, ...] = ("nonzero", "evenodd")
"""All supported fill rules."""

RunRole: TypeAlias = Literal["text", "sign", "superscript"]
"""How a :class:`TextRun` should be typeset.

* ``"text"``: normal text on the baseline.
* ``"sign"``: a leading minus sign. It is drawn like ``"text"`` but kept
  separate so layout can let it hang outside the digits.
* ``"superscript"``: a smaller, raised exponent.
"""


@dataclass(frozen=True, slots=True)
class TextRun:
    """A piece of a label with a single typesetting role.

    Attributes:
        text: The characters in this run.
        role: How the run is typeset (see :data:`RunRole`).
    """

    text: str
    role: RunRole = "text"


@dataclass(frozen=True, slots=True)
class LabelText:
    """A formatted label made up of one or more :class:`TextRun` pieces.

    Attributes:
        runs: The pieces of the label in reading order.
    """

    runs: tuple[TextRun, ...]

    @property
    def plain(self) -> str:
        """The label as plain text, with ``^`` before superscripts.

        Useful for debugging and tests, e.g. ``"2×10^6"``.
        """
        return "".join(
            f"^{run.text}" if run.role == "superscript" else run.text
            for run in self.runs
        )

    @property
    def has_sign(self) -> bool:
        """Whether the label starts with a separate minus-sign run."""
        return bool(self.runs) and self.runs[0].role == "sign"

    @property
    def has_superscript(self) -> bool:
        """Whether any run of the label is a superscript."""
        return any(run.role == "superscript" for run in self.runs)


FULL_OPACITY: Final = 0.999
"""Opacities above this are treated as exactly 1 (absorbs float round-off)."""


class Orientation(enum.Enum):
    """The direction of the lines in a :class:`LineBatch`."""

    VERTICAL = "vertical"
    """Lines of constant screen x (constant world x)."""

    HORIZONTAL = "horizontal"
    """Lines of constant screen y (constant world y)."""


@dataclass(frozen=True, slots=True)
class PaintLayer:
    """One colour in a paint stack, with an opacity multiplier.

    Attributes:
        color: The colour. Any alpha it carries is multiplied by ``opacity``.
        opacity: Opacity in ``[0, 1]``.
    """

    color: ColorLike
    opacity: float = 1.0


@dataclass(frozen=True, slots=True, eq=False)
class LineBatch:
    """A batch of parallel, axis-aligned lines drawn with the same paint.

    The paint is a small stack of :class:`PaintLayer` objects, composited
    bottom to top. This is how cross-fades are described: a line that is
    half-way between "major" and "minor" is painted as the minor colour with
    the major colour on top at the right opacity.

    Attributes:
        orientation: Whether the lines are vertical or horizontal.
        positions: Continuous screen coordinate of each line (x for vertical
            lines, y for horizontal ones).
        span: ``(start, end)`` continuous screen coordinates along the lines
            (the y range of vertical lines, the x range of horizontal lines).
            ``end`` is exclusive.
        layers: The paint stack, bottom first.
        width: Thickness of each line in whole pixels.
    """

    orientation: Orientation
    positions: NDArray[np.float64]
    span: tuple[float, float]
    layers: tuple[PaintLayer, ...]
    width: int = 1

    def __post_init__(self) -> None:
        """Validate the width and paint stack."""
        if self.width < 1:
            msg = f"line width must be at least 1 pixel, got {self.width}"
            raise ValueError(msg)
        if not self.layers:
            msg = "a LineBatch needs at least one paint layer"
            raise ValueError(msg)

    def __len__(self) -> int:
        return int(self.positions.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class Stroke:
    """Polylines drawn with a round pen of a fixed pixel width.

    Joins and ends are round, so a polyline looks the same however finely it
    is divided into segments.

    Attributes:
        paths: The polylines, each an array of shape ``(N, 2)``. A path with a
            single point is drawn as a dot.
        color: Pen colour.
        width: Pen width in pixels.
        opacity: Opacity in ``[0, 1]``. Overlapping parts of one stroke do not
            darken each other.
    """

    paths: tuple[NDArray[np.float64], ...]
    color: ColorLike
    width: float
    opacity: float = 1.0


@dataclass(frozen=True, slots=True, eq=False)
class Fill:
    """Filled polygons, with antialiased edges.

    Attributes:
        rings: Closed rings, each an array of vertices of shape ``(N, 2)``.
            Several rings are filled together, so later rings can cut holes
            (with ``"evenodd"``, or with opposite orientation and
            ``"nonzero"``).
        color: Fill colour.
        opacity: Opacity in ``[0, 1]``.
        rule: Which overlapping parts count as inside.
    """

    rings: tuple[NDArray[np.float64], ...]
    color: ColorLike
    opacity: float = 1.0
    rule: FillRule = "nonzero"


@dataclass(frozen=True, slots=True, eq=False)
class Markers:
    """Points drawn as identical fixed-size markers.

    Attributes:
        positions: Marker centres, shape ``(N, 2)``.
        shape: The marker shape.
        radius: Marker size in pixels (see
            :attr:`pygridplot.core.plot.style.PointStyle.radius`).
        color: Fill colour.
        edge_color: Outline colour, or ``None`` for no outline.
        edge_width: Outline thickness in pixels.
        opacity: Opacity in ``[0, 1]``.
    """

    positions: NDArray[np.float64]
    shape: MarkerShape
    radius: float
    color: ColorLike
    edge_color: ColorLike | None = None
    edge_width: float = 0.0
    opacity: float = 1.0


@dataclass(frozen=True, slots=True)
class FontSpec:
    """Identifies a font at a size.

    Attributes:
        name: ``None`` for the library's bundled default font, a path to a
            ``.ttf``/``.otf`` file, or a system font name.
        size: Font size in points (roughly the pixel height of the em box).
    """

    name: str | None = None
    size: int = 12

    def __post_init__(self) -> None:
        """Validate that the size is positive."""
        if self.size < 1:
            msg = f"font size must be positive, got {self.size}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TextSize:
    """The measured size of a label, as reported by a renderer.

    Sizes describe the label's **digit box**: from the top of the digits to
    the baseline, and the full advance width. Layout aligns digit boxes so
    that labels line up exactly with grid lines no matter which digits
    they contain. Superscripts poke out above the box by ``overflow_top``.

    Attributes:
        width: Total width in pixels.
        height: Height of the digit box in pixels.
        sign_width: Width of a leading minus-sign run (0 if there is none).
        overflow_top: Extra pixels above the digit box taken by superscripts.
    """

    width: int
    height: int
    sign_width: int = 0
    overflow_top: int = 0


@dataclass(frozen=True, slots=True)
class TextItem:
    """A label positioned on screen.

    Attributes:
        text: What to draw.
        left: Screen x of the left edge of the label's digit box.
        top: Screen y of the top edge of the label's digit box.
        font: The font to draw with.
        color: Text colour.
        opacity: Opacity in ``[0, 1]`` (used when fading labels in and out).
        halo_color: Colour of the outline drawn around the glyphs so labels
            stay readable on top of grid lines, or ``None`` for no halo.
        halo_width: Thickness of the halo in pixels.
    """

    text: LabelText
    left: int
    top: int
    font: FontSpec
    color: ColorLike
    opacity: float = 1.0
    halo_color: ColorLike | None = None
    halo_width: int = 0


class TextMeasurer(Protocol):
    """Anything that can report how big a label will be when drawn."""

    def measure_text(self, text: LabelText, font: FontSpec) -> TextSize:
        """Return the size of ``text`` drawn with ``font``.

        Args:
            text: The label to measure.
            font: The font it will be drawn with.

        Returns:
            The label's measured size.
        """
        ...
