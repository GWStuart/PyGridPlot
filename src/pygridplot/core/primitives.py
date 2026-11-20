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

Later milestones will add polylines, circles and polygons here.

This module also defines :class:`TextMeasurer`, the one thing the core needs
*from* a renderer: label layout has to know how big each label will be.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from pygridplot.core.coords import ColorLike
from pygridplot.core.labels import LabelText

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
