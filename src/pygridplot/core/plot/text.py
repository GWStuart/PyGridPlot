r"""Text in world space: lines of text anchored to a point.

:meth:`pygridplot.Grid.text` writes text at a world position, and the text
moves with that point while panning and zooming, at a fixed size in
pixels. Where the text sits relative to its point is its **anchor**, the
point of the text's box placed there: ``"center"`` centres the text on it,
``"bottom-left"`` puts the box's bottom-left corner there (so the text is
above and to the right of the point), and so on round the box. A ``gap``
moves the text that many pixels further from the point, on the anchored
sides, so a label can sit beside a marker instead of touching it.

The box runs from the top of the first line's digits to the lowest reach
of the last line's letters (the font's descent), so text with and without
descenders sits alike. Lines (split at ``"\n"``) are a fixed multiple of
that height apart, and aligned to the anchor's side: left-aligned for the
left anchors, right-aligned for the right ones, centred otherwise.

Positions follow the pixel conventions of :mod:`pygridplot.core.pixels`:
continuous coordinates have pixel centres at whole numbers, and the box is
moved to whole pixels (text is drawn crisp, on the pixel grid): anchored
sides away from the point, so the gap is never less than asked, and
centred boxes to the nearest pixel (:func:`~pygridplot.core.pixels.snap`).
Panning by whole pixels therefore moves the text by exactly as many.
"""

from __future__ import annotations

import math
from typing import Final, Literal, TypeAlias

from pygridplot.core.pixels import snap
from pygridplot.core.primitives import LabelText, TextRun, TextSize

Anchor: TypeAlias = Literal[
    "center",
    "left",
    "right",
    "top",
    "bottom",
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
]
"""The point of a text's box that is placed at its position."""

ANCHORS: tuple[Anchor, ...] = (
    "center",
    "left",
    "right",
    "top",
    "bottom",
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
)
"""All anchors."""

LINE_SPACING: Final = 1.25
"""The distance between lines, as a multiple of a line's box height."""


def check_anchor(anchor: object) -> Anchor:
    """Return a valid anchor.

    Args:
        anchor: The anchor given.

    Returns:
        The anchor.

    Raises:
        ValueError: If it is not one of :data:`ANCHORS`.
    """
    for known in ANCHORS:
        if anchor == known:
            return known
    msg = f"anchor must be one of {ANCHORS}, got {anchor!r}"
    raise ValueError(msg)


def check_gap(gap: object) -> float:
    """Return a valid gap between text and its position, in pixels.

    Args:
        gap: The gap given.

    Returns:
        The gap as a float.

    Raises:
        ValueError: If it is not a finite, non-negative number.
    """
    try:
        value = float(gap)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        msg = f"gap must be a number of pixels, got {gap!r}"
        raise ValueError(msg) from error
    if not (math.isfinite(value) and value >= 0.0):
        msg = f"gap must be non-negative and finite, got {gap!r}"
        raise ValueError(msg)
    return value


def text_lines(text: str) -> tuple[LabelText, ...]:
    r"""Split text into lines, each as label text of one run.

    Args:
        text: The text; ``"\n"`` starts a new line.

    Returns:
        The lines, in order (an empty line is kept, for spacing).
    """
    return tuple(LabelText((TextRun(line),)) for line in text.split("\n"))


def place_lines(
    position: tuple[float, float],
    sizes: list[TextSize],
    anchor: Anchor,
    gap: float,
) -> list[tuple[int, int]]:
    """Return where each line of a text goes, anchored to a screen position.

    Args:
        position: The anchor point, in continuous screen pixels.
        sizes: The measured size of each line (at least one).
        anchor: Which point of the text's box is placed at ``position``.
        gap: Pixels between the position and the box, on the anchored sides.

    Returns:
        Each line's ``(left, top)``: the top-left pixel of its digit box.
    """
    px, py = position
    height = max(size.height + size.descent for size in sizes)
    advance = round(LINE_SPACING * height)
    box_width = max(size.width for size in sizes)
    box_height = advance * (len(sizes) - 1) + height
    horizontal = "left" if "left" in anchor else "right" if "right" in anchor else ""
    vertical = "top" if "top" in anchor else "bottom" if "bottom" in anchor else ""
    # A box of n pixels starting at pixel p covers [p - 0.5, p + n - 0.5].
    # Anchored sides round away from the point, so the gap is never less
    # than asked (and less than a pixel more); centred boxes round to the
    # nearest pixel.
    if horizontal == "left":
        left = math.ceil(px + gap + 0.5)
    elif horizontal == "right":
        left = math.floor(px - gap - box_width + 0.5)
    else:
        left = snap(px - (box_width - 1) / 2)
    if vertical == "top":
        top = math.ceil(py + gap + 0.5)
    elif vertical == "bottom":
        top = math.floor(py - gap - box_height + 0.5)
    else:
        top = snap(py - (box_height - 1) / 2)
    placed = []
    for index, size in enumerate(sizes):
        if horizontal == "left":
            line_left = left
        elif horizontal == "right":
            line_left = left + box_width - size.width
        else:
            line_left = left + (box_width - size.width) // 2
        placed.append((line_left, top + index * advance))
    return placed
