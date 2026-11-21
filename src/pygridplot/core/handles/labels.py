"""The coordinate label shown next to a draggable point, such as ``(1.25, −0.5)``.

A dragged point can be anywhere, so its coordinates are rarely round
numbers. Each one is rounded to the size of a pixel at the point (the
smallest change a drag can make): ``1.2345678`` reads ``1.23`` at 50 px per
unit and ``1.2346`` zoomed in a hundred times. The pixel size is measured
through the view, so on a log axis the precision follows the value (a pixel
near 1000 spans more than a pixel near 1). The rounded values are written
like tick labels (:func:`pygridplot.core.grid.formatting.format_value`), so
they use the theme's number format.
"""

from __future__ import annotations

import decimal
import math
from decimal import Decimal
from typing import Final

from pygridplot.core.coords import ScreenRect
from pygridplot.core.grid.formatting import NumberFormat, format_value
from pygridplot.core.primitives import LabelText, TextRun, TextSize
from pygridplot.core.transforms import AxisTransform

LABEL_GAP_PX: Final = 4
"""Distance in pixels between a point's marker and its coordinate label."""

_CONTEXT: Final = decimal.Context(prec=60)
"""Enough digits to round any float to any pixel size it can resolve."""


def coordinate_text(
    x: float,
    y: float,
    transforms: tuple[AxisTransform, AxisTransform],
    scales: tuple[float, float],
    fmt: NumberFormat,
) -> LabelText:
    """Return the label ``(x, y)``, each rounded to a pixel at the point.

    Args:
        x: World x.
        y: World y.
        transforms: The view's x and y axis transforms.
        scales: The view's pixels per transformed unit, ``(x, y)``.
        fmt: How numbers are written.

    Returns:
        The label text.
    """
    runs = [TextRun("(")]
    runs.extend(_value_runs(x, transforms[0], scales[0], fmt))
    runs.append(TextRun(", "))
    runs.extend(_value_runs(y, transforms[1], scales[1], fmt))
    runs.append(TextRun(")"))
    return LabelText(tuple(_merge(runs)))


def rounded_value(value: float, transform: AxisTransform, scale: float) -> Decimal:
    """Round a coordinate to the size of one pixel at it.

    Args:
        value: The coordinate (finite).
        transform: Its axis transform.
        scale: Pixels per transformed unit on that axis.

    Returns:
        The value rounded to ``10^k``, the largest power of ten not above
        the world size of a pixel there.
    """
    exact = Decimal(value)
    pixel = abs(
        float(transform.inverse(float(transform.forward(value)) + 1.0 / scale)) - value
    )
    if not (math.isfinite(pixel) and pixel > 0.0):
        return exact
    quantum = Decimal(1).scaleb(math.floor(math.log10(pixel)))
    try:
        return exact.quantize(quantum, context=_CONTEXT)
    except decimal.InvalidOperation:  # far more digits than a float holds
        return exact


def label_position(
    point: tuple[float, float],
    reach: float,
    size: TextSize,
    rect: ScreenRect,
    margin: int,
) -> tuple[int, int]:
    """Place a label above and to the right of a point, inside the plane.

    The label flips to the left of the point near the right edge and below
    it near the top, and is finally kept inside the plane.

    Args:
        point: The point's screen position.
        reach: How far the marker extends from its centre, in pixels.
        size: The label's measured size.
        rect: The plane's rectangle.
        margin: Smallest distance in pixels from the label to the plane's edge.

    Returns:
        The label's ``(left, top)`` (the top of its digit box).
    """
    px, py = point
    offset = reach + LABEL_GAP_PX
    left = round(px + offset)
    if left + size.width > rect.right - margin:
        left = round(px - offset) - size.width
    top = round(py - offset) - size.height
    if top - size.overflow_top < rect.top + margin:
        top = round(py + offset)
    left = max(rect.left + margin, min(left, rect.right - margin - size.width))
    top = max(
        rect.top + margin + size.overflow_top,
        min(top, rect.bottom - margin - size.height),
    )
    return left, top


def _value_runs(
    value: float, transform: AxisTransform, scale: float, fmt: NumberFormat
) -> tuple[TextRun, ...]:
    """Format one coordinate.

    Args:
        value: The coordinate.
        transform: Its axis transform.
        scale: Pixels per transformed unit.
        fmt: How numbers are written.

    Returns:
        The coordinate's text runs.
    """
    if not math.isfinite(value):
        return (TextRun("?"),)
    return format_value(rounded_value(value, transform, scale), fmt).runs


def _merge(runs: list[TextRun]) -> list[TextRun]:
    """Join neighbouring runs typeset the same way.

    A minus sign inside the label is ordinary text: only a label's leading
    sign may hang outside it.

    Args:
        runs: The runs in order.

    Returns:
        The merged runs.
    """
    merged: list[TextRun] = []
    for run in runs:
        role = "text" if run.role == "sign" else run.role
        if merged and merged[-1].role == role:
            merged[-1] = TextRun(merged[-1].text + run.text, role)
        else:
            merged.append(TextRun(run.text, role))
    return merged
