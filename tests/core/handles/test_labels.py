"""Tests for the coordinate labels of draggable points (:mod:`core.handles.labels`)."""

from __future__ import annotations

import pytest

from pygridplot.core.coords import ScreenRect
from pygridplot.core.grid.formatting import NumberFormat
from pygridplot.core.handles.labels import (
    coordinate_text,
    label_position,
    rounded_value,
)
from pygridplot.core.primitives import TextSize
from pygridplot.core.transforms import AxisTransform

LINEAR = AxisTransform()
LOG = AxisTransform("log")


def text(
    x: float, y: float, scale: float = 50.0, fmt: NumberFormat | None = None
) -> str:
    """Return a label's plain text at an equal scale on linear axes."""
    label = coordinate_text(
        x, y, (LINEAR, LINEAR), (scale, scale), fmt or NumberFormat()
    )
    return label.plain


def test_values_are_rounded_to_a_pixel() -> None:
    assert text(1.2345678, -0.5) == "(1.23, −0.5)"  # a pixel is 0.02 units
    assert text(1.2345678, 2.0, scale=5000.0) == "(1.2346, 2)"
    assert text(1234.5678, 0.0, scale=0.05) == "(1230, 0)"  # a pixel is 20 units


def test_log_axes_round_relative_to_the_value() -> None:
    # 100 px per decade: a pixel near 1 spans ~0.023, near 1000 ~23.
    assert float(rounded_value(1.23456, LOG, 100.0)) == pytest.approx(1.23)
    assert float(rounded_value(1234.56, LOG, 100.0)) == pytest.approx(1230.0)


def test_labels_use_the_number_format() -> None:
    assert text(-1.5, 0.25, fmt=NumberFormat(minus="-")) == "(-1.5, 0.25)"
    assert text(3e8, 1.0, scale=1e-8) == "(3×10^8, 0)"


def test_labels_sit_above_right_and_stay_inside() -> None:
    rect = ScreenRect(0, 0, 400, 300)
    size = TextSize(60, 9)
    assert label_position((100.0, 100.0), 6.0, size, rect, 4) == (110, 81)
    left, _ = label_position((390.0, 100.0), 6.0, size, rect, 4)
    assert left + size.width <= 390 - 6  # flipped to the left
    _, top = label_position((100.0, 5.0), 6.0, size, rect, 4)
    assert top >= 5 + 6  # flipped below
