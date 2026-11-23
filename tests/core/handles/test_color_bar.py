"""Tests for the colour bar (:mod:`pygridplot.core.handles.color_bar`)."""

from __future__ import annotations

import pytest

from pygridplot.core.coords import ScreenRect
from pygridplot.core.grid.theme import LIGHT
from pygridplot.core.handles.color_bar import (
    BAR_LENGTH_PX,
    color_bar_primitives,
    color_bar_ticks,
    layout_color_bar,
)
from pygridplot.core.handles.panel import MARGIN_PX
from pygridplot.core.plot.color_maps import check_stops
from pygridplot.core.primitives import Image, Stroke
from tests.conftest import FakeMeasurer

RECT = ScreenRect(0, 0, 800, 600)


def texts(ticks: list) -> list[str]:
    return [text.plain for _, text in ticks]


def test_linear_ticks_are_round_and_exact() -> None:
    assert texts(color_bar_ticks(0, 1, log=False, theme=LIGHT)) == [
        "0", "0.2", "0.4", "0.6", "0.8", "1",
    ]  # fmt: skip
    assert texts(color_bar_ticks(-3, 7, log=False, theme=LIGHT)) == [
        "−2", "0", "2", "4", "6",
    ]  # fmt: skip
    values = [v for v, _ in color_bar_ticks(0.1, 0.35, log=False, theme=LIGHT)]
    assert all(0.1 <= v <= 0.35 for v in values)


def test_log_ticks_are_decades_or_their_multiples() -> None:
    assert texts(color_bar_ticks(0.1, 1000, log=True, theme=LIGHT)) == [
        "0.1", "1", "10", "100", "1000",
    ]  # fmt: skip
    assert texts(color_bar_ticks(1, 20, log=True, theme=LIGHT)) == [
        "1", "2", "5", "10", "20",
    ]  # fmt: skip
    # Nothing round inside: the ends, to four digits.
    assert texts(color_bar_ticks(1.31, 1.47, log=True, theme=LIGHT)) == ["1.31", "1.47"]


def test_the_strip_runs_from_low_at_the_bottom_to_high_at_the_top() -> None:
    panel = layout_color_bar(
        0, 1, log=False, title="speed", corner="top-right", rect=RECT, theme=LIGHT,
        measurer=FakeMeasurer(),
    )  # fmt: skip
    assert panel is not None
    assert panel.rect.right == RECT.right - MARGIN_PX
    assert panel.rect.top == RECT.top + MARGIN_PX
    assert panel.bar.height == BAR_LENGTH_PX
    first, last = panel.ticks[0], panel.ticks[-1]
    assert first[0] == pytest.approx(panel.bar.bottom - 0.5)  # 0 at the bottom
    assert last[0] == pytest.approx(panel.bar.top - 0.5)  # 1 at the top
    assert panel.title is not None
    assert panel.title[2] < panel.bar.top  # the title above
    shapes, labels = color_bar_primitives(panel, check_stops("gray"), LIGHT)
    (image,) = [s for s in shapes if isinstance(s, Image)]
    assert image.pixels.shape == (BAR_LENGTH_PX, panel.bar.width, 4)
    assert image.pixels[0, 0, 0] > 250  # white (high) at the top
    assert image.pixels[-1, 0, 0] < 5  # black (low) at the bottom
    assert any(isinstance(s, Stroke) for s in shapes)
    assert [t.text.plain for t in labels][-1] == "speed"


def test_colour_bars_stack_beyond_other_panels() -> None:
    occupied = ScreenRect(600, 10, 190, 120)
    panel = layout_color_bar(
        0, 1, log=False, title=None, corner="top-right", rect=RECT, theme=LIGHT,
        measurer=FakeMeasurer(), occupied=occupied,
    )  # fmt: skip
    assert panel is not None
    assert panel.rect.top == occupied.bottom + MARGIN_PX
    assert panel.title is None
    empty = ScreenRect(0, 0, 0, 0)
    assert (
        layout_color_bar(
            0,
            1,
            log=False,
            title=None,
            corner="top-left",
            rect=empty,
            theme=LIGHT,
            measurer=FakeMeasurer(),
        )
        is None
    )
