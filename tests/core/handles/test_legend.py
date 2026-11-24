"""Tests for the legend: layout and swatches (:mod:`pygridplot.core.handles.legend`)."""

from __future__ import annotations

import numpy as np
import pytest

from pygridplot.core.coords import ScreenRect
from pygridplot.core.grid.theme import DARK, LIGHT
from pygridplot.core.handles.legend import (
    MAX_HEAD_PX,
    MAX_LINE_PX,
    MAX_OUTLINE_PX,
    MAX_RADIUS_PX,
    SWATCH_WIDTH_PX,
    LegendEntry,
    check_label,
    layout_legend,
    legend_primitives,
    swatch_primitives,
)
from pygridplot.core.handles.panel import MARGIN_PX, PADDING_PX
from pygridplot.core.plot.style import (
    ArrowStyle,
    BarStyle,
    FillStyle,
    LineStyle,
    PointStyle,
    PolygonStyle,
    SequenceStyle,
)
from pygridplot.core.primitives import Fill, Markers, Stroke
from tests.conftest import FakeMeasurer

RECT = ScreenRect(0, 0, 800, 600)
RED = (200, 0, 0)
ENTRIES = (
    LegendEntry("sin x", LineStyle(), RED),
    LegendEntry("a longer label", FillStyle(), (0, 0, 200)),
)


def layout(*entries: LegendEntry, corner: str = "top-right", **kwargs: object):
    return layout_legend(
        entries or ENTRIES,
        corner,  # type: ignore[arg-type]
        RECT,
        LIGHT,
        FakeMeasurer(),
        **kwargs,  # type: ignore[arg-type]
    )


def test_labels_are_checked() -> None:
    assert check_label(None) is None
    assert check_label("") is None
    assert check_label("sin x") == "sin x"
    with pytest.raises(TypeError, match="string"):
        check_label(3)
    with pytest.raises(ValueError, match="one line"):
        check_label("two\nlines")


def test_entries_compare_by_label_style_and_colour() -> None:
    assert LegendEntry("a", LineStyle(), RED) == LegendEntry("a", LineStyle(), RED)
    assert LegendEntry("a", LineStyle(), RED) != LegendEntry(
        "a", LineStyle(), RED[::-1]
    )
    assert LegendEntry("a", LineStyle(), RED) != LegendEntry("a", FillStyle(), RED)


@pytest.mark.parametrize(
    "corner", ["top-left", "top-right", "bottom-left", "bottom-right"]
)
def test_legends_sit_in_their_corner(corner: str) -> None:
    panel = layout(corner=corner)
    assert panel is not None
    rect = panel.rect
    if corner.endswith("left"):
        assert rect.left == RECT.left + MARGIN_PX
    else:
        assert rect.right == RECT.right - MARGIN_PX
    if corner.startswith("top"):
        assert rect.top == RECT.top + MARGIN_PX
    else:
        assert rect.bottom == RECT.bottom - MARGIN_PX


def test_rows_go_top_to_bottom_with_swatches_beside_their_labels() -> None:
    panel = layout()
    assert panel is not None
    first, second = panel.rows
    assert [row.entry for row in panel.rows] == list(ENTRIES)
    assert first.text_top < second.text_top
    assert first.swatch_left == panel.rect.left + PADDING_PX - 0.5
    assert first.text_left > first.swatch_left + SWATCH_WIDTH_PX
    # The widest label decides the width; everything is inside the panel.
    assert panel.rect.right - PADDING_PX == second.text_left + 14 * 7
    assert panel.rect.bottom - PADDING_PX >= second.text_top + 9 + 3
    assert first.text_top <= first.center_y <= first.text_top + 9


def test_a_legend_goes_beyond_a_panel_already_in_its_corner() -> None:
    sliders = ScreenRect(RECT.right - 230, 10, 220, 80)
    below = layout(occupied=sliders)
    assert below is not None
    assert below.rect.top == sliders.bottom + MARGIN_PX
    low = ScreenRect(10, 500, 220, 90)
    above = layout(corner="bottom-left", occupied=low)
    assert above is not None
    assert above.rect.bottom == low.top - MARGIN_PX


def test_no_entries_or_no_plane_draw_nothing() -> None:
    assert layout_legend((), "top-right", RECT, LIGHT, FakeMeasurer()) is None
    empty = ScreenRect(0, 0, 0, 0)
    assert layout_legend(ENTRIES, "top-right", empty, LIGHT, FakeMeasurer()) is None


def swatch(style: object, color: object = RED) -> list:
    return swatch_primitives(
        LegendEntry("a", style, color), 10.0, 50.0, (255, 255, 255)
    )  # type: ignore[arg-type]


def reach(shapes: list) -> tuple[float, float]:
    """The leftmost and rightmost pixels the shapes cover (pens included)."""
    lows, highs = [], []
    for shape in shapes:
        if isinstance(shape, Stroke):
            xs = np.concatenate([path[:, 0] for path in shape.paths])
            lows.append(xs.min() - shape.width / 2)
            highs.append(xs.max() + shape.width / 2)
        elif isinstance(shape, Fill):
            xs = np.concatenate([ring[:, 0] for ring in shape.rings])
            lows.append(xs.min())
            highs.append(xs.max())
        else:
            lows.append(shape.positions[:, 0].min() - shape.radius)
            highs.append(shape.positions[:, 0].max() + shape.radius)
    return min(lows), max(highs)


@pytest.mark.parametrize(
    "style",
    [
        LineStyle(width=20),
        LineStyle(width=20, dash="dashed"),
        LineStyle(width=2, dash=(1, 3)),
        FillStyle(),
        PolygonStyle(width=10),
        PolygonStyle(width=10, dash="dotted"),
        PointStyle(radius=30, edge_color="white"),
        SequenceStyle(width=12, radius=12),
        ArrowStyle(width=12, head_size=40),
        BarStyle(edge_width=10),
    ],
)
def test_swatches_stay_in_their_box(style: object) -> None:
    low, high = reach(swatch(style))
    assert low >= 10.0 - 1e-9
    assert high <= 10.0 + SWATCH_WIDTH_PX + 1e-9


def test_dashed_swatches_have_dashes_at_both_ends() -> None:
    (line,) = swatch(LineStyle(dash="dashed"))
    assert isinstance(line, Stroke)
    segments = np.asarray(line.paths)
    xs = segments[..., 0]
    # From one end of the swatch to the other, with gaps.
    assert xs.min() == pytest.approx(10.0 + 2.5 / 2)
    assert xs.max() == pytest.approx(10.0 + SWATCH_WIDTH_PX - 2.5 / 2)
    drawn = np.sum(xs[:, 1] - xs[:, 0])
    assert drawn < SWATCH_WIDTH_PX - 2.5 - 4
    _, outline = swatch(PolygonStyle(dash="dashed"))
    assert isinstance(outline, Stroke)
    assert len(np.asarray(outline.paths)) >= 8  # every side starts and ends a dash


def test_swatches_look_like_their_plots() -> None:
    (line,) = swatch(LineStyle(width=20, opacity=0.5))
    assert isinstance(line, Stroke)
    assert (line.width, line.opacity, line.color) == (MAX_LINE_PX, 0.5, RED)
    (area,) = swatch(FillStyle(opacity=0.3))
    assert isinstance(area, Fill)
    assert area.opacity == 0.3
    fill, outline = swatch(PolygonStyle(width=10, fill_color=(0, 0, 255)))
    assert isinstance(fill, Fill)
    assert fill.color == (0, 0, 255)
    assert isinstance(outline, Stroke)
    assert outline.width == MAX_OUTLINE_PX
    assert [type(s) for s in swatch(PolygonStyle(fill=False))] == [Stroke]
    assert [type(s) for s in swatch(PolygonStyle(width=0))] == [Fill]
    (marker,) = swatch(PointStyle(radius=30, marker="diamond", edge_color="white"))
    assert isinstance(marker, Markers)
    assert (marker.radius, marker.shape, marker.edge_width) == (
        MAX_RADIUS_PX,
        "diamond",
        1.5,
    )
    stem, head = swatch(SequenceStyle())
    assert isinstance(stem, Stroke)
    assert isinstance(head, Markers)
    assert stem.paths[0][:, 1].max() > head.positions[0, 1]  # the stem hangs below
    assert [type(s) for s in swatch(SequenceStyle(width=0))] == [Markers]
    assert [type(s) for s in swatch(SequenceStyle(radius=0))] == [Stroke]
    (arrow,) = swatch(ArrowStyle(head_size=40))
    ring = arrow.rings[0]
    assert ring.shape == (7, 2)
    assert ring[:, 0].max() - ring[2, 0] <= MAX_HEAD_PX + 1e-9  # the head is capped
    (bar,) = swatch(ArrowStyle(head_size=0))
    assert np.ptp(bar.rings[0][:, 1]) == pytest.approx(2.5)  # no head: a plain bar


def test_primitives_draw_the_panel_swatches_and_labels() -> None:
    panel = layout()
    assert panel is not None
    shapes, texts = legend_primitives(panel, LIGHT)
    assert isinstance(shapes[0], Fill)
    assert shapes[0].color == LIGHT.background
    assert isinstance(shapes[1], Stroke)  # the border
    assert [text.text.plain for text in texts] == ["sin x", "a longer label"]
    assert all(text.color == LIGHT.label_color for text in texts)
    shapes, texts = legend_primitives(panel, DARK)
    assert shapes[0].color == DARK.background  # type: ignore[union-attr]
    assert texts[0].color == DARK.label_color


def test_bars_and_error_bars_have_their_swatches() -> None:
    fill, outline = swatch(BarStyle(opacity=0.4, edge_width=10))
    assert isinstance(fill, Fill)
    assert fill.opacity == 0.4
    assert isinstance(outline, Stroke)
    assert outline.width == MAX_OUTLINE_PX
    assert outline.opacity == 1.0
    assert [type(s) for s in swatch(BarStyle(edge_width=0))] == [Fill]
    assert [type(s) for s in swatch(BarStyle(opacity=0))] == [Stroke]
    plain = swatch_primitives(LegendEntry("a", PointStyle(), RED), 10.0, 50.0, None)
    assert [type(s) for s in plain] == [Markers]
    with_bars = swatch_primitives(
        LegendEntry("a", PointStyle(), RED, errors=True), 10.0, 50.0, None
    )
    bar, marker = with_bars
    assert isinstance(bar, Stroke)
    assert isinstance(marker, Markers)
    assert len(bar.paths) == 3  # the bar and two caps
    assert np.ptp(bar.paths[0][:, 1]) > 2 * PointStyle().radius  # beyond the marker
    alone = swatch_primitives(
        LegendEntry("a", PointStyle(radius=0), RED, errors=True), 10.0, 50.0, None
    )
    assert [type(s) for s in alone] == [Stroke]
    assert LegendEntry("a", PointStyle(), RED) != LegendEntry(
        "a", PointStyle(), RED, errors=True
    )
