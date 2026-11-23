"""Tests for text in world space (:mod:`pygridplot.core.plot.text` and its builder)."""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pygridplot.core.coords import ScreenRect
from pygridplot.core.plot.builders import text_items
from pygridplot.core.plot.style import TextStyle
from pygridplot.core.plot.text import (
    ANCHORS,
    LINE_SPACING,
    check_anchor,
    check_gap,
    place_lines,
    text_lines,
)
from pygridplot.core.primitives import FontSpec, TextSize
from pygridplot.core.view import View
from tests.conftest import FakeMeasurer

RECT = ScreenRect(0, 0, 400, 300)
FONT = FontSpec(None, 12)
ONE = [TextSize(21, 9, descent=3)]  # a three-letter line: a 21 x 12 box


def box(placed: list[tuple[int, int]], sizes: list[TextSize]) -> tuple[float, ...]:
    """The continuous edges (left, top, right, bottom) of the lines' box."""
    left = min(x for x, _ in placed) - 0.5
    right = max(x + s.width for (x, _), s in zip(placed, sizes, strict=True)) - 0.5
    top = placed[0][1] - 0.5
    last = sizes[-1]
    bottom = placed[-1][1] + last.height + last.descent - 0.5
    return left, top, right, bottom


def test_anchors_and_gaps_are_validated() -> None:
    assert check_anchor("top-left") == "top-left"
    with pytest.raises(ValueError, match="anchor"):
        check_anchor("middle")
    assert check_gap(3) == 3.0
    for bad in (-1, math.nan, math.inf, "a", None):
        with pytest.raises(ValueError, match="gap"):
            check_gap(bad)


def test_lines_are_split_at_newlines() -> None:
    assert [line.plain for line in text_lines("a\n\nbc")] == ["a", "", "bc"]
    assert [line.plain for line in text_lines("one")] == ["one"]


def test_a_centred_text_is_centred_on_its_point() -> None:
    left, top, right, bottom = box(place_lines((100.0, 50.0), ONE, "center", 0), ONE)
    assert abs((left + right) / 2 - 100.0) <= 0.5
    assert abs((top + bottom) / 2 - 50.0) <= 0.5
    assert (right - left, bottom - top) == (21, 12)


@pytest.mark.parametrize(
    ("anchor", "edges"),
    [
        ("left", {"left": 106.0}),
        ("right", {"right": 94.0}),
        ("top", {"top": 56.0}),
        ("bottom", {"bottom": 44.0}),
        ("top-left", {"left": 106.0, "top": 56.0}),
        ("bottom-right", {"right": 94.0, "bottom": 44.0}),
    ],
)
def test_anchored_sides_keep_the_gap(anchor: str, edges: dict[str, float]) -> None:
    placed = place_lines((100.0, 50.0), ONE, check_anchor(anchor), 6.0)
    left, top, right, bottom = box(placed, ONE)
    found = {"left": left, "top": top, "right": right, "bottom": bottom}
    for side, value in edges.items():
        # Never closer than the gap, and less than a pixel further.
        away = found[side] - value if side in ("left", "top") else value - found[side]
        assert 0.0 <= away < 1.0
    if "left" not in anchor and "right" not in anchor:
        assert abs((left + right) / 2 - 100.0) <= 0.5  # centred across


def test_lines_are_spaced_and_aligned_to_the_anchor() -> None:
    sizes = [TextSize(70, 9, descent=3), TextSize(28, 9, descent=3)]
    advance = round(LINE_SPACING * 12)
    for anchor, second_left in (("left", 100), ("right", 142), ("center", 121)):
        first, second = place_lines((0.0, 0.0), sizes, check_anchor(anchor), 0)
        assert second[1] - first[1] == advance
        assert second[0] - first[0] == second_left - 100
    _, top, _, bottom = box(place_lines((0.0, 0.0), sizes, "center", 0), sizes)
    assert bottom - top == advance + 12


@settings(max_examples=200, deadline=None)
@given(
    x=st.floats(-1000, 1000),
    y=st.floats(-1000, 1000),
    anchor=st.sampled_from(ANCHORS),
    gap=st.floats(0, 20),
    dx=st.integers(-50, 50),
    dy=st.integers(-50, 50),
)
def test_panning_by_whole_pixels_moves_text_exactly(
    x: float, y: float, anchor: str, gap: float, dx: int, dy: int
) -> None:
    sizes = [TextSize(35, 9, descent=3), TextSize(14, 9, descent=3)]
    where = check_anchor(anchor)
    before = place_lines((x, y), sizes, where, gap)
    # Exact ties (a box edge half way between pixels) may round either way.
    for value in (x + gap, x - gap, y + gap, y - gap, x - 17.0, y - 13.5):
        assume(abs((value + 0.5) % 1.0 - 0.5) > 1e-6)
        assume(abs(value % 1.0 - 0.5) > 1e-6)
    after = place_lines((x + dx, y + dy), sizes, where, gap)
    assert after == [(left + dx, top + dy) for left, top in before]


# --- The builder ---------------------------------------------------------------


def items(position: tuple[float, float], text: str = "max", **kwargs: object) -> list:
    view = kwargs.pop("view", View(RECT, (0.0, 0.0), 50.0))
    return text_items(
        view,  # type: ignore[arg-type]
        text,
        position,
        TextStyle(opacity=0.5),
        (10, 20, 30),
        font=FONT,
        halo=((255, 255, 255), 2),
        measurer=FakeMeasurer(),
        **kwargs,  # type: ignore[arg-type]
    )


def test_text_items_carry_the_style() -> None:
    (item,) = items((0.0, 0.0))
    assert item.text.plain == "max"
    assert (item.color, item.opacity, item.font) == ((10, 20, 30), 0.5, FONT)
    assert (item.halo_color, item.halo_width) == ((255, 255, 255), 2)
    # Centred on the origin, the middle of the plane.
    sx, _sy = View(RECT, (0.0, 0.0), 50.0).to_screen(0.0, 0.0)
    assert abs(item.left + (21 - 1) / 2 - sx) <= 0.5
    assert len(items((0.0, 0.0), "two\nlines", anchor="left", gap=4.0)) == 2


def test_text_that_cannot_be_placed_or_seen_is_left_out() -> None:
    assert items((math.nan, 0.0)) == []
    assert items((0.0, math.inf)) == []
    assert items((1e300, 1e300)) == []  # far off the plane
    assert items((10.0, 0.0)) == []  # just off the right edge (x = 4 is the edge)
    assert items((0.0, 0.0), "") == []
    log = View(RECT, (1.0, 0.0), (150.0, 50.0), equal_aspect=False, log_x=True)
    assert items((-1.0, 0.0), view=log) == []  # not positive on a log axis
    assert len(items((10.0, 0.0), view=log)) == 1
    # Of several lines, only those reaching the plane.
    edge = View(RECT, (0.0, 0.0), 50.0).to_world(200.0, -10.0)
    assert len(items(edge, "a\nb\nc", anchor="bottom")) == 0
    assert [i.text.plain for i in items(edge, "a\nb\nc", anchor="top")] == [
        "a",
        "b",
        "c",
    ]
    above = View(RECT, (0.0, 0.0), 50.0).to_world(200.0, 8.0)
    assert [i.text.plain for i in items(above, "a\nb\nc", anchor="bottom")] == ["c"]


def test_text_styles_are_validated() -> None:
    assert TextStyle(size=14).size == 14
    for size in (0, 1.5, True, -3):
        with pytest.raises(ValueError, match="size"):
            TextStyle(size=size)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="opacity"):
        TextStyle(opacity=2.0)
