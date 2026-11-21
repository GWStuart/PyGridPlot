"""Tests for tick label placement (:mod:`pygridplot.core.grid.layout`)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pygridplot.core.coords import ScreenRect
from pygridplot.core.grid.layout import (
    LabelCandidate,
    LabelSpacing,
    PlacedLabel,
    fade_collisions,
    place_origin_label,
    place_x_labels,
    place_y_labels,
    resolve_overlaps,
    separation,
    x_labels_top,
    y_label_left,
)
from pygridplot.core.primitives import LabelText, TextRun, TextSize

RECT = ScreenRect(0, 0, 400, 300)
SPACING = LabelSpacing(gap=4, edge_margin=3, axis_width=2)
TEXT = LabelText((TextRun("12"),))


def test_x_labels_sit_below_a_visible_axis() -> None:
    # A 2-px axis centred on y=100 covers rows 100 and 101.
    assert x_labels_top(100.0, RECT, 9, 0, SPACING) == 101 + 1 + 4


def test_x_labels_are_pushed_up_near_the_bottom_and_pinned_below_it() -> None:
    lowest = 300 - 3 - 9
    assert x_labels_top(287.0, RECT, 9, 0, SPACING) == lowest  # pushed
    assert x_labels_top(5000.0, RECT, 9, 0, SPACING) == lowest  # pinned


def test_x_labels_pin_to_the_top_edge_when_the_axis_is_above() -> None:
    assert x_labels_top(-500.0, RECT, 9, 0, SPACING) == 3
    assert x_labels_top(-500.0, RECT, 9, 4, SPACING) == 3 + 4  # room for overflow


@given(st.floats(-1e3, 1e3), st.floats(0.0, 1.0))
def test_x_label_row_moves_continuously(axis_y: float, step: float) -> None:
    a = x_labels_top(axis_y, RECT, 9, 0, SPACING)
    b = x_labels_top(axis_y + step, RECT, 9, 0, SPACING)
    assert abs(b - a) <= 1  # never a jump while panning


def test_x_labels_are_centred_and_the_minus_sign_hangs() -> None:
    size = TextSize(width=21, height=9, sign_width=7)
    label = LabelCandidate(100.0, TEXT, size)
    (hanging,) = place_x_labels([label], 50, hang_minus=True)
    # Digits are 14 px wide: columns 93..106 centred on 100 (half pixel right).
    assert hanging.left + 7 == 100 - (14 - 1) // 2
    (centred,) = place_x_labels([label], 50, hang_minus=False)
    assert centred.left == 100 - (21 - 1) // 2
    assert hanging.top == centred.top == 50


def test_y_labels_right_align_left_of_the_axis() -> None:
    # The axis covers columns 200 and 201; labels end 4 px left of it.
    assert y_label_left(200.0, RECT, 20, SPACING) == 200 - 4 - 20
    size = TextSize(width=14, height=9)
    candidate = LabelCandidate(150.0, TEXT, size)
    (placed,) = place_y_labels([candidate], 200.0, RECT, SPACING)
    assert placed.left + 14 == 200 - 4  # right-aligned
    assert placed.top == 150 - (9 - 1) // 2  # vertically centred


def test_y_labels_are_pushed_and_pinned_at_both_edges() -> None:
    assert y_label_left(10.0, RECT, 20, SPACING) == 3  # pushed by the left edge
    assert y_label_left(-300.0, RECT, 20, SPACING) == 3  # pinned left
    assert y_label_left(5000.0, RECT, 20, SPACING) == 400 - 3 - 20  # pinned right


@given(st.floats(-1e3, 1e3), st.floats(0.0, 1.0), st.integers(1, 120))
def test_y_label_column_moves_continuously(axis_x: float, step: float, w: int) -> None:
    a = y_label_left(axis_x, RECT, w, SPACING)
    b = y_label_left(axis_x + step, RECT, w, SPACING)
    assert abs(b - a) <= 1


@given(axis_y=st.floats(-1e6, 1e6), overflow=st.integers(0, 6))
def test_x_label_row_is_always_inside_the_plane(axis_y: float, overflow: int) -> None:
    top = x_labels_top(axis_y, RECT, 9, overflow, SPACING)
    assert top - overflow >= RECT.top + SPACING.edge_margin
    assert top + 9 <= RECT.bottom - SPACING.edge_margin


@given(axis_x=st.floats(-1e6, 1e6), width=st.integers(1, 120))
def test_y_labels_are_always_inside_the_plane(axis_x: float, width: int) -> None:
    left = y_label_left(axis_x, RECT, width, SPACING)
    assert left >= RECT.left + SPACING.edge_margin
    assert left + width <= RECT.right - SPACING.edge_margin


def test_origin_label_shares_x_row_and_y_alignment() -> None:
    size = TextSize(width=7, height=9)
    zero = LabelText((TextRun("0"),))
    label = place_origin_label(zero, size, 200.0, RECT, 106, SPACING)
    assert (label.left, label.top) == (200 - 4 - 7, 106)


def test_placed_label_box_includes_superscript_overflow() -> None:
    label = PlacedLabel(TEXT, 10, 20, TextSize(30, 9, 0, 4))
    assert label.box == (10, 16, 40, 29)
    other = PlacedLabel(TEXT, 41, 20, TextSize(5, 9))
    assert not label.overlaps(other)
    assert label.overlaps(other, padding=2)


def test_separation() -> None:
    a = PlacedLabel(TEXT, 0, 0, TextSize(10, 10))
    assert separation(a, PlacedLabel(TEXT, 15, 0, TextSize(10, 10))) == 5
    assert separation(a, PlacedLabel(TEXT, 0, 13, TextSize(10, 10))) == 3
    assert separation(a, PlacedLabel(TEXT, 5, 5, TextSize(10, 10))) < 0


def test_exact_separation_uses_sub_pixel_offsets() -> None:
    size = TextSize(10, 10)
    a = PlacedLabel(TEXT, 0, 0, size, offset=(0.25, 0.0))
    right = PlacedLabel(TEXT, 15, 0, size, offset=(-0.25, 0.0))
    left = PlacedLabel(TEXT, -15, 0, size, offset=(-0.25, 0.0))
    below = PlacedLabel(TEXT, 0, 13, size, offset=(0.0, 0.4))
    assert separation(a, right) == 5  # whole pixels by default
    assert separation(a, right, exact=True) == pytest.approx(4.5)
    assert separation(a, left, exact=True) == pytest.approx(5.5)
    assert separation(a, below, exact=True) == pytest.approx(3.4)


@given(
    start=st.floats(100.0, 120.0),
    step=st.floats(0.0, 0.2),
)
def test_fades_change_smoothly_as_labels_move(start: float, step: float) -> None:
    """Moving a label a fraction of a pixel changes its fade by that fraction."""
    size = TextSize(20, 9)
    fixed = [PlacedLabel(TEXT, 0, 100, size)]

    def faded(y: float) -> float:
        (label,) = place_y_labels([LabelCandidate(y, TEXT, size)], 10.0, RECT, SPACING)
        kept = fade_collisions(fixed, [label], padding=2, fade_px=8)
        return kept[0].opacity if kept else 0.0

    assert abs(faded(start + step) - faded(start)) <= step / 8 + 1e-9


def test_fade_collisions_fades_near_labels_and_drops_overlaps() -> None:
    size = TextSize(20, 9)
    fixed = [PlacedLabel(TEXT, 0, 100, size), PlacedLabel(TEXT, 50, 100, size)]
    hit = PlacedLabel(TEXT, 10, 104, size)  # overlaps the first fixed label
    far_below = PlacedLabel(TEXT, 10, 200, size)
    near = PlacedLabel(TEXT, 10, 100 + 9 + 6, size)  # 6 px below the row
    kept = fade_collisions(fixed, [hit, far_below, near], padding=2, fade_px=8)
    assert [k.top for k in kept] == [200, 115]
    assert kept[0].opacity == 1.0
    assert kept[1].opacity == pytest.approx((6 - 2) / 8)
    assert fade_collisions([], [hit], padding=2, fade_px=8) == [hit]


def test_fade_collisions_is_weighted_by_the_fixed_labels_opacity() -> None:
    size = TextSize(20, 9)
    faint = [PlacedLabel(TEXT, 0, 100, size, opacity=0.25)]
    hit = PlacedLabel(TEXT, 10, 104, size)
    (kept,) = fade_collisions(faint, [hit], padding=2, fade_px=8)
    assert kept.opacity == pytest.approx(0.75)


def test_resolve_overlaps_keeps_the_more_opaque_label() -> None:
    size = TextSize(20, 9)
    strong = PlacedLabel(TEXT, 0, 0, size, opacity=0.9)
    weak = PlacedLabel(TEXT, 10, 0, size, opacity=0.2)
    apart = PlacedLabel(TEXT, 60, 0, size, opacity=0.1)
    assert resolve_overlaps([weak, strong, apart], padding=4) == [strong, apart]
