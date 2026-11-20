"""Tests for the small value types and the pixel-snapping conventions."""

from __future__ import annotations

import numpy as np
import pytest

from pygridplot.core.coords import Bounds, ScreenRect
from pygridplot.core.pixels import line_start, line_starts, snap, snap_array


class _RectLike:
    """Duck-typed stand-in for ``pygame.Rect``."""

    def __init__(self, x: float, y: float, w: float, h: float) -> None:
        self.x, self.y, self.w, self.h = x, y, w, h


def test_screen_rect_edges_follow_pygame_conventions() -> None:
    rect = ScreenRect(10, 20, 100, 50)
    assert (rect.left, rect.top, rect.right, rect.bottom) == (10, 20, 110, 70)
    assert rect.size == (100, 50)
    assert rect.contains_point(10, 20)
    assert not rect.contains_point(110, 20)  # right edge is exclusive
    assert not rect.contains_point(10, 70)


def test_screen_rect_from_any_accepts_rect_like_objects_and_tuples() -> None:
    assert ScreenRect.from_any(_RectLike(1, 2, 3, 4)) == ScreenRect(1, 2, 3, 4)
    assert ScreenRect.from_any((5, 6, 7, 8)) == ScreenRect(5, 6, 7, 8)
    assert ScreenRect.from_any(_RectLike(1.7, 2.2, 3.9, 4.1)) == ScreenRect(1, 2, 3, 4)
    rect = ScreenRect(0, 0, 1, 1)
    assert ScreenRect.from_any(rect) is rect


def test_screen_rect_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ScreenRect(0, 0, -1, 5)
    with pytest.raises(TypeError):
        ScreenRect.from_any("not a rect")


def test_bounds_helpers() -> None:
    b = Bounds(-1.0, 3.0, 2.0, 4.0)
    assert b.width == 4.0
    assert b.height == 2.0
    assert b.center == (1.0, 3.0)
    assert b.range("x") == (-1.0, 3.0)
    assert b.range("y") == (2.0, 4.0)
    assert b.contains_point(3.0, 2.0)
    assert not b.contains_point(3.1, 2.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0), (0.49, 0), (0.5, 1), (1.5, 2), (2.5, 3), (-0.5, 0), (-0.51, -1)],
)
def test_snap_rounds_half_up_not_to_even(value: float, expected: int) -> None:
    assert snap(value) == expected


def test_snap_array_matches_scalar() -> None:
    values = np.array([-1.5, -0.5, 0.25, 0.5, 10.49, 10.5])
    assert snap_array(values).tolist() == [snap(v) for v in values]


@pytest.mark.parametrize(
    ("center", "width", "expected"),
    [
        (10.0, 1, 10),
        (10.4, 1, 10),
        (10.0, 2, 10),  # covers 10 and 11
        (10.2, 2, 10),
        (10.7, 2, 10),
        (10.0, 3, 9),  # covers 9, 10, 11
    ],
)
def test_line_start_centres_lines(center: float, width: int, expected: int) -> None:
    assert line_start(center, width) == expected


def test_line_starts_matches_scalar() -> None:
    centers = np.array([0.0, 3.3, 7.5, -2.6])
    for width in (1, 2, 3, 4):
        expected = [line_start(c, width) for c in centers]
        assert line_starts(centers, width).tolist() == expected
