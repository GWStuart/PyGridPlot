"""Tests for category labels on an axis (:mod:`pygridplot.core.grid.categories`)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pygridplot.core.coords import ScreenRect
from pygridplot.core.grid.categories import category_candidates, check_categories
from pygridplot.core.grid.layout import title_box
from pygridplot.core.grid.scene import GridScene, build_grid
from pygridplot.core.grid.theme import LIGHT
from pygridplot.core.view import View

from ...conftest import FakeMeasurer
from .test_scene import as_placed

RECT = ScreenRect(0, 0, 800, 600)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def grid(view: View, **kwargs: object) -> GridScene:
    names = {
        key: check_categories(value, key)
        for key, value in kwargs.items()
        if key in ("x_labels", "y_labels")
    }
    rest = {k: v for k, v in kwargs.items() if k not in names}
    return build_grid(view, LIGHT, FakeMeasurer(), **names, **rest)  # type: ignore[arg-type]


def test_categories_are_names_at_positions() -> None:
    assert check_categories(None, "x_labels") is None
    assert check_categories(["a", "b"], "x_labels") == ((0.0, "a"), (1.0, "b"))
    assert check_categories({3: "c", -1.5: "a"}, "x_labels") == (
        (-1.5, "a"),
        (3.0, "c"),
    )
    with pytest.raises(TypeError, match="strings"):
        check_categories([1, 2], "x_labels")
    with pytest.raises(TypeError, match="string"):
        check_categories("abc", "x_labels")
    with pytest.raises(ValueError, match="finite"):
        check_categories({math.inf: "far"}, "y_labels")
    with pytest.raises(TypeError, match="sequence"):
        check_categories(5, "x_labels")


def test_names_replace_the_numbers_on_their_axis() -> None:
    view = View(RECT, (3.0, 2.0), 80.0)
    scene = grid(view, x_labels=DAYS)
    shown = [item.text.plain for item in scene.axis_labels.x]
    assert shown == DAYS  # 80 px apart: every name fits
    assert scene.axis_labels.zero == ()  # no shared 0
    first = scene.axis_labels.x[0]
    assert abs(first.left + (3 * 7 - 1) / 2 - view.to_screen_x(0.0)) <= 1
    # The numeric y axis labels its own zero (unless a name is at the
    # origin, where the names win).
    assert "0" not in [item.text.plain for item in scene.axis_labels.y]
    moved = grid(view, x_labels={k + 1: name for k, name in enumerate(DAYS)})
    assert "0" in [item.text.plain for item in moved.axis_labels.y]


def test_crowded_names_are_thinned_from_the_first() -> None:
    names = [f"category {k}" for k in range(40)]  # 77 px wide each
    scene = grid(View(RECT, (20.0, 0.0), 20.0), x_labels=names)
    kept = [int(item.text.plain.split()[1]) for item in scene.axis_labels.x]
    assert kept
    stride = kept[1] - kept[0]
    assert stride in (4, 8)  # 20 px apart: every 4th at least
    assert all(k % stride == 0 for k in kept)
    # Panning does not change which are kept.
    view = View(RECT, (20.0, 0.0), 20.0)
    view.pan_pixels(37, 0)
    moved = [
        int(i.text.plain.split()[1]) for i in grid(view, x_labels=names).axis_labels.x
    ]
    assert all(k % stride == 0 for k in moved)


def test_names_fade_before_they_are_thinned_out() -> None:
    names = ["abcdefgh"] * 50  # 56 px wide
    widest = 8 * 7
    for scale, faded in ((100.0, False), (widest + 8.0, True)):
        view = View(RECT, (25.0, 0.0), scale)
        candidates = category_candidates(
            "x",
            check_categories(names, "x_labels") or (),
            view,
            LIGHT,
            FakeMeasurer(),
            LIGHT.label_font_spec,
        )
        odd = [c.opacity for c in candidates][1::2]
        assert (min(odd) < 1.0) is faded


def test_y_names_win_over_the_x_axis_zero() -> None:
    view = View(RECT, (3.0, 1.0), 80.0)
    scene = grid(view, y_labels=["low", "mid", "high"])
    assert next(item.text.plain for item in scene.axis_labels.y) == "low"
    assert all(
        item.text.plain != "0" or item.opacity < 1 for item in scene.axis_labels.x
    )


@settings(max_examples=100, deadline=None)
@given(
    cx=st.floats(-50, 50),
    cy=st.floats(-50, 50),
    log_scale=st.floats(0.0, 3.0),
    count=st.integers(1, 200),
    on_y=st.booleans(),
)
def test_names_never_overlap(
    cx: float, cy: float, log_scale: float, count: int, on_y: bool
) -> None:
    names = [f"name {k}" * (1 + k % 3) for k in range(count)]
    view = View(RECT, (cx, cy), 10**log_scale)
    key = "y_labels" if on_y else "x_labels"
    theme = replace(LIGHT, show_arrows=True)
    scene = build_grid(
        view,
        theme,
        FakeMeasurer(),
        x_title="x title",
        y_title="y title",
        **{key: check_categories(names, key)},  # type: ignore[arg-type]
    )
    measurer = FakeMeasurer()
    titles = {id(i) for i in scene.axis_labels.x_title + scene.axis_labels.y_title}
    boxes = []
    for item in scene.labels:
        placed = as_placed(item, measurer)
        if id(item) in titles:
            placed = replace(placed, size=title_box(placed.size))
        boxes.append(placed)
    for i, a in enumerate(boxes):
        assert 0.0 < a.opacity <= 1.0
        for b in boxes[i + 1 :]:
            if a.overlaps(b):
                assert a.opacity + b.opacity <= 1.0 + 1e-9, (a.text.plain, b.text.plain)
