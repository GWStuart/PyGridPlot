"""Tests for logarithmic axes (:mod:`pygridplot.core.grid.logticks`).

Besides examples for each tier (decades, log-paper lines, subdivisions),
this module checks the same properties as the linear grid's property tests:
lines stay in the plane, labels never overlap, panning is rigid and zooming
never pops a line or a label.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pygridplot.core.coords import Axis, ScreenRect
from pygridplot.core.grid.layout import PlacedLabel
from pygridplot.core.grid.logticks import (
    compute_log_ticks,
    decade_choice,
    decade_levels,
    major_threshold_px,
)
from pygridplot.core.grid.nice import NiceStep
from pygridplot.core.grid.scene import (
    GridScene,
    GridState,
    build_grid,
    choose_axis_spacing,
)
from pygridplot.core.grid.spacing import SpacingChoice
from pygridplot.core.grid.theme import LIGHT, GridTheme
from pygridplot.core.pixels import line_starts
from pygridplot.core.primitives import Orientation, TextItem
from pygridplot.core.view import View

from ...conftest import FakeMeasurer

RECT = ScreenRect(0, 0, 800, 600)


def log_view(
    scale: float = 150.0,
    center: tuple[float, float] = (1.0, 1.0),
    *,
    log_x: bool = True,
    log_y: bool = True,
) -> View:
    """Return a view of ``RECT`` with log axes and one shared scale."""
    return View(RECT, center, scale, log_x=log_x, log_y=log_y, equal_aspect=False)


def scene_for(view: View, theme: GridTheme = LIGHT) -> GridScene:
    """Build the grid with the fake measurer."""
    return build_grid(view, theme, FakeMeasurer(), GridState())


def x_label_texts(scene: GridScene) -> list[str]:
    """Return the plain text of the fully opaque x labels, left to right."""
    items = sorted(scene.axis_labels.x, key=lambda item: item.left)
    return [item.text.plain for item in items if item.opacity == 1.0]


def as_placed(item: TextItem, measurer: FakeMeasurer) -> PlacedLabel:
    """Rebuild a :class:`PlacedLabel` from a drawn label, for overlap checks."""
    size = measurer.measure_text(item.text, item.font)
    return PlacedLabel(item.text, item.left, item.top, size, item.opacity)


def axis_lines(
    view: View, axis: Axis = "x", *, margin_px: float = 0.0
) -> dict[float, tuple[float, float, float]]:
    """Map each line's exponent (rounded) to ``(position, total, major)``."""
    choice = choose_axis_spacing(view, axis, LIGHT, None)
    lines = compute_log_ticks(view, axis, choice, LIGHT, margin_px=margin_px).lines
    exponents = view.screen_to_axis(axis, lines.positions)
    return {
        round(float(u), 9): (float(p), float(t), float(m))
        for u, p, t, m in zip(
            exponents, lines.positions, lines.total, lines.major, strict=True
        )
    }


def opacities(view: View) -> dict[float, tuple[float, float]]:
    """Map each line's exponent (rounded) to its ``(total, major)`` opacity."""
    return {key: (t, m) for key, (_, t, m) in axis_lines(view).items()}


# ----------------------------------------------------------------------
# Decades
# ----------------------------------------------------------------------
def test_decade_choice_never_goes_below_one_decade() -> None:
    assert decade_choice(SpacingChoice(NiceStep(-3))) == SpacingChoice(NiceStep(0))
    blended = SpacingChoice(NiceStep(-1), NiceStep(0), 0.4)
    assert decade_choice(blended) == SpacingChoice(NiceStep(0))
    coarse = SpacingChoice(NiceStep(2), NiceStep(3), 0.5)
    assert decade_choice(coarse) is coarse


def test_decade_minor_lines_are_whole_decades() -> None:
    def minors(level: int) -> NiceStep | None:
        (only,) = decade_levels(SpacingChoice(NiceStep(level)), show_minor=True)
        return only.minor

    assert minors(0) is None  # the log-paper lines take over
    assert minors(1) == NiceStep(0)  # 2 decades: every decade is minor
    assert minors(2) == NiceStep(0)  # 5 decades
    assert minors(3) == NiceStep(1)  # 10 decades: every 2nd
    (level,) = decade_levels(SpacingChoice(NiceStep(2)), show_minor=False)
    assert level.minor is None


def test_coarse_views_label_every_few_decades_as_powers() -> None:
    scene = scene_for(log_view(scale=20.0))
    assert str(scene.x_spacing.fine) == "5"
    texts = x_label_texts(scene)
    assert "10^−10" in texts  # with a true minus sign
    assert "10^10" in texts
    assert "1" in texts
    assert "10" not in texts
    assert "10^3" not in texts


def test_default_view_labels_each_decade() -> None:
    scene = scene_for(log_view())
    texts = x_label_texts(scene)
    assert texts[texts.index("0.01") : texts.index("100") + 1] == [
        "0.01",
        "0.1",
        "1",
        "10",
        "100",
    ]
    assert not any(t.startswith(("2", "5")) for t in texts)


def test_log_axes_have_no_zero_label_and_no_axis_lines() -> None:
    scene = scene_for(log_view())
    assert scene.axis_labels.zero == ()
    assert all(b.layers[-1].color != LIGHT.axis_color for b in scene.lines)


def test_labels_of_a_log_axis_are_pinned_to_the_edge() -> None:
    scene = scene_for(log_view())
    measurer = FakeMeasurer()
    x_tops = {item.top for item in scene.axis_labels.x}
    assert len(x_tops) == 1
    (top,) = x_tops
    assert (
        top
        + measurer.measure_text(
            scene.axis_labels.x[0].text, LIGHT.label_font_spec
        ).height
        <= RECT.bottom
    )
    assert top > RECT.bottom - 30  # at the bottom edge
    assert min(item.left for item in scene.axis_labels.y) <= RECT.left + 10


def test_semilog_keeps_the_linear_axis_line_and_zero_label() -> None:
    view = log_view(scale=100.0, center=(0.0, 1.0), log_x=False)
    scene = scene_for(view)
    vertical_axis = [
        b
        for b in scene.lines
        if b.layers[-1].color == LIGHT.axis_color
        and b.orientation is Orientation.VERTICAL
    ]
    assert len(vertical_axis) == 1  # the line x = 0
    assert [item.text.plain for item in scene.axis_labels.zero] == ["0"]
    # The x labels sit at the bottom: the line y = 0 is infinitely far down.
    assert min(item.top for item in scene.axis_labels.x) > RECT.bottom - 30


# ----------------------------------------------------------------------
# Log-paper lines and subdivisions
# ----------------------------------------------------------------------
def test_log_paper_lines_fade_in_together() -> None:
    hidden = opacities(log_view(scale=60.0))
    paper = opacities(log_view(scale=110.0))
    for m in range(2, 10):
        u = round(math.log10(m), 9)
        assert u not in hidden
        assert paper[u][0] == pytest.approx(1.0)  # every one is a minor line
        assert paper[u][1] == 0.0


def test_two_and_five_become_major_before_the_other_integers() -> None:
    lines = opacities(log_view(scale=400.0))
    major = {m: lines[round(math.log10(m), 9)][1] for m in range(2, 10)}
    assert major[2] == major[5] == 1.0
    assert all(major[m] == 0.0 for m in (3, 4, 6, 7, 8, 9))
    texts = x_label_texts(scene_for(log_view(scale=400.0)))
    assert {"0.2", "0.5", "2", "5"} <= set(texts)
    assert "3" not in texts


def test_fine_views_label_integers_and_subdivisions() -> None:
    texts = x_label_texts(scene_for(log_view(scale=1500.0, center=(3.0, 3.0))))
    assert {"2", "2.5", "3", "3.5", "4", "5"} <= set(texts)


def test_deep_zoom_looks_like_a_linear_axis() -> None:
    scene = scene_for(log_view(scale=1e5, center=(3.7, 3.7)))
    values = sorted(Decimal(t) for t in x_label_texts(scene))
    steps = {b - a for a, b in itertools.pairwise(values)}
    assert len(values) >= 5
    assert steps == {Decimal("0.01")}


def test_snap_mode_switches_log_lines_outright() -> None:
    snap = replace(LIGHT, transition="snap")
    lines = compute_log_ticks(
        log_view(scale=420.0), "x", SpacingChoice(NiceStep(0)), snap, margin_px=0.0
    ).lines
    assert set(np.unique(lines.total)) <= {0.0, 1.0}
    assert set(np.unique(lines.major)) <= {0.0, 1.0}


def test_major_threshold_follows_the_spacing_rule() -> None:
    assert major_threshold_px(LIGHT) == pytest.approx(100 / math.sqrt(2.5))
    ceil = replace(LIGHT, spacing_rule="ceil")
    assert major_threshold_px(ceil) == pytest.approx(100.0)


def test_scientific_labels_are_powers_of_ten() -> None:
    texts = x_label_texts(scene_for(log_view(center=(1e8, 1.0))))
    assert {"10^7", "10^8", "10^9"} <= set(texts)
    assert not any("×" in t for t in texts)


# ----------------------------------------------------------------------
# Property tests
# ----------------------------------------------------------------------
centres = st.floats(-12.0, 12.0)  # log10 of the centre
scales = st.floats(0.0, 6.0)  # log10 of pixels per decade


@settings(max_examples=120, deadline=None)
@given(cx=centres, cy=centres, log_scale=scales)
def test_lines_stay_inside_and_opacities_are_valid(
    cx: float, cy: float, log_scale: float
) -> None:
    view = log_view(10**log_scale, (10**cx, 10**cy))
    scene = scene_for(view)
    for batch in scene.lines:
        low, high = (0, 800) if batch.orientation is Orientation.VERTICAL else (0, 600)
        assert np.all(batch.positions > low - 3)
        assert np.all(batch.positions < high + 3)
        assert all(0.0 < layer.opacity <= 1.0 for layer in batch.layers[-1:])
    for total, major in opacities(view).values():
        assert 0.0 <= major <= total <= 1.0


@settings(max_examples=120, deadline=None)
@given(cx=centres, cy=centres, log_scale=scales)
def test_log_labels_never_overlap(cx: float, cy: float, log_scale: float) -> None:
    measurer = FakeMeasurer()
    scene = scene_for(log_view(10**log_scale, (10**cx, 10**cy)))
    placed = [as_placed(item, measurer) for item in scene.labels]
    for i, a in enumerate(placed):
        assert 0.0 < a.opacity <= 1.0
        for b in placed[i + 1 :]:
            if a.overlaps(b):
                assert a.opacity + b.opacity <= 1.0 + 1e-9, (a.text.plain, b.text.plain)


def snapped_lines(
    scene: GridScene, orientation: Orientation
) -> set[tuple[int, object]]:
    """Return ``(first pixel, paint)`` of every line in one direction."""
    result = set()
    for batch in scene.lines:
        if batch.orientation is orientation:
            for start in line_starts(batch.positions, batch.width).tolist():
                result.add((start, batch.layers))
    return result


@settings(max_examples=120, deadline=None)
@given(
    cx=centres,
    cy=centres,
    log_scale=st.floats(1.0, 6.0),
    dx=st.integers(-40, 40),
    dy=st.integers(-40, 40),
)
def test_panning_a_log_view_moves_every_line_rigidly(
    cx: float, cy: float, log_scale: float, dx: int, dy: int
) -> None:
    view = log_view(10**log_scale, (10**cx, 10**cy))
    before = scene_for(view)
    near_tie = any(np.any(np.abs(b.positions % 1.0 - 0.5) < 1e-6) for b in before.lines)
    assume(not near_tie)
    view.pan_pixels(dx, dy)
    after = scene_for(view)
    for orientation, shift, size in (
        (Orientation.VERTICAL, dx, 800),
        (Orientation.HORIZONTAL, dy, 600),
    ):
        moved = {(p + shift, c) for p, c in snapped_lines(before, orientation)}
        now = snapped_lines(after, orientation)
        inside = range(45, size - 45)
        assert {m for m in moved if m[0] in inside} == {
            n for n in now if n[0] in inside
        }


@settings(max_examples=200, deadline=None)
@given(cx=centres, log_scale=st.floats(0.5, 6.0))
def test_zooming_never_pops_log_lines(cx: float, log_scale: float) -> None:
    """A tiny zoom step changes every line's opacities only a little.

    Lines are generated well beyond the plane, so any line inside it in one
    frame exists in the other too, unless it is nearly transparent.
    """
    center = (10**cx, 1.0)
    before = axis_lines(log_view(10**log_scale, center), margin_px=100.0)
    after = axis_lines(log_view(10**log_scale * 1.002, center), margin_px=100.0)
    for first, second in ((before, after), (after, before)):
        for key, (position, total, major) in first.items():
            if not RECT.left + 20 <= position <= RECT.right - 20:
                continue
            if key not in second:
                assert total < 0.1, (key, total)
                continue
            _, other_total, other_major = second[key]
            assert abs(total - other_total) < 0.1, (key, total, other_total)
            assert abs(major - other_major) < 0.1, (key, major, other_major)


def visible_label_opacities(view: View) -> dict[str, float]:
    """Map the text of every label fully inside the plane to its opacity."""
    measurer = FakeMeasurer()
    result = {}
    for item in build_grid(view, LIGHT, measurer, GridState()).labels:
        size = measurer.measure_text(item.text, item.font)
        if (
            item.left >= RECT.left
            and item.left + size.width <= RECT.right
            and item.top - size.overflow_top >= RECT.top
            and item.top + size.height <= RECT.bottom
        ):
            result[item.text.plain] = max(item.opacity, result.get(item.text.plain, 0))
    return result


@settings(max_examples=200, deadline=None)
@given(cx=centres, cy=centres, log_scale=st.floats(0.5, 6.0))
def test_zooming_never_pops_log_labels(cx: float, cy: float, log_scale: float) -> None:
    center = (10**cx, 10**cy)
    view = log_view(10**log_scale, center)
    zoomed = log_view(10**log_scale * 1.002, center)
    before = visible_label_opacities(view)
    after = visible_label_opacities(zoomed)
    for text in before.keys() & after.keys():
        assert abs(before[text] - after[text]) < 0.15, (text, before[text], after[text])
    before_all = {i.text.plain for i in scene_for(view).labels}
    after_all = {i.text.plain for i in scene_for(zoomed).labels}
    for text, opacity in before.items():
        if text not in after_all:
            assert opacity < 0.15, text
    for text, opacity in after.items():
        if text not in before_all:
            assert opacity < 0.15, text
