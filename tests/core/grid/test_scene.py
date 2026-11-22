"""Tests for assembling the full grid (:mod:`pygridplot.core.grid.scene`).

Besides examples, this module holds the property tests of PROJECT_PLAN 8.10:
for random centres and zoom levels the grid must stay well behaved.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pygridplot.core.coords import ScreenRect
from pygridplot.core.grid.layout import PlacedLabel
from pygridplot.core.grid.nice import NiceStep
from pygridplot.core.grid.scene import GridScene, GridState, build_grid
from pygridplot.core.grid.theme import DARK, LIGHT, GridTheme
from pygridplot.core.grid.ticklabels import (
    label_fade,
    label_step,
    next_label_step,
    thinning_opacity,
)
from pygridplot.core.pixels import line_starts
from pygridplot.core.primitives import Orientation, TextItem
from pygridplot.core.view import View

from ...conftest import FakeMeasurer, labels_by_axis

RECT = ScreenRect(0, 0, 800, 600)


def grid_for(
    center: tuple[float, float] = (0.0, 0.0),
    scale: float = 50.0,
    theme: GridTheme = LIGHT,
    rect: ScreenRect = RECT,
) -> GridScene:
    """Build the grid for a simple view with the fake measurer."""
    return build_grid(View(rect, center, scale), theme, FakeMeasurer(), GridState())


def label_texts(scene: GridScene) -> list[str]:
    """Return the plain text of every label in the scene."""
    return [item.text.plain for item in scene.labels]


def as_placed(item: TextItem, measurer: FakeMeasurer) -> PlacedLabel:
    """Rebuild a :class:`PlacedLabel` from a drawn label, for overlap checks."""
    size = measurer.measure_text(item.text, item.font)
    return PlacedLabel(item.text, item.left, item.top, size, item.opacity)


def test_default_view_has_major_lines_every_two_units() -> None:
    scene = grid_for()
    assert str(scene.x_spacing.fine) == "2"
    assert scene.x_spacing.coarse is None
    # Lines are grouped into one minor and one major batch per axis plus axes.
    kinds = [(b.orientation, b.layers[0].color, len(b)) for b in scene.lines]
    vertical_major = [
        k for k in kinds if k[:2] == (Orientation.VERTICAL, LIGHT.major_color)
    ]
    assert vertical_major == [(Orientation.VERTICAL, LIGHT.major_color, 8)]  # +-2..+-8
    assert scene.lines[-1].layers[0].color == LIGHT.axis_color
    assert scene.lines[-2].layers[0].color == LIGHT.axis_color


def test_painter_order_minor_major_axes() -> None:
    scene = grid_for(scale=37.0)
    colors = [b.layers[-1].color for b in scene.lines]
    first_major = colors.index(LIGHT.major_color)
    assert all(c == LIGHT.minor_color for c in colors[:first_major])
    assert colors[-2:] == [LIGHT.axis_color, LIGHT.axis_color]


def test_axis_positions_are_skipped_by_grid_lines() -> None:
    scene = grid_for()
    for batch in scene.lines[:-2]:
        assert not np.any(
            np.isclose(
                batch.positions,
                400.0 if batch.orientation is Orientation.VERTICAL else 300.0,
            )
        )


def test_labels_are_clean_and_zero_appears_once() -> None:
    texts = label_texts(grid_for())
    assert texts.count("0") == 1
    assert {"2", "−2", "8"} <= set(texts)
    assert all("0000" not in t for t in texts)


def test_zero_label_stays_while_one_axis_is_visible() -> None:
    # x-axis far below the plane, y-axis visible: the 0 sits in the pinned
    # row of x labels, labelling the y-axis line.
    scene = grid_for(center=(0.0, 100.0))
    zero = [i for i in scene.labels if i.text.plain == "0"]
    assert len(zero) == 1
    assert zero[0].opacity == 1.0
    assert zero[0].top + 9 <= 600  # pinned inside the plane
    # Both axes off-screen: nothing left to label.
    assert "0" not in label_texts(grid_for(center=(100.0, 100.0)))


def test_y_labels_pin_to_the_left_edge() -> None:
    scene = grid_for(center=(100.0, 0.0))
    y_items = [i for i in scene.labels if i.left <= 10 and i.text.plain != "0"]
    assert y_items


def test_axes_are_omitted_when_off_screen() -> None:
    scene = grid_for(center=(1000.0, 1000.0))
    assert all(b.layers[0].color != LIGHT.axis_color for b in scene.lines)


def test_visibility_switches() -> None:
    bare = replace(
        LIGHT, show_minor=False, show_major=False, show_axes=False, show_labels=False
    )
    scene = grid_for(theme=bare)
    assert scene.lines == ()
    assert scene.labels == ()
    no_major = replace(LIGHT, show_major=False)
    colors = {b.layers[0].color for b in grid_for(theme=no_major).lines}
    assert LIGHT.major_color not in colors


def test_dark_theme_uses_its_own_colours() -> None:
    scene = grid_for(theme=DARK)
    assert scene.background == DARK.background
    assert {i.color for i in scene.labels} == {DARK.label_color}
    assert {i.halo_color for i in scene.labels} == {DARK.background}


def test_empty_rect_gives_empty_scene() -> None:
    scene = grid_for(rect=ScreenRect(0, 0, 0, 0))
    assert scene.lines == ()
    assert scene.labels == ()


def test_cross_fade_blends_paint_and_label_opacity() -> None:
    # Put the zoom right on the 1 <-> 2 switching point (raw spacing sqrt(2)).
    scale = 100.0 / math.sqrt(2.0)
    scene = grid_for(scale=scale)
    choice = scene.x_spacing
    assert choice.coarse is not None
    assert choice.blend == pytest.approx(0.5)
    opacities = {i.text.plain: i.opacity for i in scene.labels}
    assert opacities["2"] == 1.0  # labelled by both levels
    assert "1" not in opacities  # finer-only labels have faded out by the midpoint
    # A little before the midpoint the finer level's own labels are still fading.
    early = grid_for(scale=scale * 10**0.02)
    early_opacities = {i.text.plain: i.opacity for i in early.labels}
    assert 0.0 < early_opacities["1"] < 1.0
    assert early_opacities["2"] == 1.0
    stacked = [b for b in scene.lines if len(b.layers) == 2]
    assert stacked, "lines that are major in one level and minor in the other"
    for batch in stacked:
        assert batch.layers[0].color == LIGHT.minor_color
        assert batch.layers[1].opacity == pytest.approx(0.5)


def test_snap_mode_uses_hysteresis_state() -> None:
    theme = replace(LIGHT, transition="snap", hysteresis=0.04)
    state = GridState()
    view = View(RECT, (0.0, 0.0), 100.0 / 10**0.1505)  # just below 1|2 boundary
    first = build_grid(view, theme, FakeMeasurer(), state).x_spacing.fine
    view.set_scale(100.0 / 10**0.17)  # past the boundary, inside hysteresis
    second = build_grid(view, theme, FakeMeasurer(), state).x_spacing.fine
    assert first == second == NiceStep(0)
    view.set_scale(100.0 / 10**0.2)
    assert build_grid(view, theme, FakeMeasurer(), state).x_spacing.fine == NiceStep(1)


def test_label_fade_is_sequential() -> None:
    assert label_fade(0.0) == 0.0
    assert label_fade(0.5) == 0.0
    assert label_fade(0.75) == pytest.approx(0.5)
    assert label_fade(1.0) == 1.0


def test_label_step_only_uses_nice_multiples() -> None:
    assert label_step(NiceStep(1), 50.0, 60.0) == NiceStep(1)  # 2 -> 100px
    assert str(label_step(NiceStep(1), 20.0, 60.0)) == "10"  # 2 -> 5 is skipped
    assert str(label_step(NiceStep(2), 5.0, 60.0)) == "20"  # 5 -> 10 -> 20


def test_long_labels_are_thinned_out() -> None:
    # Near 1e5 with a 0.01 spacing labels are like "100000.37": 9 chars wide.
    scene = grid_for(center=(100000.37, 0.0), scale=5000.0)
    measurer = FakeMeasurer()
    # The x-axis covers rows 300-301, so its labels' row starts at 302 + gap.
    x_row = 302 + LIGHT.label_gap
    x_labels = sorted(
        (
            as_placed(i, measurer)
            for i in scene.labels
            if i.top == x_row and i.text.plain != "0"
        ),
        key=lambda p: p.left,
    )
    assert len(x_labels) > 2
    for a, b in itertools.pairwise(x_labels):
        gap = b.left - (a.left + a.size.width)
        assert gap >= 4 - 1  # never closer than the hard minimum
        if gap < LIGHT.label_min_separation - 1:  # closer than comfortable:
            assert min(a.opacity, b.opacity) < 1.0  # one of them is fading


# ----------------------------------------------------------------------
# Property tests
# ----------------------------------------------------------------------
centres = st.floats(-1e6, 1e6, allow_nan=False)
log_scales = st.floats(-4.0, 9.0)


@settings(max_examples=150, deadline=None)
@given(cx=centres, cy=centres, log_scale=log_scales)
def test_lines_stay_inside_the_plane(cx: float, cy: float, log_scale: float) -> None:
    scene = grid_for(center=(cx, cy), scale=10**log_scale)
    for batch in scene.lines:
        low, high = (0, 800) if batch.orientation is Orientation.VERTICAL else (0, 600)
        assert np.all(batch.positions > low - 3)
        assert np.all(batch.positions < high + 3)
        assert 0.0 < batch.layers[-1].opacity <= 1.0


@settings(max_examples=150, deadline=None)
@given(cx=centres, cy=centres, log_scale=log_scales)
def test_major_spacing_stays_near_the_target(
    cx: float, cy: float, log_scale: float
) -> None:
    view = View(RECT, (cx, cy), 10**log_scale)
    scale = view.scale  # may have been clamped by the precision limits
    scene = build_grid(view, LIGHT, FakeMeasurer(), GridState())
    for step, weight in scene.x_spacing.weighted_steps():
        px = step.value * scale
        # Nearest rule: within sqrt(2.5) of the target, widened by the band.
        widen = 10**LIGHT.fade_band
        assert 100 / math.sqrt(2.5) / widen - 1e-6 <= px
        assert px <= 100 * math.sqrt(2.5) * widen + 1e-6
        assert 0.0 < weight <= 1.0


@settings(max_examples=150, deadline=None)
@given(cx=centres, cy=centres, log_scale=log_scales)
def test_labels_never_overlap(cx: float, cy: float, log_scale: float) -> None:
    measurer = FakeMeasurer()
    scene = grid_for(center=(cx, cy), scale=10**log_scale)
    placed = [as_placed(item, measurer) for item in scene.labels]
    for i, a in enumerate(placed):
        assert 0.0 < a.opacity <= 1.0
        for b in placed[i + 1 :]:
            # Labels may only share space while one fades out as the other
            # fades in: together they are never more than one opaque label.
            if a.overlaps(b):
                assert a.opacity + b.opacity <= 1.0 + 1e-9, (a.text.plain, b.text.plain)


def snapped_lines(
    scene: GridScene, orientation: Orientation
) -> set[tuple[int, object]]:
    """Return ``(first pixel, top paint colour)`` of every line in one direction."""
    result = set()
    for batch in scene.lines:
        if batch.orientation is orientation:
            for start in line_starts(batch.positions, batch.width).tolist():
                result.add((start, batch.layers[-1].color))
    return result


@settings(max_examples=150, deadline=None)
@given(
    cx=st.floats(-1e4, 1e4),
    cy=st.floats(-1e4, 1e4),
    log_scale=st.floats(-3.0, 7.0),
    dx=st.integers(-40, 40),
    dy=st.integers(-40, 40),
)
def test_panning_by_whole_pixels_moves_every_line_rigidly(
    cx: float, cy: float, log_scale: float, dx: int, dy: int
) -> None:
    view = View(RECT, (cx, cy), 10**log_scale)
    before = build_grid(view, LIGHT, FakeMeasurer())
    # A line of width w is placed by rounding its centre minus w / 2, so its
    # ties lie at half pixels for odd widths and whole pixels for even ones
    # (the 2 px axes). Exact ties may round either way after a pan, as the
    # pan's float round-off decides them.
    near_tie = any(
        np.any(np.abs((b.positions - b.width / 2 + 0.5) % 1.0 - 0.5) < 1e-6)
        for b in before.lines
    )
    assume(not near_tie)
    view.pan_pixels(dx, dy)
    after = build_grid(view, LIGHT, FakeMeasurer())
    for orientation, shift, size in (
        (Orientation.VERTICAL, dx, 800),
        (Orientation.HORIZONTAL, dy, 600),
    ):
        moved = {(p + shift, c) for p, c in snapped_lines(before, orientation)}
        now = snapped_lines(after, orientation)
        inside = range(45, size - 45)  # away from edges, where lines enter/leave
        assert {m for m in moved if m[0] in inside} == {
            n for n in now if n[0] in inside
        }


@settings(max_examples=150, deadline=None)
@given(cx=st.floats(-1e9, 1e9), cy=st.floats(-1e9, 1e9), log_scale=st.floats(-9, 9))
def test_labels_on_one_axis_never_mix_notations(
    cx: float, cy: float, log_scale: float
) -> None:
    scene = grid_for(center=(cx, cy), scale=10**log_scale)
    x_row = {i.top for i in scene.labels}
    for top in x_row:
        same_row = [
            i.text for i in scene.labels if i.top == top and i.text.plain != "0"
        ]
        if len(same_row) > 3:  # a row of x labels
            assert len({t.has_superscript for t in same_row}) == 1


def test_large_offsets_prefer_fixed_notation_when_digits_matter() -> None:
    scene = grid_for(center=(1e6 + 0.123, 2.5), scale=1e6)
    texts = label_texts(scene)
    assert any(t.startswith("1000000.") for t in texts)
    assert not any("×" in t for t in texts)


def test_label_steps_are_nested() -> None:
    steps = [NiceStep(2)]  # 5
    for _ in range(5):
        steps.append(next_label_step(steps[-1]))
    assert [str(s) for s in steps] == ["5", "10", "20", "100", "200", "1000"]
    assert all(b.is_multiple_of(a) for a, b in itertools.pairwise(steps))


def test_thinning_opacity() -> None:
    assert thinning_opacity(50.0, 50.0, 60.0) == 0.0
    assert thinning_opacity(60.0, 50.0, 60.0) == 1.0
    assert thinning_opacity(100.0, 50.0, 60.0) == 1.0
    assert 0.0 < thinning_opacity(55.0, 50.0, 60.0) < 1.0
    assert thinning_opacity(55.0, 50.0, 50.0) == 1.0  # no fade zone


def label_opacities(scene: GridScene) -> dict[tuple[str, str], float]:
    """Map every label fully inside the plane, by axis and text, to its opacity.

    Labels crossing the plane's edge are left out: sliding out of view is
    not a pop.
    """
    measurer = FakeMeasurer()
    inside = {}
    for key, item in labels_by_axis(scene):
        size = measurer.measure_text(item.text, item.font)
        fully_inside = (
            item.left >= RECT.left
            and item.left + size.width <= RECT.right
            and item.top - size.overflow_top >= RECT.top
            and item.top + size.height <= RECT.bottom
        )
        if fully_inside:
            inside[key] = item.opacity
    return inside


def notations(scene: GridScene) -> tuple[bool, bool]:
    """Return whether the x and the y labels use scientific notation."""
    return (
        any(item.text.has_superscript for item in scene.axis_labels.x),
        any(item.text.has_superscript for item in scene.axis_labels.y),
    )


@settings(max_examples=200, deadline=None)
@given(log_scale=st.floats(4.0, 8.0), cy=st.floats(-3.0, 3.0))
def test_zooming_never_pops_labels(log_scale: float, cy: float) -> None:
    """Tiny zoom steps change every label's opacity only a little.

    The view is near x = 12345.678, where labels are long enough for thinning
    to kick in, so this covers cross-fades and continuous thinning together.
    """
    center = (12345.678, cy)
    # Switching an axis between fixed and scientific notation is instant (a
    # known limitation), so frames on either side of a switch are skipped.
    assume(
        notations(grid_for(center=center, scale=10**log_scale))
        == notations(grid_for(center=center, scale=10**log_scale * 1.002))
    )
    before = label_opacities(grid_for(center=center, scale=10**log_scale))
    after = label_opacities(grid_for(center=center, scale=10**log_scale * 1.002))
    for key in before.keys() & after.keys():
        assert abs(before[key] - after[key]) < 0.15, (key, before[key], after[key])
    # A label that is fully inside in one frame but missing (or only partly
    # inside) in the other must be nearly transparent, unless it merely
    # crossed the edge.
    zoomed = grid_for(center=center, scale=10**log_scale * 1.002)
    after_all = {key for key, _ in labels_by_axis(zoomed)}
    before_all = {
        key for key, _ in labels_by_axis(grid_for(center=center, scale=10**log_scale))
    }
    for key, opacity in before.items():
        if key not in after_all:
            assert opacity < 0.15, key
    for key, opacity in after.items():
        if key not in before_all:
            assert opacity < 0.15, key


def test_partly_major_lines_blend_to_the_requested_opacity() -> None:
    from pygridplot.core.grid.gridlines import _paint_for

    def flatten(total: float, major: float) -> tuple[float, float, float]:
        """Return the weights of background, minor and major colour."""
        layers, _, _ = _paint_for(total, major, LIGHT)
        weights = {"bg": 1.0, "minor": 0.0, "major": 0.0}
        for layer in layers:
            name = "minor" if layer.color == LIGHT.minor_color else "major"
            for key in weights:
                weights[key] *= 1.0 - layer.opacity
            weights[name] += layer.opacity
        return weights["bg"], weights["minor"], weights["major"]

    assert flatten(0.5, 0.0) == pytest.approx((0.5, 0.5, 0.0))
    assert flatten(0.6, 0.6) == pytest.approx((0.4, 0.0, 0.6))
    assert flatten(1.0, 0.3) == pytest.approx((0.0, 0.7, 0.3))
    assert flatten(0.8, 0.2) == pytest.approx((0.2, 0.6, 0.2))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minor_width", 0),
        ("axis_width", 0),
        ("target_major_px", 4),
        ("fade_band", -0.1),
        ("hysteresis", -1.0),
        ("label_size", 0),
        ("label_halo_width", -1),
        ("palette", ()),
    ],
)
def test_themes_reject_invalid_settings(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field.split("_")[0]):
        replace(LIGHT, **{field: value})
