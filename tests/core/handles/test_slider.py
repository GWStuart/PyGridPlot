"""Tests for sliders and their panels (:mod:`pygridplot.core.handles.slider`)."""

from __future__ import annotations

import math
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pygridplot.core.coords import ScreenRect
from pygridplot.core.grid.formatting import NumberFormat
from pygridplot.core.grid.theme import DARK, LIGHT
from pygridplot.core.handles import panel as panels
from pygridplot.core.handles.checkbox import Checkbox
from pygridplot.core.handles.panel import Panel, layout_panels, panel_primitives
from pygridplot.core.handles.slider import WHEEL_FRACTION, Slider
from pygridplot.core.primitives import Fill, LabelText, Markers, Stroke
from tests.conftest import FakeMeasurer

FMT = NumberFormat()
RECT = ScreenRect(0, 0, 800, 600)


def text(slider: Slider[Any]) -> str:
    """The slider's text, as a plain string."""
    return "".join(run.text for run in slider.text(FMT).runs)


# --- Values --------------------------------------------------------------------


def test_values_start_low_and_stay_in_range() -> None:
    slider = Slider("a", (-1.0, 3.0))
    assert slider.value == -1.0
    slider.value = 10
    assert slider.value == 3.0
    slider.value = -5
    assert slider.value == -1.0
    assert Slider("a", (0, 1), value=0.25).value == 0.25


def test_integer_sliders_give_ints() -> None:
    n = Slider("n", (1, 50), value=10, step=1)
    assert n.integer
    assert n.value == 10
    assert isinstance(n.value, int)
    n.value = 12.6
    assert n.value == 13
    assert isinstance(n.value, int)
    evens = Slider("k", (0, 10), step=2)
    evens.value = 5.2
    assert evens.value == 6
    assert not Slider("x", (0, 10), step=1.0).integer  # a float step: floats
    assert isinstance(Slider("x", (0, 10), step=1.0, value=3).value, float)


def test_steps_are_exact_decimals() -> None:
    slider = Slider("a", (0.0, 1.0), step=0.1)
    slider.value = 0.3
    assert slider.value == 0.3  # not 0.30000000000000004
    slider.value = 0.71
    assert slider.value == 0.7
    shifted = Slider("b", (0.05, 1.0), step=0.1)
    shifted.value = 0.36
    assert shifted.value == 0.35  # steps are counted from low
    assert Slider("c", (0.0, 1.0), step=0.3, value=1.0).value == 0.9  # last below high
    # One step across the range, though 1.001 - 1.0 < 0.001 in floats.
    one = Slider("d", (1.0, 1.001), step=0.001, value=1.001)
    assert one.value == 1.001


def test_log_sliders_are_geometric() -> None:
    slider = Slider("r", (1.0, 100.0), log=True)
    assert slider.value_at(0.5) == pytest.approx(10.0)
    slider.value = 10.0
    assert slider.fraction == pytest.approx(0.5)
    whole = Slider("n", (1, 1000), step=1, log=True)
    assert whole.value_at(1 / 3) == 10  # a whole number on a geometric track
    with pytest.raises(ValueError, match="positive"):
        Slider("r", (0.0, 1.0), log=True)


def test_drags_are_rounded_to_a_pixel_of_the_track() -> None:
    slider = Slider("a", (0.0, 5.0))
    slider.track_px = 200.0  # 0.025 per pixel: hundredths
    assert slider.value_at(0.123456) == 0.62
    assert slider.drag_to(0.123456)
    assert slider.value == 0.62
    log = Slider("r", (1.0, 1000.0), log=True)
    log.track_px = 300.0  # 100 px per decade: about 2.3% per pixel
    assert log.value_at(0.5) == pytest.approx(31.6)


def test_the_wheel_moves_by_a_step() -> None:
    n = Slider("n", (0, 10), value=5, step=1)
    assert n.nudge(2)
    assert n.value == 7
    n.nudge(100)
    assert n.value == 10
    assert not n.nudge(1)  # already at the end
    smooth = Slider("a", (0.0, 1.0), value=0.5)
    smooth.track_px = 200.0
    smooth.nudge(-3)
    assert smooth.value == pytest.approx(0.5 - 3 * WHEEL_FRACTION)


def test_moved_is_stable_through_a_frame() -> None:
    slider = Slider("a", (0, 1))
    slider.begin_frame()
    assert not slider.moved
    slider.value = 0.5
    assert not slider.moved  # until the next frame
    slider.begin_frame()
    assert slider.moved
    slider.value = 0.5  # the same value: no change
    slider.begin_frame()
    assert not slider.moved


@pytest.mark.parametrize(
    ("args", "kwargs", "error"),
    [
        ((3, (0, 1)), {}, TypeError),  # label not a string
        (("a", (1, 0)), {}, ValueError),
        (("a", (0, math.inf)), {}, ValueError),
        (("a", (0,)), {}, ValueError),
        (("a", ("0", 1)), {}, TypeError),
        (("a", (0, 1)), {"step": 0}, ValueError),
        (("a", (0, 1)), {"step": 2.0}, ValueError),  # wider than the range
        (("a", (0.5, 3)), {"step": 1}, ValueError),  # an int step needs whole ends
        (("a", (0, 1)), {"step": True}, TypeError),
        (("a", (0, 1)), {"value": math.nan}, ValueError),
        (("a", (0, 1)), {"corner": "middle"}, ValueError),
    ],
)
def test_settings_are_validated(
    args: tuple[Any, ...], kwargs: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        Slider(*args, **kwargs)


def test_values_set_from_code_are_validated() -> None:
    slider = Slider("a", (0, 1))
    with pytest.raises(ValueError, match="NaN"):
        slider.value = math.nan
    with pytest.raises(TypeError, match="number"):
        slider.value = "0.5"  # type: ignore[assignment]


def test_the_text_shows_as_many_digits_as_the_slider_can_set() -> None:
    assert text(Slider("n", (1, 50), value=10, step=1)) == "n = 10"
    assert text(Slider("p", (-3.0, 3.0), value=-1.25, step=0.25)) == "p = −1.25"
    smooth = Slider("a", (0.0, 5.0), value=math.pi)
    smooth.track_px = 200.0
    assert text(smooth) == "a = 3.14"
    smooth.track_px = 20_000.0
    assert text(smooth) == "a = 3.1416"
    assert text(Slider("", (0, 9), value=4, step=1)) == "4"  # the value alone
    assert text(Slider("t", (0.0, 1.0), value=0.375, step=0.125)) == "t = 0.375"
    angle = Slider("φ", (0.0, math.pi), value=math.pi / 3, step=math.pi / 12)
    assert text(angle) == "φ = 1.047"  # π/12 has no last decimal
    sign = Slider("b", (-5, 5), value=-2, step=1).text(FMT)
    assert all(run.role != "sign" for run in sign.runs)  # a minus inside the text


def test_removing_calls_back_once() -> None:
    removed: list[Slider[Any]] = []
    slider = Slider("a", (0, 1), on_remove=removed.append)
    slider.set_interaction(hovered=True, dragging=True)
    slider.remove()
    slider.remove()
    assert removed == [slider]
    assert not slider.hovered
    assert not slider.dragging
    assert "Slider('a'" in repr(slider)


@settings(max_examples=100, deadline=None)
@given(
    low=st.floats(-1e6, 1e6),
    width=st.floats(1e-3, 1e6),
    t=st.floats(0.0, 1.0),
    log=st.booleans(),
    steps=st.one_of(st.none(), st.integers(1, 1000)),
)
def test_dragged_values_are_always_in_range_and_on_the_step(
    low: float, width: float, t: float, log: bool, steps: int | None
) -> None:
    if log:
        low = abs(low) + 1e-3
    high = low + width
    step = None if steps is None else width / steps
    slider = Slider("a", (low, high), step=step, log=log)
    slider.track_px = 200.0
    value = slider.value_at(t)
    assert low <= value <= high
    if step is not None:
        k = (value - low) / step
        # On the step, to the spacing of floats near the values.
        assert abs(k - round(k)) * step <= 4 * math.ulp(max(abs(low), abs(high)))


# --- Panels --------------------------------------------------------------------


def layout(
    *controls: Slider[Any] | Checkbox, rect: ScreenRect = RECT
) -> tuple[Panel, ...]:
    return layout_panels(controls, rect, LIGHT, FakeMeasurer())


def test_panels_sit_in_their_corners_and_stack_in_order() -> None:
    a, b = Slider("a", (0, 1)), Slider("b", (0, 1))
    c = Slider("c", (0, 1), corner="top-right")
    found = {panel.rows[0].slider.corner: panel for panel in layout(a, b, c)}
    left = found["bottom-left"]
    assert left.rect.left == RECT.left + panels.MARGIN_PX
    assert left.rect.bottom == RECT.bottom - panels.MARGIN_PX
    assert [row.slider for row in left.rows] == [a, b]  # top to bottom
    assert left.rows[0].y < left.rows[1].y
    right = found["top-right"]
    assert right.rect.right == RECT.right - panels.MARGIN_PX
    assert right.rect.top == RECT.top + panels.MARGIN_PX
    for panel in found.values():
        for row in panel.rows:
            assert row.end - row.start == panels.TRACK_PX
            assert panel.rect.contains_point(row.start, row.y)
            assert row.slider.track_px == panels.TRACK_PX


def test_rows_map_the_cursor_to_the_track() -> None:
    slider = Slider("a", (0, 10), value=2.5)
    row = layout(slider)[0].rows[0]
    assert row.handle_x == pytest.approx(row.start + 0.25 * (row.end - row.start))
    assert row.fraction_at(row.start - 50) == 0.0
    assert row.fraction_at(row.end + 50) == 1.0
    assert row.fraction_at((row.start + row.end) / 2) == pytest.approx(0.5)
    assert row.contains(row.end, row.y + panels.HANDLE_RADIUS_PX)
    assert not row.contains(row.end, row.y - 30)  # the text above does not grab
    assert row.on_handle(row.handle_x + 3, row.y)
    assert not row.on_handle(row.end, row.y)


def test_small_planes_get_shorter_tracks_or_none() -> None:
    narrow = ScreenRect(0, 0, 150, 300)
    row = layout(Slider("a", (0, 1)), rect=narrow)[0].rows[0]
    assert panels.MIN_TRACK_PX <= row.end - row.start < panels.TRACK_PX
    assert layout(Slider("a", (0, 1)), rect=ScreenRect(0, 0, 60, 300)) == ()


def test_primitives_draw_the_panel_track_and_handle() -> None:
    slider = Slider("a", (0, 1), value=0.5)
    panel = layout(slider)[0]
    shapes, texts = panel_primitives(panel, LIGHT, {slider: (200, 0, 0)})
    assert isinstance(shapes[0], Fill)  # the panel, under everything
    strokes = [s for s in shapes if isinstance(s, Stroke)]
    markers = [s for s in shapes if isinstance(s, Markers)]
    assert len(strokes) == 3  # border, track, filled part
    assert len(markers) == 1
    assert markers[0].positions[0, 0] == pytest.approx(panel.rows[0].handle_x)
    assert isinstance(texts[0].text, LabelText)
    assert texts[0].color == LIGHT.label_color
    slider.set_interaction(hovered=True, dragging=False)
    shapes, _ = panel_primitives(panel, LIGHT, {slider: (200, 0, 0)})
    assert len([s for s in shapes if isinstance(s, Markers)]) == 2  # with the ring


def test_disabled_sliders_are_faded_and_dark_themes_work() -> None:
    slider = Slider("a", (0, 1), enabled=False)
    panel = layout(slider)[0]
    shapes, texts = panel_primitives(panel, DARK, {slider: (200, 0, 0)})
    assert texts[0].opacity == panels.DISABLED_OPACITY
    assert shapes[0].color == DARK.background  # type: ignore[union-attr]
    clear = LIGHT.__class__(background=None, label_halo_width=0)
    shapes, _ = panel_primitives(panel, clear, {slider: (200, 0, 0)})
    assert not any(isinstance(s, Fill) for s in shapes)  # transparent: no panel


# --- Playing -------------------------------------------------------------------


def played(slider: Slider[Any], seconds: float, frames: int) -> list[float]:
    """The values after each of ``frames`` advances of ``seconds``."""
    values = []
    for _ in range(frames):
        slider.advance(seconds)
        values.append(slider.value)
    return values


def test_play_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="play"):
        Slider("t", (0, 1), play="forever")  # type: ignore[arg-type]
    for bad in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError, match="duration"):
            Slider("t", (0, 1), play="loop", duration=bad)
    with pytest.raises(TypeError, match="duration"):
        Slider("t", (0, 1), duration="2")  # type: ignore[arg-type]
    still = Slider("t", (0, 1))
    with pytest.raises(ValueError, match="cannot play"):
        still.playing = True
    still.playing = False  # stopping is always fine
    assert not still.advance(1.0)


def test_looping_starts_again_from_the_low_end() -> None:
    t = Slider("t", (0, 10), play="loop", duration=2)
    assert not t.playing
    assert not t.advance(0.5)  # not playing yet
    t.playing = True
    assert played(t, 0.5, 5) == pytest.approx([2.5, 5.0, 7.5, 0.0, 2.5])
    assert t.playing


def test_bouncing_turns_round_at_both_ends() -> None:
    t = Slider("t", (-1, 1), value=0, play="bounce", duration=1)
    t.playing = True
    values = played(t, 0.25, 10)
    assert values == pytest.approx([0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0, 0.5, 1])
    # One long frame still lands where it should, however many turns: from
    # the top, down the whole track and three quarters of the way back up.
    t.advance(1.75)
    assert t.value == pytest.approx(0.5)


def test_playing_once_stops_at_the_end_and_restarts_from_the_start() -> None:
    t = Slider("t", (0, 1), play="once", duration=1)
    t.playing = True
    assert played(t, 0.4, 3) == pytest.approx([0.4, 0.8, 1.0])
    assert not t.playing
    assert not t.advance(0.4)
    t.playing = True  # at the end: back to the start
    assert t.value == 0.0
    t.value = 0.5
    t.playing = False
    t.playing = True  # not at the end: from where it is
    assert t.value == 0.5


def test_stepped_sliders_give_every_value_an_equal_share() -> None:
    n = Slider("n", (1, 5), step=1, play="loop", duration=5)
    n.playing = True
    n.advance(0.05)  # sample half way between frames, never on a boundary
    values = played(n, 0.1, 50)  # one sweep, five values, a second each
    assert all(isinstance(v, int) for v in values)
    assert [values.count(k) for k in range(1, 6)] == [10, 10, 10, 10, 10]
    tenths = Slider("a", (0, 1), step=0.1, play="once", duration=1.1)
    tenths.playing = True
    assert played(tenths, 0.1, 11)[-3:] == [0.9, 1.0, 1.0]  # exact decimals


def test_unstepped_and_log_sliders_sweep_smoothly_and_geometrically() -> None:
    smooth = Slider("s", (0, 3), play="loop", duration=3)
    smooth.track_px = 200  # drags would round to a pixel of this track
    smooth.playing = True
    smooth.advance(0.123456)
    assert smooth.value == pytest.approx(0.123456)  # not rounded
    r = Slider("r", (1, 1000), log=True, play="once", duration=3)
    r.playing = True
    assert played(r, 1.0, 3) == pytest.approx([10.0, 100.0, 1000.0])


def test_dragging_holds_a_playing_slider_and_it_plays_on_from_there() -> None:
    t = Slider("t", (0, 10), play="loop", duration=10)
    t.playing = True
    t.advance(1.0)
    t.set_interaction(hovered=True, dragging=True)
    t.drag_to(0.6)
    assert not t.advance(1.0)  # held
    assert t.value == 6.0
    t.set_interaction(hovered=False, dragging=False)
    t.advance(1.0)
    assert t.value == pytest.approx(7.0)
    t.value = 2.0  # set from code: plays on from there too
    t.advance(1.0)
    assert t.value == pytest.approx(3.0)
    t.nudge(1)
    t.advance(0.5)
    assert t.value == pytest.approx(3.6, abs=1e-9)  # 1% of the track, then 0.5


def test_playing_changes_count_as_moves_and_play_can_be_taken_away() -> None:
    t = Slider("t", (0, 1), play="bounce")
    t.playing = True
    t.begin_frame()
    t.advance(0.1)
    t.begin_frame()
    assert t.moved
    t.play = None  # no play button any more: stops
    assert not t.playing
    t.play = "loop"
    t.playing = True
    t.remove()
    assert not t.playing


# --- Panels with play buttons and checkboxes ------------------------------------


def test_play_buttons_sit_left_of_their_tracks_and_tracks_end_together() -> None:
    a = Slider("a", (0, 1))
    t = Slider("t", (0, 1), play="loop")
    panel = layout(a, t)[0]
    plain, playing = panel.rows
    assert isinstance(plain, panels.SliderRow)
    assert isinstance(playing, panels.SliderRow)
    assert plain.button is None
    assert playing.button is not None
    bx, by = playing.button
    assert by == playing.y
    assert bx + panels.PLAY_RADIUS_PX < playing.start - panels.HANDLE_RADIUS_PX
    assert plain.end == playing.end  # every track ends at the same place
    assert playing.end - playing.start == panels.TRACK_PX
    assert t.track_px == panels.TRACK_PX
    assert a.track_px == plain.end - plain.start > panels.TRACK_PX
    assert playing.on_button(bx + 3, by - 3)
    assert not playing.on_button(playing.start, playing.y)
    assert not plain.on_button(plain.start, plain.y)
    # Panels without play buttons are as they were.
    (alone,) = layout(Slider("b", (0, 1)))[0].rows
    assert isinstance(alone, panels.SliderRow)
    assert alone.end - alone.start == panels.TRACK_PX


def test_checkboxes_take_rows_in_the_order_given() -> None:
    a = Slider("a", (0, 1))
    box = Checkbox("show it")
    b = Slider("b", (0, 1))
    panel = layout(a, box, b)[0]
    assert [row.control for row in panel.rows] == [a, box, b]
    row = panel.rows[1]
    assert isinstance(row, panels.CheckboxRow)
    left, top, right, bottom = row.box
    assert right - left == bottom - top == panels.BOX_PX
    assert row.text_left > right
    assert row.contains((left + right) / 2, (top + bottom) / 2)  # the box
    assert row.contains(row.text_left + 10, row.text_top + 4)  # the label
    assert not row.contains(row.text_left + 10, panel.rows[0].y)
    assert panel.rect.contains_point(row.text_left, row.text_top)
    assert panel.rows[0].y < top < panel.rows[2].y


def test_play_buttons_show_play_or_pause() -> None:
    t = Slider("t", (0, 1), play="loop")
    panel = layout(t)[0]
    shapes, _ = panel_primitives(panel, LIGHT, {t: (0, 100, 200)})
    signs = [s for s in shapes if isinstance(s, Fill)][1:]  # after the panel
    assert [len(s.rings) for s in signs] == [1]  # a triangle
    assert signs[0].rings[0].shape == (3, 2)
    t.playing = True
    shapes, _ = panel_primitives(layout(t)[0], LIGHT, {t: (0, 100, 200)})
    signs = [s for s in shapes if isinstance(s, Fill)][1:]
    assert [len(s.rings) for s in signs] == [2]  # two bars
    t.set_interaction(hovered=False, dragging=False, play_hovered=True)
    row = layout(t)[0].rows[0]
    rings = [
        s for s in panels.row_primitives(row, LIGHT, "red")[0] if isinstance(s, Markers)
    ]
    assert len(rings) == 3  # the handle, the button's ring and the button
    assert panels.row_key(row, "red") != panels.row_key(
        layout(Slider("t", (0, 1), play="loop"))[0].rows[0], "red"
    )


def test_checkboxes_are_outlined_or_ticked() -> None:
    box = Checkbox("a")
    row = layout(box)[0].rows[0]
    shapes, text = panels.row_primitives(row, LIGHT, (200, 0, 0))
    assert [type(s) for s in shapes] == [Stroke]  # an outline
    assert text.text.plain == "a"
    before = panels.row_key(row, (200, 0, 0))
    box.value = True
    box.set_interaction(hovered=True)
    shapes, _ = panels.row_primitives(row, LIGHT, (200, 0, 0))
    assert [type(s) for s in shapes] == [Fill, Fill, Stroke]  # ring, box, tick
    assert shapes[1].color == (200, 0, 0)
    assert panels.row_key(row, (200, 0, 0)) != before
    box.enabled = False
    shapes, text = panels.row_primitives(row, DARK, (200, 0, 0))
    assert [type(s) for s in shapes] == [Fill, Stroke]  # no ring when disabled
    assert text.opacity == panels.DISABLED_OPACITY
