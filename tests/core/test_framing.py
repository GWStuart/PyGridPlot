"""Tests for framing a view (:mod:`pygridplot.core.framing`)."""

from __future__ import annotations

import pytest

from pygridplot.core.coords import ScreenRect
from pygridplot.core.framing import (
    ViewFraming,
    aspect_choice,
    equal_aspect_for,
    validate_ranges,
)
from pygridplot.core.view import View

EMPTY = ScreenRect(0, 0, 0, 0)
REAL = ScreenRect(0, 0, 1000, 500)


def test_aspect_rules() -> None:
    assert aspect_choice(None, (0, 1), (0, 1)) is False  # both ranges: fill each
    assert aspect_choice(None, (0, 1), None) is None
    assert aspect_choice(True, (0, 1), (0, 1)) is True  # explicit wins
    assert equal_aspect_for(None, log_x=False, log_y=False)
    assert equal_aspect_for(None, log_x=True, log_y=True)
    assert not equal_aspect_for(None, log_x=True, log_y=False)
    assert equal_aspect_for(True, log_x=True, log_y=False)


def unsettled(rect: ScreenRect = EMPTY) -> tuple[View, ViewFraming]:
    """A view of an automatic plane that has not been drawn yet."""
    view = View(rect, equal_aspect=False)
    return view, ViewFraming(view, False, settled=False)


def test_fits_wait_for_the_real_size() -> None:
    view, framing = unsettled()
    framing.fit(((0, 10), (0, 5)), home=True)
    assert not framing.settled
    view.rect = REAL
    framing.settle()
    assert framing.settled
    assert view.scale_x == pytest.approx(499.5 / 5)
    assert framing.home == view.state()


def test_a_provisional_fit_is_redone_at_the_real_size() -> None:
    view, framing = unsettled(ScreenRect(0, 0, 1, 1))  # e.g. a 1x1 display
    framing.fit(((0, 10), None), home=True)
    view.rect = REAL
    framing.settle()
    assert view.bounds.x_min <= 0 < 10 <= view.bounds.x_max * (1 + 1e-12)
    assert view.scale_x == pytest.approx(499.5 / 5)


def test_moving_the_view_before_settling_keeps_the_move() -> None:
    view, framing = unsettled(ScreenRect(0, 0, 1, 1))
    framing.fit(((0, 10), None), home=True)
    view.pan_world(100.0, 0.0)
    view.rect = REAL
    framing.settle()
    assert view.center_x == pytest.approx(105.0)  # not refitted
    probe = view.copy()
    probe.restore(framing.home)
    assert probe.center_x == pytest.approx(5.0)  # but home is the fit


def test_an_explicit_home_replaces_a_pending_one() -> None:
    view, framing = unsettled()
    framing.fit(((0, 10), None), home=True)
    chosen = View(REAL, (3.0, 4.0), 20.0).state()
    framing.home = chosen
    view.rect = REAL
    framing.settle()
    assert framing.home == chosen


def test_set_log_refits_and_becomes_home() -> None:
    view = View(REAL, (5.0, 5.0))
    framing = ViewFraming(view, None, settled=True)
    assert framing.set_log("y", log=True)
    assert view.log_y
    assert not view.equal_aspect  # mixed axes
    assert framing.home == view.state()
    assert not framing.set_log("y", log=True)  # nothing to do


def test_invalid_ranges_raise_and_leave_the_view_alone() -> None:
    view = View(REAL, log_x=True, log_y=True)
    before = view.state()
    framing = ViewFraming(view, None, settled=True)
    with pytest.raises(ValueError, match="positive"):
        framing.fit(((-1, 10), None), home=True)
    assert view.state() == before
    with pytest.raises(ValueError, match="low < high"):
        validate_ranges(view, ((10, 1), None))
