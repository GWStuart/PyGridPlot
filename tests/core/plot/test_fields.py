"""Tests for the lattice of vector and slope fields (:mod:`core.plot.fields`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pygridplot.core.coords import ScreenRect
from pygridplot.core.plot.fields import (
    FADE_START,
    MIN_SPACING,
    FieldLattice,
    check_spacing,
    field_lattice,
    lattice_step,
    screen_directions,
)
from pygridplot.core.view import View

RECT = ScreenRect(0, 0, 800, 600)
SPACING = 40.0


def lattice(view: View, spacing: float = SPACING) -> FieldLattice:
    """Return a view's lattice, which must exist."""
    nodes = field_lattice(view, spacing, 20.0)
    assert nodes is not None
    return nodes


@settings(max_examples=200, deadline=None)
@given(scale=st.floats(1e-6, 1e9), spacing=st.floats(MIN_SPACING, 500))
def test_steps_are_powers_of_two_spaced_within_an_octave(
    scale: float, spacing: float
) -> None:
    step, fade = lattice_step(scale, spacing)
    mantissa, _ = math.frexp(step)
    assert mantissa == 0.5  # an exact power of two
    assert spacing / 2 * (1 - 1e-12) <= step * scale < spacing * (1 + 1e-12)
    assert 0.0 <= fade <= 1.0
    octave = math.log2(2 * step * scale / spacing)
    if octave < FADE_START:
        assert fade == 0.0  # new nodes only appear in the second half


def test_new_nodes_fade_in_smoothly_through_the_octave() -> None:
    fades = [
        lattice_step(spacing * 2.0 ** (t - 1), spacing)[1]
        for t in (0.5, 0.75)
        for spacing in (SPACING,)
    ]
    assert fades[0] == pytest.approx(0.0, abs=1e-12)
    assert fades[1] == pytest.approx(0.5)
    just_before = lattice_step(SPACING * (1 - 1e-9), SPACING)
    assert just_before[1] == pytest.approx(1.0, abs=1e-6)


def visible(
    nodes: FieldLattice, threshold: float = 1e-6
) -> dict[tuple[float, float], float]:
    """Map each visible node's world position (rounded) to its fade."""
    shown = nodes.fade > threshold
    return {
        (round(float(x), 9), round(float(y), 9)): float(f)
        for x, y, f in zip(
            nodes.x[shown], nodes.y[shown], nodes.fade[shown], strict=True
        )
    }


@pytest.mark.parametrize("log", [False, True])
def test_zooming_through_a_halving_changes_nothing(log: bool) -> None:
    """Where the step halves, the visible arrows are the same on both sides."""
    view = View(RECT, (0.3, 0.7) if not log else (2.0, 2.0), 50.0, log_x=log, log_y=log)
    step, _ = lattice_step(view.scale_x, SPACING)
    boundary = SPACING / step  # the scale at which step * scale == SPACING
    view.set_scale(boundary * (1 - 1e-9))
    before = visible(lattice(view))
    view.set_scale(boundary * (1 + 1e-9))
    after = visible(lattice(view))
    assert before.keys() == after.keys()
    for key, fade in after.items():
        assert fade == pytest.approx(before[key], abs=1e-6)


def test_density_stays_within_an_octave_on_screen() -> None:
    for scale in (0.01, 3.0, 50.0, 77.0, 1e6):
        view = View(RECT, (0.0, 0.0), scale)
        nodes = lattice(view)
        full = nodes.fade == 1.0
        columns = np.unique(np.round(nodes.px[full], 6))
        gaps = np.diff(columns)
        assert np.allclose(gaps, gaps[0])
        assert SPACING * (1 - 1e-9) <= gaps[0] < 2 * SPACING


def test_the_lattice_covers_the_plane_and_the_margin() -> None:
    view = View(RECT, (0.37, -1.2), 50.0)
    nodes = field_lattice(view, SPACING, 30.0)
    assert nodes is not None
    assert nodes.px.min() >= -0.5 - 30.0
    assert nodes.px.max() <= 799.5 + 30.0
    full = nodes.fade == 1.0  # the full nodes are at most 2 * SPACING apart
    assert nodes.px[full].min() < -0.5 + 2 * SPACING
    assert nodes.py[full].max() > 599.5 - 2 * SPACING


@settings(max_examples=40, deadline=None)
@given(dx=st.integers(-300, 300), dy=st.integers(-300, 300))
def test_panning_moves_the_lattice_rigidly(dx: int, dy: int) -> None:
    view = View(RECT, (0.31, -0.17), 63.0)
    before = lattice(view)
    view.pan_pixels(dx, dy)
    after = lattice(view)
    moved = {
        (round(float(px + dx), 6), round(float(py + dy), 6))
        for px, py in zip(before.px, before.py, strict=True)
        if 100 < px + dx < 700 and 100 < py + dy < 500
    }
    now = {
        (round(float(px), 6), round(float(py), 6))
        for px, py in zip(after.px, after.py, strict=True)
    }
    assert moved <= now


def test_log_axes_lattices_are_even_on_screen() -> None:
    view = View(RECT, log_x=True, log_y=True)
    nodes = lattice(view)
    columns = np.unique(np.round(nodes.px, 6))
    assert np.allclose(np.diff(columns), np.diff(columns)[0])
    assert (nodes.x > 0).all()
    assert (nodes.y > 0).all()


@pytest.mark.parametrize(
    ("center", "scale"), [((1e15, -3e14), 1e3), ((1e6, 0.0), 1e14), ((0.0, 0.0), 1e-9)]
)
def test_lattices_stay_exact_at_the_limits_of_the_view(
    center: tuple[float, float], scale: float
) -> None:
    view = View(RECT, center, scale)  # clamped to what the view allows
    nodes = lattice(view)
    for pixels in (nodes.px, nodes.py):
        lines = np.unique(pixels)
        assert np.allclose(np.diff(lines), np.diff(lines)[0], atol=1e-6)
    assert field_lattice(View(ScreenRect(0, 0, 0, 0)), SPACING, 0.0) is None


def test_spacing_is_validated() -> None:
    assert check_spacing(25) == 25.0
    for bad in (MIN_SPACING / 2, 0.0, -40.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="spacing"):
            check_spacing(bad)


def test_directions_on_linear_axes() -> None:
    view = View(RECT, (0.0, 0.0), 50.0)
    x = np.zeros(5)
    u = np.array([1.0, 0.0, -3.0, 0.0, math.nan])
    v = np.array([0.0, 2.0, 4.0, 0.0, 1.0])
    d = screen_directions(view, x, x, u, v)
    np.testing.assert_allclose(d[:3], [[1.0, 0.0], [0.0, -1.0], [-0.6, -0.8]])
    assert np.isnan(d[3:]).all()  # zero and undefined vectors have no direction


def test_directions_follow_unequal_and_log_axes() -> None:
    stretched = View(RECT, (0.0, 0.0), (50.0, 100.0), equal_aspect=False)
    d = screen_directions(stretched, np.zeros(1), np.zeros(1), np.ones(1), np.ones(1))
    np.testing.assert_allclose(d, [[1 / math.sqrt(5), -2 / math.sqrt(5)]])
    # On log-log axes, the field (x, 2y) is tangent to y = x², which is the
    # straight line of slope 2 everywhere.
    loglog = View(RECT, log_x=True, log_y=True)
    x = np.array([0.01, 1.0, 300.0])
    y = np.array([5.0, 0.2, 1e4])
    d = screen_directions(loglog, x, y, x, 2 * y)
    np.testing.assert_allclose(
        d, np.tile([1 / math.sqrt(5), -2 / math.sqrt(5)], (3, 1))
    )


def test_huge_vectors_still_have_a_direction() -> None:
    view = View(RECT, (0.0, 0.0), 50.0)
    zero = np.zeros(2)
    d = screen_directions(
        view, zero, zero, np.array([1e300, 3e307]), np.array([1e300, 4e307])
    )
    np.testing.assert_allclose(d, [[1 / math.sqrt(2), -1 / math.sqrt(2)], [0.6, -0.8]])
    infinite = screen_directions(
        view, zero[:1], zero[:1], np.array([math.inf]), np.ones(1)
    )
    assert np.isnan(infinite).all()
