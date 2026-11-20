"""Tests for grid line generation and classification (:mod:`ticks`)."""

from __future__ import annotations

import numpy as np
import pytest

from pygridplot.core.coords import ScreenRect
from pygridplot.core.nice import NiceStep
from pygridplot.core.spacing import SpacingChoice
from pygridplot.core.ticks import (
    GridLevel,
    LineClass,
    base_step,
    classify,
    compute_axis_ticks,
    grid_levels,
    index_range,
)
from pygridplot.core.view import View


def test_index_range_uses_ceil_and_floor() -> None:
    assert index_range(-1.0, 1.0, 0.5) == (-2, 2)
    assert index_range(0.1, 0.9, 0.5) == (1, 1)
    assert index_range(0.6, 0.9, 0.5) == (2, 1)  # empty


def test_grid_levels_expand_a_blend() -> None:
    single = grid_levels(SpacingChoice(NiceStep(0)), show_minor=True)
    assert single == (GridLevel(NiceStep(0), NiceStep(-2), 1.0),)
    blended = grid_levels(
        SpacingChoice(NiceStep(0), NiceStep(1), 0.3), show_minor=False
    )
    assert [lv.major for lv in blended] == [NiceStep(0), NiceStep(1)]
    assert [lv.minor for lv in blended] == [None, None]
    assert [lv.weight for lv in blended] == pytest.approx([0.7, 0.3])


def test_base_step_divides_everything() -> None:
    levels = grid_levels(SpacingChoice(NiceStep(0), NiceStep(1), 0.5), show_minor=True)
    # Steps involved: 1, 0.2, 2, 0.5 -> common divisor 0.1.
    assert str(base_step(levels)) == "0.1"
    with pytest.raises(ValueError, match="level"):
        base_step(())


def test_classify_uses_integer_divisibility() -> None:
    base = NiceStep(-3)  # 0.1
    level = GridLevel(NiceStep(0), NiceStep(-2), 1.0)  # major 1, minor 0.2
    indices = np.arange(-10, 11, dtype=np.int64)
    classes = classify(indices, base, level)
    for n, cls in zip(indices.tolist(), classes.tolist(), strict=True):
        if n % 10 == 0:
            assert cls == LineClass.MAJOR
        elif n % 2 == 0:
            assert cls == LineClass.MINOR
        else:
            assert cls == LineClass.NONE


def test_compute_axis_ticks_covers_the_visible_range() -> None:
    view = View(ScreenRect(0, 0, 800, 600), scale=50.0)
    ticks = compute_axis_ticks(
        view, "x", SpacingChoice(NiceStep(1)), show_minor=True, margin_px=0.0
    )
    assert str(ticks.base) == "0.5"
    values = ticks.indices * ticks.base.value
    bounds = view.bounds
    assert values.min() >= bounds.x_min
    assert values.max() <= bounds.x_max
    assert values.min() - ticks.base.value < bounds.x_min  # nothing missed
    assert values.max() + ticks.base.value > bounds.x_max
    np.testing.assert_allclose(ticks.positions, view.to_screen_x(values))
    majors = ticks.indices[ticks.classes[0] == LineClass.MAJOR]
    assert (majors % 4 == 0).all()  # major 2 = 4 x minor 0.5


def test_compute_axis_ticks_respects_the_cap() -> None:
    view = View(ScreenRect(0, 0, 800, 600), scale=50.0)
    choice = SpacingChoice(NiceStep(-3))  # major 0.1, minor 0.02: 800 lines
    capped = compute_axis_ticks(view, "x", choice, show_minor=True, max_lines=500)
    assert capped.levels[0].minor is None  # minor lines dropped first
    assert len(capped) <= 500
    empty = compute_axis_ticks(view, "x", choice, show_minor=True, max_lines=10)
    assert len(empty) == 0
