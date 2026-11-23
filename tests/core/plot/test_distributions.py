"""Tests for box and violin plots (``core.plot.distributions`` and their builders)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pygridplot.core.coords import ScreenRect
from pygridplot.core.plot.builders import (
    OUTLIER_RADIUS_PX,
    box_plot_primitives,
    violin_primitives,
)
from pygridplot.core.plot.distributions import (
    VIOLIN_POINTS,
    as_groups,
    box_stats,
    check_bandwidth,
    check_positions,
    check_whisker,
    silverman_bandwidth,
    violin_shape,
)
from pygridplot.core.plot.style import BarStyle
from pygridplot.core.view import View

RECT = ScreenRect(0, 0, 400, 300)
RED = (200, 0, 0)


def test_groups_are_one_or_many() -> None:
    (one,) = as_groups([3, 1, 2])
    assert one.tolist() == [3, 1, 2]
    two = as_groups([[1, 2], [3, 4, 5]])  # of different sizes
    assert [g.tolist() for g in two] == [[1, 2], [3, 4, 5]]
    rows = as_groups(np.arange(6.0).reshape(2, 3))
    assert [g.tolist() for g in rows] == [[0, 1, 2], [3, 4, 5]]
    assert as_groups(np.array([1.0, 2.0]))[0].tolist() == [1, 2]
    with pytest.raises(ValueError, match="samples"):
        as_groups([["a"]])
    with pytest.raises(ValueError, match="samples"):
        as_groups(np.zeros((2, 2, 2)))


def test_positions_and_settings_are_checked() -> None:
    assert check_positions(None, 1, 3).tolist() == [1, 2, 3]
    assert check_positions([5, 9], 0, 2).tolist() == [5, 9]
    with pytest.raises(ValueError, match="2 finite"):
        check_positions([1], 0, 2)
    with pytest.raises(ValueError, match="start"):
        check_positions(None, math.nan, 2)
    assert check_whisker(math.inf) == math.inf
    with pytest.raises(ValueError, match="whisker"):
        check_whisker(-1)
    assert check_bandwidth(None) is None
    with pytest.raises(ValueError, match="bandwidth"):
        check_bandwidth(0)


def test_box_statistics_follow_tukey() -> None:
    samples = np.r_[np.arange(1.0, 10.0), 30.0]  # 1 ... 9 and an outlier
    stats = box_stats(samples)
    assert stats is not None
    q1, median, q3 = np.percentile(samples, [25, 50, 75])
    assert (stats.q1, stats.median, stats.q3) == pytest.approx((q1, median, q3))
    assert stats.low == 1.0
    assert stats.high == 9.0  # 30 is beyond q3 + 1.5 IQR
    assert stats.outliers.tolist() == [30.0]
    every = box_stats(samples, whisker=math.inf)
    assert every is not None
    assert every.high == 30.0
    assert every.outliers.size == 0
    assert box_stats(np.array([math.nan])) is None
    same = box_stats(np.array([2.0, 2.0, 2.0]))
    assert same is not None
    assert same.low == same.q1 == same.median == same.q3 == same.high == 2.0


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=60,
    )
)
def test_boxes_hold_their_samples_in_order(samples: list[float]) -> None:
    stats = box_stats(np.array(samples))
    assert stats is not None
    assert stats.low <= stats.q1 <= stats.median <= stats.q3 <= stats.high
    inside = [s for s in samples if stats.low <= s <= stats.high]
    assert len(inside) + stats.outliers.size == len(samples)
    assert min(samples) <= stats.low
    assert stats.high <= max(samples)


def test_violins_are_densities_over_the_samples() -> None:
    rng = np.random.default_rng(1)
    samples = rng.normal(size=2000)
    shape = violin_shape(samples)
    assert shape is not None
    values, density = shape
    assert values.shape == density.shape == (VIOLIN_POINTS,)
    assert values[0] == samples.min()
    assert values[-1] == samples.max()
    middle = density[np.argmin(np.abs(values))]
    assert middle == pytest.approx(1 / math.sqrt(2 * math.pi), abs=0.03)
    area = float(np.sum((density[1:] + density[:-1]) / 2 * np.diff(values)))
    assert area == pytest.approx(1.0, abs=0.02)  # nearly all: the tails are cut
    assert silverman_bandwidth(samples) == pytest.approx(0.9 * 2000**-0.2, rel=0.1)
    assert violin_shape(np.array([1.0, 1.0])) is None
    assert violin_shape(np.array([1.0])) is None
    wide = violin_shape(np.array([0.0, 1.0]), bandwidth=50.0)  # the kernel's guard
    assert wide is not None


# --- Builders -------------------------------------------------------------------


def test_box_plots_draw_boxes_whiskers_medians_and_outliers() -> None:
    v = View(RECT, (0.5, 15.0), (50.0, 9.0), equal_aspect=False)  # y up to 31
    groups = [np.r_[np.arange(1.0, 10.0), 30.0], np.arange(2.0, 7.0)]
    boxes, lines, medians, outliers = box_plot_primitives(
        v, groups, np.array([0.0, 1.0]), BarStyle(), RED
    )
    assert boxes is not None
    assert lines is not None
    assert medians is not None
    assert outliers is not None
    first = boxes.rings[0]
    assert np.ptp(first[:, 0]) == pytest.approx(0.5 * 50)  # half the gap
    q1, q3 = np.percentile(groups[0], [25, 75])
    assert sorted(set(first[:, 1].tolist())) == pytest.approx(
        [v.to_screen_y(q3), v.to_screen_y(q1)]
    )
    assert medians.width > lines.width
    assert outliers.radius == OUTLIER_RADIUS_PX
    assert outliers.positions[:, 1].tolist() == pytest.approx([v.to_screen_y(30.0)])
    lying = box_plot_primitives(
        v, groups, np.array([0.0, 1.0]), BarStyle(), RED, horizontal=True, width=0.25
    )[0]
    assert lying is not None
    assert np.ptp(lying.rings[0][:, 1]) == pytest.approx(0.25 * 9)  # across y
    assert box_plot_primitives(
        v, [np.array([math.nan])], np.zeros(1), BarStyle(), RED
    ) == (
        None,
        None,
        None,
        None,
    )


def test_violins_are_mirrored_and_as_wide_as_asked() -> None:
    v = View(RECT, (0.0, 0.0), (50.0, 25.0), equal_aspect=False)
    rng = np.random.default_rng(2)
    fill, outline, medians = violin_primitives(
        v, [rng.normal(size=500)], np.array([0.0]), BarStyle(), RED, width=2.0
    )
    assert fill is not None
    assert outline is not None
    assert medians is not None
    ring = fill.rings[0]
    centre = v.to_screen_x(0.0)
    assert ring[:, 0].max() - centre == pytest.approx(centre - ring[:, 0].min())
    assert np.ptp(ring[:, 0]) == pytest.approx(2.0 * 50, rel=1e-6)  # 2 wide at most
    log = View(RECT, (0.0, 10.0), (50.0, 150.0), equal_aspect=False, log_y=True)
    fill, _, _ = violin_primitives(
        log,
        [rng.lognormal(2, 1, 500), [-1.0, -2.0]],
        np.array([0.0, 1.0]),
        BarStyle(),
        RED,
    )
    assert fill is not None
    assert len(fill.rings) == 1  # nothing positive: no violin on a log axis
