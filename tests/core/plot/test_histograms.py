"""Tests for histograms (:mod:`pygridplot.core.plot.histograms` and its builder)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pygridplot.core.coords import ScreenRect
from pygridplot.core.plot.builders import (
    SEPARATOR_FADE_END_PX,
    SEPARATOR_FADE_START_PX,
    histogram_shapes,
)
from pygridplot.core.plot.geometry import polygon_area
from pygridplot.core.plot.histograms import (
    as_samples,
    auto_bin_width,
    bin_by_edges,
    bin_samples,
    check_edges,
    nice_width,
)
from pygridplot.core.plot.style import BarStyle
from pygridplot.core.transforms import AxisTransform
from pygridplot.core.view import View

RECT = ScreenRect(0, 0, 400, 300)
RED = (200, 0, 0)


@pytest.mark.parametrize(
    ("width", "nice"),
    # Nearest on a log scale: between 5 and 10 the middle is √50 ≈ 7.07.
    [(0.13, 0.1), (0.16, 0.2), (0.3, 0.2), (0.34, 0.5), (7.0, 5.0), (7.2, 10.0)],
)
def test_widths_round_to_one_two_five(width: float, nice: float) -> None:
    assert nice_width(width) == pytest.approx(nice)


def test_automatic_widths_follow_the_samples() -> None:
    rng = np.random.default_rng(3)
    normal = rng.normal(size=1000)
    # Freedman-Diaconis: 2 × 1.35 / 10 = 0.27, nearest 0.2.
    assert auto_bin_width(normal) == pytest.approx(0.2)
    assert auto_bin_width(normal * 100) == pytest.approx(20.0)
    mostly_equal = np.r_[np.zeros(100), np.linspace(0, 70, 8)]
    assert auto_bin_width(mostly_equal) == pytest.approx(10.0)  # Sturges
    assert auto_bin_width(np.full(5, 3.0)) == 1.0
    assert auto_bin_width(np.zeros(0)) == 1.0


def test_bins_are_anchored_at_multiples_of_the_width() -> None:
    bins = bin_samples(np.array([0.0, 0.1, 0.49, 0.5, -0.01, 2.2]), 0.5)
    assert bins.index.tolist() == [-1, 0, 1, 4]
    assert bins.left.tolist() == [-0.5, 0.0, 0.5, 2.0]
    assert bins.right.tolist() == [0.0, 0.5, 1.0, 2.5]
    assert bins.counts.tolist() == [1, 3, 1, 1]  # 0.5 is in [0.5, 1)


def test_far_samples_get_bins_of_their_own_cheaply() -> None:
    samples = np.r_[np.linspace(0, 1, 100), 1e12, -1e300]
    bins = bin_samples(samples, 0.01)  # 1e14 bins apart: counted by sorting
    assert bins.counts.sum() == 101  # -1e300 is too far to number exactly
    assert bins.index[-1] == 10**14
    assert bin_samples(np.array([1e300]), 1.0).index.shape == (0,)


@settings(max_examples=100, deadline=None)
@given(
    samples=st.lists(st.floats(-1e3, 1e3), min_size=1, max_size=200),
    extra=st.lists(st.floats(-1e3, 1e3), max_size=50),
    width=st.floats(1e-3, 1e2),
)
def test_adding_samples_never_moves_a_bin(
    samples: list[float], extra: list[float], width: float
) -> None:
    before = bin_samples(np.array(samples), width)
    after = bin_samples(np.array(samples + extra), width)
    assert before.counts.sum() == len(samples)
    assert after.counts.sum() == len(samples) + len(extra)
    # Every bin keeps its edges, and only gains samples.
    found = dict(zip(after.index.tolist(), after.counts.tolist(), strict=True))
    for k, count, left in zip(before.index, before.counts, before.left, strict=True):
        assert found[int(k)] >= count
        assert left == k * width
    assert np.all(np.diff(after.index) > 0)


def test_explicit_edges_count_as_numpy_does() -> None:
    edges = np.array([0.0, 1.0, 2.0, 4.0])
    samples = np.array([0.0, 0.5, 1.0, 3.9, 4.0, 5.0, -1.0])
    bins = bin_by_edges(samples, edges, AxisTransform())
    assert bins.index.tolist() == [0, 1, 2]
    assert bins.counts.tolist() == [2, 1, 2]  # 4.0 is in the last bin
    assert bins.right.tolist() == [1.0, 2.0, 4.0]
    log = bin_by_edges(
        np.array([0.5, 5.0]), np.array([0.0, 1.0, 10.0]), AxisTransform("log")
    )
    assert log.index.tolist() == [1]  # [0, 1) cannot be shown on a log axis
    assert log.left.tolist() == [0.0]
    assert log.right.tolist() == [1.0]


def test_edges_and_samples_are_checked() -> None:
    assert check_edges(None) is None
    assert check_edges([0, 1, 3]).tolist() == [0, 1, 3]  # type: ignore[union-attr]
    for bad in ([1], [0, 0, 1], [0, math.inf], [[0, 1]], ["a", "b"]):
        with pytest.raises(ValueError, match="edges"):
            check_edges(bad)
    assert as_samples([[1, 2], [3, 4]]).tolist() == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="samples"):
        as_samples(["x"])


# --- The builder ---------------------------------------------------------------


def shapes(samples: object, **kwargs: object) -> tuple:
    v = kwargs.pop("view", View(RECT, (0.0, 1.0), (50.0, 50.0)))
    style = kwargs.pop("style", BarStyle())
    return histogram_shapes(v, np.asarray(samples, float), style, RED, **kwargs)  # type: ignore[arg-type]


def test_neighbouring_bins_are_one_shape_with_lines_between() -> None:
    v = View(RECT, (0.0, 1.0), (50.0, 50.0))
    fill, outline, lines = shapes([0.2, 1.1, 1.2, 1.3, 3.5], width=1.0, view=v)
    assert fill is not None
    assert outline is not None
    assert lines is not None
    run, _ = fill.rings  # [0, 2) together, [3, 4) on its own
    x0, x2 = v.to_screen_x(0.0), v.to_screen_x(2.0)
    assert run[:, 0].min() == pytest.approx(x0)
    assert run[:, 0].max() == pytest.approx(x2)
    assert run[:, 1].min() == pytest.approx(v.to_screen_y(3.0))  # the tallest
    assert len(outline.paths) == 2
    # One line between the two neighbours, up to the lower one's top.
    (between,) = lines.paths
    assert between[:, 0].tolist() == pytest.approx([v.to_screen_x(1.0)] * 2)
    assert sorted(between[:, 1].tolist()) == pytest.approx(
        [v.to_screen_y(1.0), v.to_screen_y(0.0)]
    )
    assert lines.opacity == 1.0  # 50 px bins


def test_lines_between_bins_fade_as_bins_narrow() -> None:
    samples = np.linspace(0, 1.999, 200)
    for scale, opacity in ((50.0, 1.0), (6.0, 0.6), (2.0, None)):
        v = View(RECT, (0.0, 0.0), (scale, 50.0), equal_aspect=False)
        _, _, lines = shapes(samples, width=1.0, view=v)
        if opacity is None:
            assert lines is None
        else:
            assert lines is not None
            expected = (scale - SEPARATOR_FADE_START_PX) / (
                SEPARATOR_FADE_END_PX - SEPARATOR_FADE_START_PX
            )
            assert lines.opacity == pytest.approx(min(expected, 1.0))


def test_density_bars_have_area_one() -> None:
    rng = np.random.default_rng(5)
    samples = rng.normal(size=500)
    # Every bar on screen: x from -4 to 4, y from -0.05 to 0.55.
    v = View(RECT, (0.0, 0.25), (50.0, 500.0), equal_aspect=False)
    fill, _, _ = shapes(samples, density=True, width=0.5, view=v)
    assert fill is not None
    area = sum(abs(polygon_area(ring)) for ring in fill.rings) / (50.0 * 500.0)
    assert area == pytest.approx(1.0)


def test_density_with_explicit_edges_uses_the_samples_in_them() -> None:
    v = View(RECT, (0.0, 0.0), (50.0, 50.0))
    fill, _, _ = shapes([0.5, 1.5, 1.5, 9.0], edges=[0, 1, 2], density=True, view=v)
    assert fill is not None
    (ring,) = fill.rings
    # Three samples in the edges: densities 1/3 and 2/3.
    assert sorted(set(ring[:, 1].tolist())) == pytest.approx(
        [v.to_screen_y(2 / 3), v.to_screen_y(1 / 3), v.to_screen_y(0.0)]
    )


def test_bins_off_the_plane_are_left_out() -> None:
    v = View(RECT, (0.0, 0.0), (50.0, 50.0))
    fill, _, _ = shapes(np.r_[np.zeros(3), 1e6], width=1.0, view=v)
    assert fill is not None
    assert len(fill.rings) == 1
    assert shapes([1e6], width=1.0, view=v) == (None, None, None)
    assert shapes([math.nan, math.inf], view=v) == (None, None, None)
    assert shapes([], view=v) == (None, None, None)


def test_histograms_on_log_axes() -> None:
    v = View(RECT, (10.0, 10.0), 150.0, log_x=True, log_y=True)
    samples = np.r_[np.full(30, 2.0), np.full(3, 20.0), -1.0, 0.0]
    fill, _, lines = shapes(samples, width=0.5, view=v)  # half a decade
    assert fill is not None
    for ring in fill.rings:
        assert np.ptp(ring[:, 0]) == pytest.approx(75.0)  # equal on screen
        assert ring[:, 1].max() >= RECT.bottom - 0.5  # from 0: the bottom edge
    assert lines is None  # the bins are not neighbours
    # A line between neighbours stops at the lower bar, with 0 at -inf.
    _, _, lines = shapes(np.r_[np.full(30, 2.0), np.full(3, 4.0)], width=0.5, view=v)
    assert lines is not None
    (between,) = lines.paths
    assert between[:, 1].min() == pytest.approx(v.to_screen_y(3.0))


@settings(max_examples=60, deadline=None)
@given(
    cx=st.floats(-0.5, 0.5),
    dx=st.integers(-20, 20),
    dy=st.integers(-20, 20),
)
def test_histograms_move_rigidly(cx: float, dx: int, dy: int) -> None:
    samples = np.array([-1.2, -0.3, 0.1, 0.2, 0.4, 0.9, 1.5])
    v = View(RECT, (cx, 1.0), (50.0, 50.0))
    before = shapes(samples, width=0.5, view=v)
    v.pan_pixels(dx, dy)
    after = shapes(samples, width=0.5, view=v)
    for a, b in zip(before, after, strict=True):
        assert a is not None
        assert b is not None
        first = a.rings if hasattr(a, "rings") else a.paths
        second = b.rings if hasattr(b, "rings") else b.paths
        for p, q in zip(first, second, strict=True):
            np.testing.assert_allclose(q, p + np.array([dx, dy]), atol=1e-6)


def test_cumulative_histograms_rise_through_the_gaps() -> None:
    v = View(RECT, (2.0, 2.0), (50.0, 50.0))
    samples = [0.2, 0.4, 1.5, 3.5]  # bins [0, 1), [1, 2), then [3, 4)
    fill, outline, _ = shapes(samples, width=1.0, cumulative=True, view=v)
    assert fill is not None
    assert outline is not None
    (ring,) = fill.rings  # one shape: the gap at [2, 3) holds at 3
    tops = sorted({round(float(v.to_world(0, y)[1]), 6) for y in ring[:, 1].tolist()})
    assert tops == [0.0, 2.0, 3.0, 4.0]
    fraction, _, _ = shapes(samples, width=1.0, cumulative=True, density=True, view=v)
    assert fraction is not None
    top = min(fraction.rings[0][:, 1].tolist())
    assert v.to_world(0, top)[1] == pytest.approx(1.0)  # all the samples: 1
