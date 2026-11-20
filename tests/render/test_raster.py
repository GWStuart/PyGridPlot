"""Tests for the coverage rasterisers (:mod:`pygridplot.render.raster`).

These run without pygame: rasterisation is pure NumPy. Coverage is checked
against exact geometry (areas, lengths times widths), and the property that
matters most for curve quality is tested directly: splitting a line into
many segments must not change how it looks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pygridplot.render.raster import (
    SparseCoverage,
    fill_coverage,
    marker_coverage,
    stroke_coverage,
)

CLIP = (0, 0, 200, 120)


def as_image(
    coverage: SparseCoverage, size: tuple[int, int] = (200, 120)
) -> np.ndarray:
    """Draw sparse coverage into a dense ``(height, width)`` array."""
    image = np.zeros((size[1], size[0]))
    image[coverage.y, coverage.x] = coverage.alpha
    return image


def stroke(points: list[tuple[float, float]], width: float) -> SparseCoverage:
    """Rasterise one polyline, asserting that something was covered."""
    coverage = stroke_coverage([np.array(points, dtype=float)], width, CLIP)
    assert coverage is not None
    return coverage


@pytest.mark.parametrize("width", [1.0, 2.5, 4.0])
def test_horizontal_line_covers_exactly_its_width(width: float) -> None:
    image = as_image(stroke([(20.0, 60.3), (180.0, 60.3)], width))
    column_sums = image[:, 50:150].sum(axis=0)  # away from the round caps
    np.testing.assert_allclose(column_sums, width, atol=0.02)


def test_diagonal_line_area_equals_length_times_width() -> None:
    a, b = (30.0, 20.0), (170.0, 100.0)
    width = 3.0
    image = as_image(stroke([a, b], width))
    length = math.dist(a, b)
    expected = length * width + math.pi * (width / 2) ** 2  # body plus round caps
    assert image.sum() == pytest.approx(expected, rel=0.02)


def test_pixels_are_unique_and_coverage_is_bounded() -> None:
    coverage = stroke([(10.0, 10.0), (60.0, 80.0), (120.0, 15.0)], 2.5)
    pixels = coverage.y * 1000 + coverage.x
    assert np.unique(pixels).shape == pixels.shape
    assert np.all(coverage.alpha > 0)
    assert np.all(coverage.alpha <= 1)


@settings(max_examples=40, deadline=None)
@given(pieces=st.integers(2, 60), y=st.floats(20, 100), width=st.floats(1.0, 5.0))
def test_splitting_a_line_does_not_change_it(
    pieces: int, y: float, width: float
) -> None:
    """Round joins combined with max: no dark spots where segments meet."""
    xs = np.linspace(20.0, 180.0, pieces + 1)
    whole = as_image(stroke([(20.0, y), (180.0, y + 7.0)], width))
    split = as_image(
        stroke(list(zip(xs, y + 7.0 * (xs - 20) / 160, strict=True)), width)
    )
    np.testing.assert_allclose(split, whole, atol=1e-4)


def test_joins_are_round() -> None:
    image = as_image(stroke([(40.0, 100.0), (100.0, 20.0), (160.0, 100.0)], 6.0))
    # The outside of the sharp corner is a circle of radius 3 around the apex.
    apex_row = image[17]  # 3 pixels above the apex at y = 20
    assert apex_row[100] > 0.4
    assert apex_row[94:97].max() < 0.05


def test_dots_and_thin_lines() -> None:
    dot = as_image(stroke([(50.0, 50.0)], 6.0))
    assert dot.sum() == pytest.approx(math.pi * 9, rel=0.05)
    hairline = as_image(stroke([(20.0, 40.0), (180.0, 40.0)], 0.4))
    assert hairline[:, 60:140].sum(axis=0) == pytest.approx(np.full(80, 0.4), abs=0.02)


def test_clipping_and_empty_input() -> None:
    coverage = stroke_coverage([np.array([[-50.0, 60.0], [400.0, 60.0]])], 3.0, CLIP)
    assert coverage is not None
    assert coverage.x.min() >= 0
    assert coverage.x.max() < 200
    assert stroke_coverage([], 3.0, CLIP) is None
    assert (
        stroke_coverage([np.array([[500.0, 500.0], [600.0, 600.0]])], 3, CLIP) is None
    )
    assert stroke_coverage([np.array([[np.nan, 1.0], [5.0, 5.0]])], 3.0, CLIP) is None


# ----------------------------------------------------------------------
# Fills
# ----------------------------------------------------------------------
def fill_image(rings: list[np.ndarray], rule: str = "nonzero") -> np.ndarray:
    """Rasterise rings into a dense image of the whole clip box."""
    coverage = fill_coverage(rings, rule, CLIP)  # type: ignore[arg-type]
    image = np.zeros((120, 200))
    if coverage is not None:
        h, w = coverage.alpha.shape
        image[coverage.top : coverage.top + h, coverage.left : coverage.left + w] = (
            coverage.alpha
        )
    return image


def test_rectangle_with_fractional_edges() -> None:
    # Pixel centres are integers: this covers 10.25 px horizontally.
    rect = np.array([[20.25, 30.0], [30.5, 30.0], [30.5, 50.0], [20.25, 50.0]])
    image = fill_image([rect])
    row = image[40]
    assert row[20] == pytest.approx(0.25)  # pixel 20 spans [19.5, 20.5)
    assert row[21:30] == pytest.approx(np.ones(9))
    assert row[30] == pytest.approx(1.0)  # pixel 30 spans [29.5, 30.5)
    assert row[31] == pytest.approx(0.0)
    assert image.sum() == pytest.approx(10.25 * 20, rel=1e-6)


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.tuples(st.floats(5, 195), st.floats(5, 115)), min_size=3, max_size=3)
)
def test_triangle_area_is_preserved(points: list[tuple[float, float]]) -> None:
    ring = np.array(points)
    x, y = ring[:, 0], ring[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    total = fill_image([ring]).sum()
    perimeter = sum(math.dist(points[i], points[i - 1]) for i in range(3))
    # Vertical antialiasing uses 5 sub-scanlines: error grows with the edges.
    assert total == pytest.approx(area, abs=0.1 * perimeter + 0.5)


def star() -> np.ndarray:
    """A self-intersecting five-pointed star (a pentagram)."""
    angles = np.pi / 2 + np.arange(5) * 4 * np.pi / 5
    return np.column_stack((100 + 50 * np.cos(angles), 60 - 50 * np.sin(angles)))


def test_fill_rules_differ_in_the_middle_of_a_star() -> None:
    assert fill_image([star()], "nonzero")[60, 100] == pytest.approx(1.0)
    assert fill_image([star()], "evenodd")[60, 100] == pytest.approx(0.0)
    tip = fill_image([star()], "evenodd")[20, 100]  # a point of the star
    assert tip == pytest.approx(1.0)


def test_holes() -> None:
    outer = np.array([[20.0, 20.0], [180.0, 20.0], [180.0, 100.0], [20.0, 100.0]])
    inner = np.array([[60.0, 40.0], [60.0, 80.0], [140.0, 80.0], [140.0, 40.0]])
    image = fill_image([outer, inner], "evenodd")
    assert image[60, 100] == 0.0
    assert image[30, 100] == 1.0
    reversed_hole = fill_image([outer, inner], "nonzero")  # opposite orientation
    assert reversed_hole[60, 100] == 0.0


def test_fill_clipping_and_degenerate_input() -> None:
    big = np.array([[-500.0, -500.0], [700.0, -500.0], [700.0, 700.0], [-500.0, 700.0]])
    coverage = fill_coverage([big], "nonzero", CLIP)
    assert coverage is not None
    assert coverage.alpha.shape == (120, 200)
    assert np.allclose(coverage.alpha, 1.0)
    flat = np.array([[10.0, 10.0], [50.0, 10.0], [90.0, 10.0]])
    assert fill_coverage([flat], "nonzero", CLIP) is None
    assert fill_coverage([], "nonzero", CLIP) is None


# ----------------------------------------------------------------------
# Markers
# ----------------------------------------------------------------------
@pytest.mark.parametrize("shape", ["circle", "square", "diamond"])
def test_solid_markers_have_the_circles_area(shape: str) -> None:
    fill, edge, _ = marker_coverage(shape, 6.0, 0.0)  # type: ignore[arg-type]
    assert fill.sum() == pytest.approx(math.pi * 36, rel=0.03)
    assert edge.sum() == 0.0


def test_marker_phase_shifts_the_centre() -> None:
    fill, _, half = marker_coverage("circle", 4.0, 0.0, (0.25, 0.5))
    rows, cols = np.indices(fill.shape)
    cx = (fill * cols).sum() / fill.sum() - half
    cy = (fill * rows).sum() / fill.sum() - half
    # Distance-field antialiasing approximates area coverage, which biases the
    # centroid of a small circle by about 1/100 px.
    assert cx == pytest.approx(0.25, abs=0.02)
    assert cy == pytest.approx(0.5, abs=0.02)


def test_outline_ring_sits_outside_the_body() -> None:
    plain, _, _ = marker_coverage("circle", 5.0, 0.0)
    body, ring, _ = marker_coverage("circle", 5.0, 2.0)
    assert body.sum() == pytest.approx(plain.sum(), rel=1e-9)  # body keeps its size
    assert np.all(body + ring <= 1.0 + 1e-9)  # they do not overlap
    ring_area = math.pi * (7.0**2 - 5.0**2)
    assert ring.sum() == pytest.approx(ring_area, rel=0.03)


@pytest.mark.parametrize("shape", ["cross", "plus"])
def test_line_markers_fit_in_their_stamp(shape: str) -> None:
    fill, _, half = marker_coverage(shape, 8.0, 0.0)  # type: ignore[arg-type]
    assert fill[0].max() == 0.0  # nothing touches the stamp's border
    assert fill[:, 0].max() == 0.0
    assert fill[half, half] == 1.0


def test_vertical_lines_with_long_pieces_keep_their_width() -> None:
    image = as_image(stroke([(80.4, 5.0), (80.4, 115.0)], 3.0))
    row_sums = image[20:100, :].sum(axis=1)
    np.testing.assert_allclose(row_sums, 3.0, atol=0.02)
