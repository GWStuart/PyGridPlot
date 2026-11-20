"""Tests for turning plotted objects into primitives (:mod:`core.plotting`)."""

from __future__ import annotations

import numpy as np
import pytest

from pygridplot.core.coords import ScreenRect
from pygridplot.core.plotting import (
    function_stroke,
    point_markers,
    polygon_shapes,
    polyline_stroke,
)
from pygridplot.core.style import LineStyle, PointStyle, PolygonStyle
from pygridplot.core.view import View

RECT = ScreenRect(0, 0, 800, 600)
LINE = LineStyle(width=3.0)


def make_view(center: tuple[float, float] = (0.0, 0.0), scale: float = 50.0) -> View:
    """Return an 800x600 view."""
    return View(RECT, center, scale)


def inside_margin(points: np.ndarray, margin: float) -> bool:
    """Whether every point lies within the plane grown by ``margin`` pixels."""
    return bool(
        np.all(points[:, 0] >= -0.5 - margin)
        and np.all(points[:, 0] <= 799.5 + margin)
        and np.all(points[:, 1] >= -0.5 - margin)
        and np.all(points[:, 1] <= 599.5 + margin)
    )


def test_function_stroke_is_clipped_to_the_plane() -> None:
    stroke = function_stroke(make_view(), lambda x: x**3, LINE, "red")
    assert stroke is not None
    assert stroke.color == "red"
    assert stroke.width == 3.0
    for path in stroke.paths:
        assert inside_margin(path, 1.5 + 2.0 + 1e-6)


def test_function_stroke_respects_the_domain() -> None:
    view = make_view()
    stroke = function_stroke(view, np.sin, LINE, "red", domain=(0.0, 2.0))
    assert stroke is not None
    xs = np.concatenate([view.to_world_x(p[:, 0]) for p in stroke.paths])
    assert xs.min() == pytest.approx(0.0, abs=1e-9)
    assert xs.max() == pytest.approx(2.0, abs=1e-9)
    assert function_stroke(view, np.sin, LINE, "red", domain=(100, 200)) is None
    with pytest.raises(ValueError, match="domain"):
        function_stroke(view, np.sin, LINE, "red", domain=(2.0, 1.0))


def test_invisible_function_gives_no_stroke() -> None:
    assert function_stroke(make_view(), lambda x: x + 1000, LINE, "red") is None


def test_panning_moves_function_strokes_rigidly() -> None:
    view = make_view(center=(0.3, 0.1))
    before = function_stroke(view, np.sin, LINE, "red")
    view.pan_pixels(37, -12)
    after = function_stroke(view, np.sin, LINE, "red")
    assert before is not None
    assert after is not None
    old = np.concatenate(before.paths) + np.array([37.0, -12.0])
    new = np.concatenate(after.paths)
    # Every interior vertex of the old curve is also a vertex of the new one.
    interior = old[(old[:, 0] > 60) & (old[:, 0] < 740)]
    lookup = {(round(x, 6), round(y, 6)) for x, y in new.tolist()}
    hits = [(round(x, 6), round(y, 6)) in lookup for x, y in interior.tolist()]
    assert all(hits)


def test_polyline_stroke_clips_and_closes() -> None:
    view = make_view()
    stroke = polyline_stroke(view, [(0, 0), (1, 0), (1, 1)], LINE, "blue", closed=True)
    assert stroke is not None
    (path,) = stroke.paths
    np.testing.assert_allclose(path[0], path[-1])  # closed
    far = polyline_stroke(view, [(100, 100), (101, 100)], LINE, "blue")
    assert far is None


def test_polygon_shapes_fill_and_outline_switches() -> None:
    view = make_view()
    triangle = [(0, 0), (2, 0), (1, 2)]
    fill, outline = polygon_shapes(view, triangle, PolygonStyle(), "green")
    assert fill is not None
    assert outline is not None
    assert fill.color == "green"
    assert fill.opacity == 0.25
    no_fill, _ = polygon_shapes(view, triangle, PolygonStyle(fill=False), "green")
    assert no_fill is None
    _, no_line = polygon_shapes(view, triangle, PolygonStyle(width=0), "green")
    assert no_line is None
    custom, _ = polygon_shapes(view, triangle, PolygonStyle(fill_color="gold"), "green")
    assert custom is not None
    assert custom.color == "gold"


def test_huge_polygons_are_clipped_to_the_screen() -> None:
    view = make_view(scale=1e9)  # vertices end up ~1e9 pixels away
    fill, outline = polygon_shapes(
        view, [(-1, -1), (1, -1), (1, 1), (-1, 1)], PolygonStyle(), "green"
    )
    assert fill is not None
    assert inside_margin(fill.rings[0], 2.0 + 1e-6)
    assert outline is None  # the square's edges are all far off-screen


def test_point_markers_cull_and_skip_nan() -> None:
    view = make_view()
    points = [(0, 0), (np.nan, 1), (1000, 0), (1, 1)]
    markers = point_markers(view, points, PointStyle(radius=5), "black")
    assert markers is not None
    assert markers.positions.shape == (2, 2)
    np.testing.assert_allclose(markers.positions[0], view.to_screen(0.0, 0.0))
    assert markers.edge_width == 0.0  # no edge colour: no outline
    outlined = point_markers(
        view, [(0, 0)], PointStyle(edge_color="white", edge_width=2), "black"
    )
    assert outlined is not None
    assert outlined.edge_width == 2.0
    assert point_markers(view, [(500, 500)], PointStyle(), "black") is None


def test_empty_rect_draws_nothing() -> None:
    view = View(ScreenRect(0, 0, 0, 0))
    assert function_stroke(view, np.sin, LINE, "red") is None
    assert point_markers(view, [(0, 0)], PointStyle(), "red") is None
