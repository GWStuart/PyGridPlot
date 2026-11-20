"""Tests for turning plotted objects into primitives (:mod:`core.plotting`)."""

from __future__ import annotations

import numpy as np
import pytest

from pygridplot.core.coords import ScreenRect
from pygridplot.core.plot.builders import (
    function_stroke,
    parametric_stroke,
    point_markers,
    polar_stroke,
    polygon_shapes,
    polyline_stroke,
)
from pygridplot.core.plot.style import LineStyle, PointStyle, PolygonStyle
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


# ----------------------------------------------------------------------
# Logarithmic axes
# ----------------------------------------------------------------------
def log_view(*, log_x: bool = True, log_y: bool = True, **kwargs: object) -> View:
    """Return an 800x600 view with log axes."""
    return View(RECT, log_x=log_x, log_y=log_y, **kwargs)  # type: ignore[arg-type]


def all_points(stroke: object) -> np.ndarray:
    """Return every vertex of a stroke as one ``(N, 2)`` array."""
    assert stroke is not None
    return np.concatenate(stroke.paths)  # type: ignore[attr-defined]


def straightness(points: np.ndarray) -> float:
    """Return the largest distance in pixels from the best straight line."""
    slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
    return float(np.max(np.abs(points[:, 1] - (slope * points[:, 0] + intercept))))


def test_power_laws_are_straight_on_log_log_axes() -> None:
    view = log_view()
    for power in (-1.0, 0.5, 2.0, 3.0):
        points = all_points(function_stroke(view, lambda x, k=power: x**k, LINE, "red"))
        assert straightness(points) < 1e-6
        slope = np.polyfit(points[:, 0], points[:, 1], 1)[0]
        assert slope == pytest.approx(-power)  # screen y points down


def test_exponentials_are_straight_on_a_semi_log_axis() -> None:
    view = log_view(log_x=False, scale=(50.0, 150.0), equal_aspect=False)
    points = all_points(function_stroke(view, np.exp, LINE, "red"))
    assert straightness(points) < 1e-6


def test_zero_on_a_log_axis_sends_the_curve_off_the_edge() -> None:
    view = log_view(log_x=False, scale=(50.0, 150.0), equal_aspect=False)
    stroke = function_stroke(view, lambda x: x**2, LINE, "red")
    assert stroke is not None
    assert len(stroke.paths) == 2  # broken at x = 0, where log(0) is -inf
    for path in stroke.paths:
        assert path[:, 1].max() > 599.0  # both branches reach the bottom edge


def test_negative_values_leave_gaps_on_a_log_axis() -> None:
    view = log_view(log_x=False, scale=(50.0, 150.0), equal_aspect=False)
    stroke = function_stroke(view, np.sin, LINE, "red")
    assert stroke is not None
    for path in stroke.paths:  # only the humps where sin(x) > 0 remain
        xs = view.to_world_x(path[:, 0])
        assert np.all(np.sin(xs[1:-1]) > -1e-9)


def test_domain_on_a_log_axis_drops_non_positive_values() -> None:
    view = log_view()
    full = all_points(function_stroke(view, lambda x: x, LINE, "red"))
    clipped = all_points(
        function_stroke(view, lambda x: x, LINE, "red", domain=(-5, 10))
    )
    assert clipped[:, 0].min() == pytest.approx(full[:, 0].min())
    assert clipped[:, 0].max() == pytest.approx(view.to_screen_x(10.0))
    assert function_stroke(view, lambda x: x, LINE, "red", domain=(-5, 0)) is None


def test_panning_a_log_view_moves_function_strokes_rigidly() -> None:
    view = log_view(center=(3.0, 3.0))
    before = all_points(function_stroke(view, np.sqrt, LINE, "red"))
    view.pan_pixels(7.0, -4.0)
    after = all_points(function_stroke(view, np.sqrt, LINE, "red"))
    inner = (before[:, 0] > 100) & (before[:, 0] < 700)
    moved = before[inner] + np.array([7.0, -4.0])
    assert np.isin(np.round(moved[:, 0], 6), np.round(after[:, 0], 6)).all()


def test_points_and_shapes_are_transformed_vertex_by_vertex() -> None:
    view = log_view()
    markers = point_markers(
        view, [(10.0, 100.0), (-1.0, 1.0), (1.0, 0.0)], PointStyle(), "red"
    )
    assert markers is not None
    assert markers.positions.tolist() == [[550.0, 0.0]]  # the others cannot be shown
    stroke = polyline_stroke(view, [(1.0, 1.0), (10.0, 10.0)], LINE, "red")
    assert stroke is not None
    (path,) = stroke.paths
    assert path.tolist() == [[400.0, 300.0], [550.0, 150.0]]  # straight on screen


# ----------------------------------------------------------------------
# Parametric and polar curves
# ----------------------------------------------------------------------
def test_parametric_circle_has_the_right_radius_on_screen() -> None:
    view = make_view()
    stroke = parametric_stroke(view, np.cos, np.sin, (0, 2 * np.pi), LINE, "red")
    points = all_points(stroke)
    assert np.hypot(points[:, 0] - 400, points[:, 1] - 300) == pytest.approx(50.0)


def test_parametric_components_may_be_numbers() -> None:
    stroke = parametric_stroke(make_view(), 2.0, lambda t: t, (-1, 1), LINE, "red")
    points = all_points(stroke)
    assert np.all(points[:, 0] == 500.0)  # the vertical line x = 2
    assert points[:, 1].min() == pytest.approx(250.0)
    assert points[:, 1].max() == pytest.approx(350.0)
    with pytest.raises(TypeError, match="function"):
        parametric_stroke(make_view(), "x", 1.0, (0, 1), LINE, "red")  # type: ignore[arg-type]


def test_curve_ranges_are_validated() -> None:
    with pytest.raises(ValueError, match="t_range"):
        parametric_stroke(make_view(), np.cos, np.sin, (1, 0), LINE, "red")
    with pytest.raises(ValueError, match="theta_range"):
        polar_stroke(make_view(), 1.0, (0, np.inf), LINE, "red")


def test_polar_curves_use_standard_conventions() -> None:
    view = make_view()
    circle = all_points(polar_stroke(view, 2.0, (0, 2 * np.pi), LINE, "red"))
    assert np.hypot(circle[:, 0] - 400, circle[:, 1] - 300) == pytest.approx(100.0)
    # theta = 0 points along +x; a negative radius lies on the opposite side.
    start = all_points(polar_stroke(view, 2.0, (0, 0.1), LINE, "red"))[0]
    assert start == pytest.approx((500.0, 300.0))
    opposite = all_points(polar_stroke(view, -2.0, (0, 0.1), LINE, "red"))[0]
    assert opposite == pytest.approx((300.0, 300.0))
    # Angles grow anticlockwise: a quarter turn reaches +y (up on screen).
    quarter = all_points(polar_stroke(view, 2.0, (0, np.pi / 2), LINE, "red"))[-1]
    assert quarter == pytest.approx((400.0, 200.0))


def test_polar_curves_skip_undefined_angles() -> None:
    stroke = polar_stroke(
        make_view(), lambda t: np.sqrt(np.cos(2 * t)), (0, 2 * np.pi), LINE, "red"
    )
    assert stroke is not None
    assert len(stroke.paths) == 3  # the two lobes, one split by the range's start


def test_curves_follow_log_axes() -> None:
    view = log_view()
    # x = e^t, y = e^2t is the line y = x^2: straight on log-log axes.
    stroke = parametric_stroke(
        view, np.exp, lambda t: np.exp(2 * t), (-3, 3), LINE, "red"
    )
    assert straightness(all_points(stroke)) < 1e-6


def test_invisible_curves_give_no_stroke() -> None:
    far = parametric_stroke(
        make_view(), lambda t: 1e3 + np.cos(t), np.sin, (0, 6), LINE, "red"
    )
    assert far is None
    assert (
        parametric_stroke(
            View(ScreenRect(0, 0, 0, 0)), np.cos, np.sin, (0, 6), LINE, "red"
        )
        is None
    )
