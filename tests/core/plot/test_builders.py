"""Tests for turning plotted objects into primitives (:mod:`core.plot.builders`)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pygridplot.core.coords import ScreenRect
from pygridplot.core.plot.builders import (
    area_fill,
    arrow_fill,
    function_stroke,
    implicit_stroke,
    parametric_stroke,
    point_markers,
    polar_stroke,
    polygon_shapes,
    polyline_stroke,
    region_fill,
    slope_field_strokes,
    vector_field_fills,
)
from pygridplot.core.plot.implicit import ImplicitSampling
from pygridplot.core.plot.style import (
    ArrowStyle,
    FillStyle,
    LineStyle,
    PointStyle,
    PolygonStyle,
)
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


# ----------------------------------------------------------------------
# Implicit curves
# ----------------------------------------------------------------------
def test_implicit_circle_is_accurate_and_clipped() -> None:
    view = make_view(center=(3.0, 0.0))  # the circle is cut by the left edge
    stroke = implicit_stroke(view, lambda x, y: x**2 + y**2 - 16, LINE, "red")
    assert stroke is not None
    assert stroke.color == "red"
    assert stroke.width == 3.0
    for path in stroke.paths:
        assert inside_margin(path, 1.5 + 2.0 + 1e-6)
    x, y = view.to_world(*all_points(stroke).T)
    inside = (x > view.bounds.x_min + 0.1) & (y > view.bounds.y_min + 0.1)
    assert np.abs(np.hypot(x, y)[inside] - 4).max() * 50 < 0.01


def test_implicit_level_selects_the_curve() -> None:
    view = make_view()
    radius_squared = lambda x, y: x**2 + y**2  # noqa: E731
    by_level = all_points(implicit_stroke(view, radius_squared, LINE, "r", level=4))
    by_offset = all_points(
        implicit_stroke(view, lambda x, y: x**2 + y**2 - 4, LINE, "r")
    )
    np.testing.assert_allclose(by_level, by_offset)
    with pytest.raises(ValueError, match="level"):
        implicit_stroke(view, radius_squared, LINE, "r", level=math.nan)
    with pytest.raises(ValueError, match="level"):
        implicit_stroke(view, radius_squared, LINE, "r", level=[[1.0, 2.0]])
    several = implicit_stroke(view, radius_squared, LINE, "r", level=[1, 4, 9])
    assert several is not None
    x, y = view.to_world(*all_points(several).T)
    radius = np.hypot(x, y)
    for r in (1, 2, 3):  # each circle is there
        assert np.abs(radius - r).min() < 1e-3


def test_implicit_curves_accept_scalar_and_raising_functions() -> None:
    view = make_view()
    # math.sqrt raises for x < 0, and takes only numbers.
    stroke = implicit_stroke(view, lambda x, y: math.sqrt(x) - y, LINE, "r")
    x, y = view.to_world(*all_points(stroke).T)
    assert x.min() >= 0
    assert np.abs(np.sqrt(x) - y)[x > 0.5].max() * 50 < 0.1

    def broken(x: object, y: object) -> float:
        raise ZeroDivisionError

    assert implicit_stroke(view, broken, LINE, "r") is None
    assert implicit_stroke(view, lambda x, y: "text", LINE, "r") is None


def test_implicit_curves_follow_log_axes() -> None:
    view = log_view()
    # y = x^2 is a straight line on log-log axes (and its other branches,
    # at negative x, are not visible there).
    stroke = implicit_stroke(view, lambda x, y: y - x**2, LINE, "red")
    assert straightness(all_points(stroke)) < 0.01
    # The fine grid is spaced evenly on screen, not in x.
    gaps = np.hypot(*np.diff(stroke.paths[0], axis=0).T)  # type: ignore[union-attr]
    assert gaps.max() < 3 * 2.0


def test_implicit_panning_moves_strokes_rigidly() -> None:
    func = lambda x, y: x**3 + y**3 - 6 * x * y + 1  # noqa: E731
    view = make_view(center=(0.37, -0.21))
    before = all_points(implicit_stroke(view, func, LINE, "red"))
    view.pan_pixels(11.0, -6.0)
    after = all_points(implicit_stroke(view, func, LINE, "red"))
    inner = (
        (before[:, 0] > 60)
        & (before[:, 0] < 740)
        & (before[:, 1] > 60)
        & (before[:, 1] < 540)
    )
    moved = {tuple(p) for p in np.round(before[inner] + (11.0, -6.0), 6)}
    assert moved <= {tuple(p) for p in np.round(after, 6)}


def test_implicit_sampling_settings_are_used() -> None:
    view = make_view()
    circle = lambda x, y: x**2 + y**2 - 9  # noqa: E731
    fine = implicit_stroke(view, circle, LINE, "r")
    coarse = implicit_stroke(
        view, circle, LINE, "r", sampling=ImplicitSampling(resolution_px=8.0)
    )
    assert fine is not None
    assert coarse is not None
    assert len(all_points(coarse)) < len(all_points(fine)) / 3


def test_invisible_implicit_curves_give_no_stroke() -> None:
    circle = lambda x, y: (x - 1e3) ** 2 + y**2 - 1  # noqa: E731
    assert implicit_stroke(make_view(), circle, LINE, "r") is None
    empty = View(ScreenRect(0, 0, 0, 0))
    assert implicit_stroke(empty, lambda x, y: x - y, LINE, "r") is None


def test_implicit_curves_stay_correct_at_extreme_zoom() -> None:
    view = make_view(center=(1.0, 1.0))
    view.zoom(1e9)
    stroke = implicit_stroke(view, lambda x, y: x**2 + y**2 - 2, LINE, "r")
    points = all_points(stroke)
    # Tangent to the circle at (1, 1): the line x + y = 2, through the centre.
    assert straightness(points) < 0.01
    cx, cy = 400.0, 300.0
    distance = np.abs((points[:, 0] - cx) - (points[:, 1] - cy)) / math.sqrt(2)
    assert distance.max() < 0.05


# ----------------------------------------------------------------------
# Areas between graphs, and regions
# ----------------------------------------------------------------------
FILL = FillStyle(opacity=0.4)


def all_rings(fill: object) -> np.ndarray:
    """Return every vertex of a fill as one ``(N, 2)`` array."""
    assert fill is not None
    return np.concatenate(fill.rings)  # type: ignore[attr-defined]


def test_area_fill_follows_both_graphs() -> None:
    view = make_view()
    fill = area_fill(view, np.sin, lambda x: x - 5, FILL, "red", domain=(1.0, 3.0))
    assert fill is not None
    assert fill.color == "red"
    assert fill.opacity == 0.4
    assert fill.rule == "nonzero"
    points = all_rings(fill)
    xs = view.to_world_x(points[:, 0])
    ys = view.to_world_y(points[:, 1])
    assert xs.min() == pytest.approx(1.0, abs=1e-9)
    assert xs.max() == pytest.approx(3.0, abs=1e-9)
    # Every vertex lies on one of the two graphs.
    on_graph = np.minimum(np.abs(ys - np.sin(xs)), np.abs(ys - (xs - 5)))
    assert on_graph.max() < 1e-9


def test_area_fill_is_clipped_to_the_plane() -> None:
    fill = area_fill(make_view(), lambda x: 1e6 * x, lambda x: -1e9 + 0 * x, FILL, "r")
    assert inside_margin(all_rings(fill), 2.0 + 1e-6)


def test_area_fill_takes_constants() -> None:
    view = make_view()
    band = area_fill(view, 1, -1.0, FILL, "r")
    ys = view.to_world_y(all_rings(band)[:, 1])
    assert sorted({round(float(y), 9) for y in ys}) == [-1.0, 1.0]
    below = area_fill(view, np.sin, -math.inf, FILL, "r")
    assert all_rings(below)[:, 1].max() >= 599.5  # reaches the bottom edge
    assert area_fill(view, np.sin, math.nan, FILL, "r") is None
    with pytest.raises(TypeError, match="function of x or a number"):
        area_fill(view, np.sin, "0", FILL, "r")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="function of x or a number"):
        area_fill(view, np.sin, True, FILL, "r")


def test_area_fill_stops_at_jumps() -> None:
    view = make_view()
    fill = area_fill(view, np.tan, 0.0, FILL, "r", domain=(0.5, 2.5))
    assert fill is not None
    assert len(fill.rings) == 2  # either side of the asymptote at pi / 2
    pole = view.to_screen_x(math.pi / 2)
    left, right = (view.to_world_x(r[:, 0]) for r in fill.rings)
    assert left.max() < math.pi / 2 < right.min()
    assert pole - view.to_screen_x(left.max()) < 1.0  # close to the pole


def test_area_fill_on_log_axes() -> None:
    view = log_view()
    # y = x^2 is straight on log-log axes; 0 lies infinitely far below.
    fill = area_fill(view, lambda x: x**2, 0.0, FILL, "r", domain=(0.1, 10.0))
    points = all_rings(fill)
    assert points[:, 1].max() >= 599.5
    upper = points[points[:, 1] < 590]
    assert straightness(upper) < 0.01


def test_invisible_areas_give_no_fill() -> None:
    view = make_view()
    assert area_fill(view, np.sin, np.cos, FILL, "r", domain=(100, 200)) is None
    assert area_fill(view, np.sin, np.cos, FillStyle(opacity=0), "r") is None
    assert area_fill(View(ScreenRect(0, 0, 0, 0)), np.sin, 0, FILL, "r") is None
    with pytest.raises(ValueError, match="domain"):
        area_fill(view, np.sin, 0, FILL, "r", domain=(1.0, 1.0))


def test_region_fill_is_accurate() -> None:
    view = make_view()
    fill = region_fill(view, lambda x, y: x**2 + y**2 < 4, FILL, "blue")
    assert fill is not None
    assert fill.rule == "evenodd"
    assert fill.color == "blue"
    (ring,) = fill.rings
    radius_px = np.hypot(ring[:, 0] - 400.0, ring[:, 1] - 300.0)
    assert np.abs(radius_px - 100.0).max() < 0.15


def test_region_fill_accepts_scalar_and_raising_conditions() -> None:
    view = make_view()

    def quadrant(x: float, y: float) -> bool:
        if x < 0:
            raise ValueError
        return x > 0 and y > 0  # Python's `and`: single numbers only

    fill = region_fill(view, quadrant, FILL, "r")
    points = all_rings(fill)
    assert points[:, 0].min() == pytest.approx(400.0, abs=0.15)
    assert points[:, 1].max() == pytest.approx(300.0, abs=0.15)


def even_odd(points: np.ndarray, rings: tuple[np.ndarray, ...]) -> np.ndarray:
    """Classify screen points by the even-odd rule against rings."""
    edges = np.concatenate(
        [np.column_stack((r, np.roll(r, -1, axis=0))) for r in rings]
    )
    px, py = points[:, 0:1], points[:, 1:2]
    x0, y0, x1, y1 = (edges[:, k][None, :] for k in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing_x = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
    crossings = ((y0 > py) != (y1 > py)) & (crossing_x > px)
    return np.asarray(crossings.sum(axis=1) % 2 == 1)


def test_region_fill_follows_log_axes() -> None:
    view = log_view()
    fill = region_fill(view, lambda x, y: y < x**2, FILL, "r")
    assert fill is not None
    rng = np.random.default_rng(3)
    points = np.column_stack((rng.uniform(0, 800, 4000), rng.uniform(0, 600, 4000)))
    x, y = view.to_world_x(points[:, 0]), view.to_world_y(points[:, 1])
    # Away from the boundary (more than a pixel from it, on screen), the fill
    # matches the condition exactly. On log-log axes the boundary is the
    # straight line log y = 2 log x, so the distance is easy to measure.
    distance_px = np.abs(np.log10(y) - 2 * np.log10(x)) / math.sqrt(5)
    distance_px *= view.scale_x
    away = distance_px > 1.0
    assert np.array_equal(even_odd(points, fill.rings)[away], (y < x**2)[away])


def test_region_panning_moves_fills_rigidly() -> None:
    condition = lambda x, y: x**3 + y**3 - 6 * x * y + 1 < 0  # noqa: E731
    view = make_view(center=(0.37, -0.21))
    before = all_rings(region_fill(view, condition, FILL, "r"))
    view.pan_pixels(11.0, -6.0)
    after = all_rings(region_fill(view, condition, FILL, "r"))
    inner = (
        (before[:, 0] > 60)
        & (before[:, 0] < 740)
        & (before[:, 1] > 60)
        & (before[:, 1] < 540)
    )
    moved = {tuple(p) for p in np.round(before[inner] + (11.0, -6.0), 6)}
    assert moved <= {tuple(p) for p in np.round(after, 6)}


def test_invisible_regions_give_no_fill() -> None:
    view = make_view()
    assert region_fill(view, lambda x, y: x > 1e3, FILL, "r") is None
    assert region_fill(view, lambda x, y: x > 0, FillStyle(opacity=0), "r") is None
    empty = View(ScreenRect(0, 0, 0, 0))
    assert region_fill(empty, lambda x, y: x > 0, FILL, "r") is None


# ----------------------------------------------------------------------
# Arrows and fields
# ----------------------------------------------------------------------
ARROW = ArrowStyle(width=2.0, head_size=10.0)


def tips_and_tails(fill: object) -> tuple[np.ndarray, np.ndarray]:
    """Return each arrow's tip and the middle of its tail, in pixels."""
    assert fill is not None
    rings = np.array(fill.rings)  # type: ignore[attr-defined]
    return rings[:, 3], (rings[:, 0] + rings[:, 6]) / 2


def test_arrows_run_between_world_points() -> None:
    view = make_view()
    fill = arrow_fill(view, (0, 0), (2, 1), ARROW, "red")
    assert fill is not None
    assert fill.rule == "nonzero"
    assert fill.color == "red"
    tips, tails = tips_and_tails(fill)
    np.testing.assert_allclose(tips, [view.to_screen(2.0, 1.0)])
    np.testing.assert_allclose(tails, [view.to_screen(0.0, 0.0)])


def test_one_end_is_shared_by_many() -> None:
    view = make_view()
    fill = arrow_fill(view, (0, 0), [(1, 0), (0, 1), (-1, 0)], ARROW, "r")
    tips, tails = tips_and_tails(fill)
    assert len(tips) == 3
    np.testing.assert_allclose(tails, np.tile(view.to_screen(0.0, 0.0), (3, 1)))
    back = arrow_fill(view, [(1, 0), (0, 1)], (0, 0), ARROW, "r")
    assert len(tips_and_tails(back)[0]) == 2
    with pytest.raises(ValueError, match="starts and"):
        arrow_fill(view, [(0, 0), (1, 1)], [(1, 0), (0, 1), (2, 2)], ARROW, "r")


def test_arrows_skip_nan_and_hidden_arrows() -> None:
    view = make_view()
    starts = [(0, 0), (math.nan, 0), (1000, 1000), (1, 1)]
    ends = [(1, 0), (1, 1), (1001, 1000), (1, 1)]  # one hidden, one of zero length
    fill = arrow_fill(view, starts, ends, ARROW, "r")
    assert len(tips_and_tails(fill)[0]) == 1
    assert arrow_fill(view, (1000, 0), (1001, 0), ARROW, "r") is None
    assert arrow_fill(view, (0, 0), (1, 0), ArrowStyle(opacity=0), "r") is None
    assert arrow_fill(View(ScreenRect(0, 0, 0, 0)), (0, 0), (1, 0), ARROW, "r") is None


def test_arrows_crossing_the_plane_are_kept() -> None:
    fill = arrow_fill(make_view(), (-1000, 0), (1000, 0), ARROW, "r")
    assert fill is not None  # both ends far outside, but it crosses the plane


def test_arrows_follow_log_axes_end_by_end() -> None:
    view = log_view()
    fill = arrow_fill(view, (1, 1), (100, 10), ARROW, "r")
    tips, _ = tips_and_tails(fill)
    np.testing.assert_allclose(tips, [view.to_screen(100.0, 10.0)])
    # An end at zero lies infinitely far away: the arrow runs off the plane.
    to_zero = arrow_fill(view, (1, 1), (0, 1), ARROW, "r")
    assert to_zero is not None
    assert tips_and_tails(to_zero)[0][0, 0] < -1e6


FIELD = ArrowStyle(width=1.5)


def field_arrows(fills: tuple[object, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return the tips and tail middles of every arrow of a field."""
    tips, tails = zip(*(tips_and_tails(fill) for fill in fills), strict=True)
    return np.concatenate(tips), np.concatenate(tails)


def test_vector_field_arrows_point_along_the_field() -> None:
    view = make_view()
    fills = vector_field_fills(view, lambda x, y: (-y, x), FIELD, "r")
    assert fills
    tips, tails = field_arrows(fills)
    centres = (tips + tails) / 2
    along = tips - tails
    radial = centres - (400.0, 300.0)
    # A rotation: every arrow is at right angles to the radius, anticlockwise
    # (on screen, with y down, the cross product is negative).
    moving = np.hypot(*radial.T) > 1
    cosine = np.sum(along * radial, axis=1) / (np.hypot(*along.T) * np.hypot(*radial.T))
    assert np.abs(cosine[moving]).max() < 1e-9
    cross = radial[:, 0] * along[:, 1] - radial[:, 1] * along[:, 0]
    assert (cross[moving] < 0).all()


def test_vector_field_lengths() -> None:
    view = make_view(scale=50.0)  # a resting zoom: no arrows fading in
    same = vector_field_fills(view, lambda x, y: (x, y), FIELD, "r")
    assert len(same) == 1
    tips, tails = field_arrows(same)
    np.testing.assert_allclose(np.hypot(*(tips - tails).T), 0.8 * 40.0)
    scaled = vector_field_fills(view, lambda x, y: (x, y), FIELD, "r", scale=10.0)
    tips, tails = field_arrows(scaled)
    centres = (tips + tails) / 2
    magnitude = np.hypot(*((centres - (400.0, 300.0)) / 50.0).T)
    expected = np.minimum(10.0 * magnitude, 0.8 * 40.0)
    np.testing.assert_allclose(np.hypot(*(tips - tails).T), expected, atol=1e-9)


def test_arrows_fading_in_are_shorter_and_fainter() -> None:
    view = make_view(scale=74.0)  # most of the way through an octave
    fills = vector_field_fills(view, lambda x, y: (1 + 0 * x, 0 * y), FIELD, "r")
    assert len(fills) == 2  # the full arrows and the fading ones
    full, fading = fills
    assert full.opacity == 1.0
    assert 0 < fading.opacity < 1
    lengths = [np.hypot(*np.subtract(*tips_and_tails(f)).T) for f in fills]
    np.testing.assert_allclose(lengths[1], lengths[0][0] * fading.opacity)


def test_vector_fields_skip_undefined_and_zero_vectors() -> None:
    view = make_view()

    def partly(x: float, y: float) -> tuple[float, float]:
        if x < 0:
            raise ValueError
        return (0.0, 0.0) if y < 0 else (1.0, math.sqrt(y))  # scalar-only

    tips, _ = field_arrows(vector_field_fills(view, partly, FIELD, "r"))
    assert (tips[:, 0] > 400 - 40).all()  # nothing left of x = 0
    assert (tips[:, 1] < 300 + 40).all()  # nor below y = 0
    assert vector_field_fills(view, lambda x, y: (np.nan, y), FIELD, "r") == ()


def test_vector_fields_move_rigidly_when_panning() -> None:
    view = make_view(center=(0.37, -0.21), scale=61.0)
    field = lambda x, y: (np.sin(y), np.cos(x))  # noqa: E731
    before, _ = field_arrows(vector_field_fills(view, field, FIELD, "r"))
    view.pan_pixels(13.0, -7.0)
    after, _ = field_arrows(vector_field_fills(view, field, FIELD, "r"))
    inner = (before[:, 0] > 60) & (before[:, 0] < 740)
    inner &= (before[:, 1] > 60) & (before[:, 1] < 540)
    moved = {tuple(p) for p in np.round(before[inner] + (13.0, -7.0), 6)}
    assert moved <= {tuple(p) for p in np.round(after, 6)}


def test_vector_field_settings_are_validated() -> None:
    view = make_view()
    field = lambda x, y: (x, y)  # noqa: E731
    with pytest.raises(ValueError, match="spacing"):
        vector_field_fills(view, field, FIELD, "r", spacing=2.0)
    for bad in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError, match="scale"):
            vector_field_fills(view, field, FIELD, "r", scale=bad)


def test_slope_fields_have_the_slope_of_the_equation() -> None:
    view = make_view()
    strokes = slope_field_strokes(view, lambda x, y: x - y, LINE, "r")
    assert len(strokes) == 1
    segments = np.array(strokes[0].paths)
    centres = segments.mean(axis=1)
    x, y = view.to_world(centres[:, 0], centres[:, 1])
    dx, dy = (segments[:, 1] - segments[:, 0]).T
    np.testing.assert_allclose(-dy / dx, x - y, atol=1e-9)  # y is down on screen
    np.testing.assert_allclose(np.hypot(dx, dy), 0.6 * 40.0)


def test_infinite_slopes_are_vertical() -> None:
    view = make_view()
    strokes = slope_field_strokes(view, lambda x, y: 1 / (x * 0), LINE, "r")
    segments = np.array(strokes[0].paths)
    dx = segments[:, 1, 0] - segments[:, 0, 0]
    assert np.abs(dx).max() < 1e-9


def test_slope_fields_are_tangent_to_solutions_on_log_axes() -> None:
    # dy/dx = y / x has the solutions y = c x: lines of slope 1 on log-log.
    view = log_view()
    strokes = slope_field_strokes(view, lambda x, y: y / x, LINE, "r")
    segments = np.concatenate([np.array(s.paths) for s in strokes])
    dx, dy = (segments[:, 1] - segments[:, 0]).T
    np.testing.assert_allclose(-dy / dx, 1.0)


def test_invisible_fields_give_nothing() -> None:
    empty = View(ScreenRect(0, 0, 0, 0))
    assert vector_field_fills(empty, lambda x, y: (x, y), FIELD, "r") == ()
    assert slope_field_strokes(empty, lambda x, y: x, LINE, "r") == ()
    hidden = LineStyle(opacity=0.0)
    assert slope_field_strokes(make_view(), lambda x, y: x, hidden, "r") == ()
