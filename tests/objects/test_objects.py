"""Tests of persistent objects on a plane: drawing, attributes, colours, errors."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import pygridplot as pgp
from pygridplot.objects import PlotObject

from .helpers import make_plane, pixels

# --- Drawing like the plane's methods -----------------------------------------

SCENES: list[tuple[str, Callable[[], list[Any]], Callable[[pgp.Plane, Any], None]]] = [
    (
        "function",
        lambda: [pgp.Function(lambda x: 2 * np.sin(x) + 1, width=4)],
        lambda p, s: p.plot(s, lambda x: 2 * np.sin(x) + 1, width=4),
    ),
    (
        "function with a domain",
        lambda: [pgp.Function(lambda x: x**2 / 4, domain=(-2, 3), opacity=0.5)],
        lambda p, s: p.plot(s, lambda x: x**2 / 4, domain=(-2, 3), opacity=0.5),
    ),
    (
        "parametric",
        lambda: [
            pgp.ParametricCurve(
                lambda t: 2 * np.cos(3 * t), lambda t: 2 * np.sin(2 * t)
            )
        ],
        lambda p, s: p.parametric(
            s,
            lambda t: 2 * np.cos(3 * t),
            lambda t: 2 * np.sin(2 * t),
            (0, 2 * math.pi),
        ),
    ),
    (
        "polar",
        lambda: [pgp.PolarCurve(lambda th: 1 + np.cos(th), (0, math.pi))],
        lambda p, s: p.polar(s, lambda th: 1 + np.cos(th), (0, math.pi)),
    ),
    (
        "implicit contours",
        lambda: [pgp.ImplicitCurve(lambda x, y: x * y, (1, 2, 4), color="black")],
        lambda p, s: p.implicit(s, lambda x, y: x * y, (1, 2, 4), color="black"),
    ),
    (
        "shapes",
        lambda: [
            pgp.Points([(0, 0), (1, 1), (2, 4)], marker="square"),
            pgp.Polyline(np.array([(-3, 0), (-2, 1), (-1, 0)]), closed=True),
            pgp.Polygon([(1, -1), (3, -1), (2, -3)], fill_opacity=0.5),
            pgp.Point((-2, -2), radius=7),
        ],
        lambda p, s: (
            p.points(s, [(0, 0), (1, 1), (2, 4)], marker="square"),
            p.polyline(s, [(-3, 0), (-2, 1), (-1, 0)], closed=True),
            p.polygon(s, [(1, -1), (3, -1), (2, -3)], fill_opacity=0.5),
            p.points(s, [(-2, -2)], radius=7),
        ),
    ),
]


@pytest.mark.parametrize(
    ("make", "by_hand"), [s[1:] for s in SCENES], ids=[s[0] for s in SCENES]
)
def test_objects_draw_like_the_planes_methods(
    make: Callable[[], list[Any]], by_hand: Callable[[pgp.Plane, Any], None]
) -> None:
    """Objects added to a plane draw exactly what the same calls draw by hand."""
    plane, surface = make_plane()
    plane.add(*make())
    plane.draw(surface)
    expected_plane, expected = make_plane()
    by_hand(expected_plane, expected)
    assert pixels(surface) == pixels(expected)
    single_plane, single = make_plane()  # one at a time, with draw()
    for obj in make():
        obj.draw(single_plane, single)
    assert pixels(single) == pixels(expected)


def test_objects_mix_with_immediate_drawing() -> None:
    plane, surface = make_plane()
    plane.add(pgp.Function(np.cos, color="red"))
    plane.draw_grid(surface)
    plane.plot(surface, np.sin, color="blue")  # below the objects
    plane.draw_objects(surface)
    expected_plane, expected = make_plane()
    expected_plane.plot(expected, np.sin, color="blue")
    expected_plane.plot(expected, np.cos, color="red")
    assert pixels(surface) == pixels(expected)


# --- Attributes ---------------------------------------------------------------


def test_attributes_are_read_every_frame() -> None:
    plane, surface = make_plane()
    f = plane.add(pgp.Function(np.sin))
    plane.draw(surface)
    first = pixels(surface)
    f.func = np.cos
    f.color = "red"
    f.width = 5
    plane.draw(surface)
    expected_plane, expected = make_plane()
    expected_plane.plot(expected, np.cos, color="red", width=5)
    assert pixels(surface) == pixels(expected)
    f.visible = False
    plane.draw(surface)
    assert pixels(surface) == pixels(make_plane()[1])
    f.visible = True
    f.func = np.sin
    f.color = None
    f.width = 2.5
    plane.draw(surface)
    assert pixels(surface) == first


def test_data_can_change_every_frame() -> None:
    plane, surface = make_plane()
    line = plane.add(pgp.Polyline(np.zeros((0, 2))))
    for step in range(3):
        xs = np.linspace(-3, 3, 50)
        line.points = np.column_stack((xs, np.sin(xs + step)))
        plane.draw(surface)
        expected_plane, expected = make_plane()
        expected_plane.polyline(expected, np.column_stack((xs, np.sin(xs + step))))
        assert pixels(surface) == pixels(expected)


def test_coordinates_can_be_functions() -> None:
    a = pgp.Point((1, 2))
    plane, surface = make_plane()
    plane.add(pgp.Polyline([(0, 0), (lambda: a.x, lambda: a.y)], color="black"))
    a.position = (2, -1)
    plane.draw(surface)
    expected_plane, expected = make_plane()
    expected_plane.polyline(expected, [(0, 0), (2, -1)], color="black")
    assert pixels(surface) == pixels(expected)


def test_closures_make_objects_depend_on_each_other() -> None:
    a = pgp.Point((1, 2))
    f = pgp.Function(lambda x: a.y * np.sin(x - a.x))
    assert f(1.0) == 0.0
    a.position = (0, 3)
    assert f(math.pi / 2) == pytest.approx(3.0)


# --- Functions are functions ----------------------------------------------------


def test_functions_can_be_called_composed_and_differentiated() -> None:
    f = pgp.Function(lambda x: x**3 / 3)
    g = pgp.Function(lambda x: f(x) * 3)
    xs = np.linspace(-2, 2, 9)
    np.testing.assert_allclose(f(xs), xs**3 / 3)
    np.testing.assert_allclose(g(xs), xs**3)
    np.testing.assert_allclose(f.derivative()(xs), xs**2)
    np.testing.assert_allclose(pgp.derivative(g, 2)(xs), 6 * xs)
    derivative = f.derivative()
    f.func = lambda x: x**2  # the derivative follows the function
    np.testing.assert_allclose(derivative(xs), 2 * xs)


def test_curves_can_be_called() -> None:
    c = pgp.ParametricCurve(np.cos, np.sin)
    assert c(0.0) == (1.0, 0.0)
    r = pgp.PolarCurve(lambda th: 2 + np.cos(th))
    assert r(0.0) == 3.0
    assert pgp.PolarCurve(2)(np.zeros(3)).tolist() == [2.0, 2.0, 2.0]
    assert pgp.ParametricCurve(1, np.sin)(0.0) == (1.0, 0.0)
    F = pgp.ImplicitCurve(lambda x, y: x**2 + y**2)  # noqa: N806 - the usual name
    assert F(3.0, 4.0) == 25.0


# --- The plane's list of objects -------------------------------------------------


def test_adding_and_removing() -> None:
    plane, surface = make_plane()
    f = pgp.Function(np.sin)
    g = pgp.Function(np.cos)
    assert plane.add(f, g) is f
    assert plane.objects == (f, g)
    with pytest.raises(ValueError, match="already on this plane"):
        plane.add(f)
    with pytest.raises(TypeError, match="not an object a plane can draw"):
        plane.add(np.sin)  # type: ignore[arg-type]
    plane.remove(f)
    assert plane.objects == (g,)
    with pytest.raises(ValueError, match="not on this plane"):
        plane.remove(f)
    plane.clear_objects()
    assert plane.objects == ()
    plane.draw(surface)


def test_palette_slots_stay_with_their_objects() -> None:
    plane, surface = make_plane()
    f, g, h = (pgp.Function(func) for func in (np.sin, np.cos, np.tanh))
    plane.add(f, g, h)
    assert (f.palette, g.palette, h.palette) == (0, 1, 2)
    g.visible = False
    plane.draw(surface)
    expected_plane, expected = make_plane()
    palette = pgp.LIGHT.palette
    expected_plane.plot(expected, np.sin, color=palette[0])
    expected_plane.plot(expected, np.tanh, color=palette[2])  # not palette[1]
    assert pixels(surface) == pixels(expected)
    plane.remove(g)
    plane.add(pgp.Function(np.exp))
    assert plane.objects[-1].palette == 3  # type: ignore[attr-defined]


def test_a_program_can_add_its_own_drawable() -> None:
    """Anything with update, render, handles and detach can be added."""

    class Grid:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def update(self, plane: pgp.Plane) -> None:
            self.calls.append("update")

        def render(self, plane: pgp.Plane, surface: Any) -> None:
            self.calls.append("render")
            plane.points(surface, [(0, 0)], color="black")

        def handles(self, plane: pgp.Plane) -> tuple[pgp.DraggablePoint, ...]:
            return ()

        def detach(self, plane: pgp.Plane) -> None:
            self.calls.append("detach")

    plane, surface = make_plane()
    mine = plane.add(Grid())
    plane.draw(surface)
    plane.remove(mine)
    assert mine.calls == ["update", "render", "detach"]


def test_unchanged_objects_are_reused_between_frames() -> None:
    plane, surface = make_plane()
    plane.add(pgp.Function(np.sin), pgp.ImplicitCurve(lambda x, y: x**2 + y**2, 4))
    plane.draw(surface)
    hits = plane.cache_stats.hits
    plane.draw(surface)
    assert plane.cache_stats.hits >= hits + 2


def test_an_object_on_two_planes() -> None:
    f = pgp.Function(np.sin)
    first, first_surface = make_plane()
    second, second_surface = make_plane()
    first.add(f)
    second.add(f)
    first.draw(first_surface)
    second.draw(second_surface)
    assert pixels(first_surface) == pixels(second_surface)


# --- Errors ---------------------------------------------------------------------


def test_drawing_errors_are_recorded_and_warned_about_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plane, surface = make_plane()
    f = plane.add(pgp.Function(np.sin))
    g = plane.add(pgp.Function(np.cos))
    drawn: list[object] = []
    plot = plane.plot

    def refuse_sin(surface_: Any, func: Any, **kwargs: Any) -> None:
        if func is np.sin:
            raise ValueError("cannot draw this")
        drawn.append(func)
        plot(surface_, func, **kwargs)

    monkeypatch.setattr(plane, "plot", refuse_sin)
    with pytest.warns(pgp.PlotWarning, match="cannot draw this") as caught:
        plane.draw(surface)
    assert caught[0].filename == __file__  # points at the program's call
    assert f.error == "cannot draw this"
    assert g.error is None
    assert drawn == [np.cos]  # the rest still drew
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plane.draw(surface)  # the same error: no second warning
    monkeypatch.undo()
    plane.draw(surface)
    assert f.error is None


def test_errors_inside_functions_leave_gaps() -> None:
    def broken(x: Any) -> Any:
        raise ZeroDivisionError

    plane, surface = make_plane()
    f = plane.add(pgp.Function(broken))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plane.draw(surface)
    assert f.error is None


def test_settings_are_checked() -> None:
    with pytest.raises(TypeError, match="LineStyle"):
        pgp.Function(np.sin, style=pgp.PointStyle())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a function"):
        pgp.Function(2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a function"):
        pgp.ImplicitCurve("x^2 + y^2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="domain"):
        pgp.Function(np.sin, domain=(2, 1))
    with pytest.raises(ValueError, match="infinite"):
        pgp.Points([(0, math.inf)])
    with pytest.raises(TypeError, match="numbers or functions"):
        pgp.Points([(0, "1")])


def test_reprs() -> None:
    assert repr(pgp.Function(np.sin)) == "<Function: y = sin>"
    assert repr(pgp.ParametricCurve(np.cos, np.sin)) == "<ParametricCurve: (cos, sin)>"
    assert repr(pgp.Point((1, 2))) == "<Point: (1, 2)>"
    assert repr(pgp.Polygon([(0, 0), (1, 0), (0, 1)])) == "<Polygon: 3 vertices>"
    assert isinstance(pgp.Points([]), PlotObject)
