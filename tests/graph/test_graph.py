"""Tests of :class:`pygridplot.Graph`: items, names, errors, evaluation and drawing."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import pygridplot as pgp
from pygridplot.graph import (
    DraggableItem,
    GraphError,
    GraphWarning,
    ParameterItem,
    PlotItem,
    PointItem,
)

from .conftest import frame, make_plane, pixels

# --- The list of items -------------------------------------------------------


def test_items_keep_their_order_and_names() -> None:
    graph = pgp.Graph()
    a = graph.parameter("a", 2)
    f = graph.plot("a x", name="f")
    line = graph.plot("x")
    assert graph.items == (a, f, line)
    assert list(graph) == [a, f, line]
    assert len(graph) == 3
    assert graph.names == ("a", "f")
    assert graph["f"] is f
    assert graph[2] is line
    assert "a" in graph
    assert line in graph
    assert "b" not in graph
    assert graph.get("b") is None
    with pytest.raises(KeyError, match="no item is named 'b'"):
        graph["b"]
    assert repr(graph) == "Graph(3 items: a, f)"


def test_names_are_unique_and_valid() -> None:
    graph = pgp.Graph()
    graph.parameter("a", 1)
    with pytest.raises(GraphError, match="already taken"):
        graph.plot("x", name="a")
    for bad in ("2a", "sin", "pi", "x", "θ", "a b", ""):
        with pytest.raises(GraphError):
            graph.plot("x", name=bad)
    item = graph.plot("x", name="b")
    with pytest.raises(GraphError, match="already taken"):
        item.name = "a"
    item.name = "c"
    assert graph.names == ("a", "c")


def test_remove_clear_and_move() -> None:
    graph = pgp.Graph()
    a = graph.parameter("a", 1)
    f = graph.plot("x", name="f")
    g = graph.plot("x^2", name="g")
    graph.move("g", 0)
    assert graph.items == (g, a, f)
    assert graph.remove("a") is a
    assert a.graph is None
    assert graph.items == (g, f)
    with pytest.raises(KeyError):
        graph.remove(a)
    graph.clear()
    assert len(graph) == 0


def test_an_item_belongs_to_one_graph() -> None:
    first, second = pgp.Graph(), pgp.Graph()
    item = first.plot("x")
    with pytest.raises(GraphError, match="already belongs"):
        second.append(item)
    first.remove(item)
    second.append(item, index=0)
    assert item.graph is second


def test_palette_slots_are_given_in_order_and_kept() -> None:
    graph = pgp.Graph()
    first = graph.plot("x")
    graph.parameter("a", 1)  # not drawn in a colour: no slot
    second = graph.plot("x^2")
    third = graph.add("(1, 2)")
    assert (first.palette, second.palette, third.palette) == (0, 1, 2)  # type: ignore[union-attr]
    graph.remove(second)
    assert graph.plot("x^3").palette == 3  # slots are not reused...
    graph.clear()
    assert graph.plot("x").palette == 0  # ...until the graph is cleared


# --- Names, order and errors -----------------------------------------------------


def test_items_may_use_names_defined_later() -> None:
    graph = pgp.Graph()
    graph.add("y = f(x) + b")
    graph.add("b = 2a")
    graph.add("f(x) = a x^2")
    graph.add("a = 3")
    assert graph.errors == {}
    assert graph.evaluate("f(2) + b") == 18.0


def test_circular_definitions_are_errors() -> None:
    graph = pgp.Graph()
    f = graph.add("f(x) = g(x) + 1")
    g = graph.add("g(x) = f(x)")
    user = graph.add("y = f(x) + 1")
    fine = graph.add("y = x")
    errors = graph.errors
    assert errors[f] == "circular definition: f → g → f"
    assert errors[g] == "circular definition: g → f → g"
    assert errors[user] == "depends on 'f', which has an error"
    assert fine not in errors
    a = graph.add("a = a + 1")
    assert graph.errors[a] == "circular definition: a → a"


def test_an_error_spreads_to_the_items_that_depend_on_it() -> None:
    graph = pgp.Graph()
    a = graph.add("a = b + 1")
    f = graph.add("f(x) = a x")
    errors = graph.errors
    assert errors[a] == "unknown name 'b'"
    assert errors[f] == "depends on 'a', which has an error"
    assert a.error == "unknown name 'b'"
    graph.add("b = 1")
    assert graph.errors == {}
    assert a.error is None


def test_items_that_are_not_values_cannot_be_used_in_expressions() -> None:
    graph = pgp.Graph()
    graph.add("C = (cos(t), sin(t))")
    use = graph.add("y = C + 1")
    assert (
        graph.errors[use]
        == "'C' is a parametric curve and cannot be used in an expression"
    )


def test_errors_warn_once_each() -> None:
    graph = pgp.Graph()
    item = graph.add("y = q x")
    plane, surface = make_plane()
    with pytest.warns(GraphWarning, match="item 0 \\(plot\\): unknown name 'q'"):
        graph.draw(plane, surface)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph.draw(plane, surface)  # the same error: no warning
        item.name = "f"  # recompiles, same error
        graph.draw(plane, surface)
    item.expression = "r x"
    with pytest.warns(GraphWarning, match="'f': unknown name 'r'"):
        graph.draw(plane, surface)


def test_a_drawing_error_is_reported_and_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error raised by the plane skips the item, which records it."""
    graph = pgp.Graph()
    bad = graph.add("x^2 + y^2 = 1")
    good = graph.plot("x")
    plane, surface = make_plane()

    def refuse(*args: object, **kwargs: object) -> None:
        raise ValueError("cannot draw this")

    monkeypatch.setattr(plane, "implicit", refuse)
    with pytest.warns(GraphWarning, match="item 0 \\(implicit\\): cannot draw this"):
        graph.draw(plane, surface)
    assert bad.error == "cannot draw this"
    assert graph.errors == {bad: "cannot draw this"}
    assert good.error is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph.draw(plane, surface)  # the same error: no second warning
    monkeypatch.undo()
    graph.draw(plane, surface)
    assert bad.error is None
    assert graph.errors == {}


def test_errors_are_listed_without_warnings() -> None:
    graph = pgp.Graph()
    item = graph.add("y = q")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert graph.errors == {item: "unknown name 'q'"}
        assert item.error == "unknown name 'q'"


# --- Parameters, definitions and evaluation ----------------------------------------


def test_parameters_are_live() -> None:
    graph = pgp.Graph()
    a = graph.parameter("a", 2)
    double = graph.plot("a x", name="g")
    function = graph.function("g")
    assert function(3.0) == 6.0
    a.value = 5  # a number: read live, nothing recompiles
    assert function(3.0) == 15.0
    a.value = "1 + 1"  # an expression: the graph recompiles
    assert graph.function("g")(3.0) == 6.0
    assert double.error is None


def test_evaluate() -> None:
    graph = pgp.Graph()
    graph.add("f(x) = x^2")
    graph.add("g(u, v) = u - v")
    graph.add("P = (3, f(3))")
    graph.add("k = 2")
    assert graph.evaluate("f(2) + g(5, 1) + k") == 10.0
    assert graph.evaluate("P.y - P.x") == 6.0
    assert graph.evaluate("f'(3)") == pytest.approx(6.0)
    values = graph.evaluate("f(x) + k", x=np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(values, [2.0, 3.0, 6.0])
    with pytest.raises(pgp.ExpressionError, match="unknown name 'q'"):
        graph.evaluate("q + 1")


def test_function_returns_named_functions() -> None:
    graph = pgp.Graph()
    graph.add("f(x) = sin(x)")
    graph.add("r = 1")  # unnamed
    graph.polar("2 + cos(θ)", name="R")
    graph.implicit("x^2 + y^2", name="F")
    graph.add("g(u, v) = u v")
    graph.add("a = 1")
    xs = np.linspace(0, 3, 7)
    np.testing.assert_allclose(graph.function("f")(xs), np.sin(xs))
    np.testing.assert_allclose(pgp.derivative(graph.function("f"))(xs), np.cos(xs))
    np.testing.assert_allclose(graph.function("R")(xs), 2 + np.cos(xs))
    assert graph.function("F")(3.0, 4.0) == 25.0
    assert graph.function("g")(2.0, 3.0) == 6.0
    with pytest.raises(GraphError, match="not a function"):
        graph.function("a")
    graph.add("h(x) = q")
    with pytest.raises(GraphError, match="has an error"):
        graph.function("h")


def test_python_functions_can_be_registered_or_given() -> None:
    graph = pgp.Graph(functions={"sinc": np.sinc})
    graph.register("bump", lambda x: np.exp(-(x**2)))
    item = graph.add("y = sinc(x) + bump(x)")
    graph.plot(np.cos, name="c")
    xs = np.linspace(-2, 2, 9)
    assert graph.errors == {}
    np.testing.assert_allclose(graph.evaluate("c(x)", x=xs), np.cos(xs))
    np.testing.assert_allclose(
        graph.evaluate("sinc(x) + bump(x)", x=xs), np.sinc(xs) + np.exp(-(xs**2))
    )
    with pytest.raises(GraphError, match="already the name of an item"):
        graph.register("c", np.sin)
    with pytest.raises(GraphError, match="registered function"):
        graph.plot("x", name="sinc")
    with pytest.raises(TypeError, match="must be registered with a function"):
        graph.register("two", 2)  # type: ignore[arg-type]
    graph.unregister("bump")
    assert graph.errors[item] == "unknown name 'bump'"


def test_definitions() -> None:
    graph = pgp.Graph()
    g = graph.define("g(u, v)", "u^2 + v")
    assert (g.name, g.parameters, g.expression) == ("g", ("u", "v"), "u^2 + v")
    assert graph.evaluate("g(2, 1)") == 5.0
    g.expression = "u v"
    assert graph.evaluate("g(2, 1)") == 2.0
    with pytest.raises(GraphError):
        graph.define("h(x, x)", "x")
    with pytest.raises(GraphError):
        graph.define("h(sin)", "1")
    with pytest.raises(GraphError):
        graph.define("2(x)", "x")


# --- Drawing -------------------------------------------------------------------


def palette(index: int) -> Any:
    """Return the light theme's palette colour of a slot."""
    colors = pgp.LIGHT.palette
    return colors[index % len(colors)]


SCENES: list[
    tuple[str, Callable[[pgp.Graph], Any], Callable[[pgp.Plane, Any], None]]
] = [
    (
        "plot",
        lambda g: g.add("y = 2sin(x) + 1", width=4),
        lambda p, s: p.plot(s, lambda x: 2 * np.sin(x) + 1, width=4, color=palette(0)),
    ),
    (
        "plot with a domain",
        lambda g: g.plot("x^2 / 4", domain=(-2, 3), opacity=0.5),
        lambda p, s: p.plot(
            s, lambda x: x**2 / 4, domain=(-2, 3), opacity=0.5, color=palette(0)
        ),
    ),
    (
        "parametric",
        lambda g: g.add("(2cos(3t), 2sin(2t))"),
        lambda p, s: p.parametric(
            s,
            lambda t: 2 * np.cos(3 * t),
            lambda t: 2 * np.sin(2 * t),
            (0, 2 * math.pi),
            color=palette(0),
        ),
    ),
    (
        "polar",
        lambda g: g.add("r = 1 + cos(θ)"),
        lambda p, s: p.polar(s, lambda th: 1 + np.cos(th), color=palette(0)),
    ),
    (
        "implicit",
        lambda g: g.add("x^2 + y^2 = 9", color="black"),
        lambda p, s: p.implicit(s, lambda x, y: x**2 + y**2, 9.0, color="black"),
    ),
    (
        "contours",
        lambda g: g.implicit("x y", [1, 2, 4]),
        lambda p, s: p.implicit(s, lambda x, y: x * y, (1, 2, 4), color=palette(0)),
    ),
    (
        "point",
        lambda g: g.add("(1, 2)", radius=7, marker="square"),
        lambda p, s: p.points(s, [(1, 2)], radius=7, marker="square", color=palette(0)),
    ),
    (
        "points, polyline and polygon",
        lambda g: (
            g.points([(0, 0), (1, 1), (2, 4)]),
            g.polyline([(-3, 0), (-2, 1), (-1, 0)], closed=True),
            g.polygon([(1, -1), (3, -1), (2, -3)], fill_opacity=0.5),
        ),
        lambda p, s: (
            p.points(s, [(0, 0), (1, 1), (2, 4)], color=palette(0)),
            p.polyline(s, [(-3, 0), (-2, 1), (-1, 0)], closed=True, color=palette(1)),
            p.polygon(
                s, [(1, -1), (3, -1), (2, -3)], fill_opacity=0.5, color=palette(2)
            ),
        ),
    ),
    (
        "names in expressions",
        lambda g: (
            g.add("a = 2"),
            g.add("f(x) = a x"),
            g.add("y = f(x) - f'(x)"),
            g.add("P = (a, f(a))"),
            g.polyline([(0, 0), ("P.x", "P.y")]),
        ),
        lambda p, s: (
            p.plot(s, lambda x: 2 * x, color=palette(0)),
            p.plot(s, lambda x: 2 * x - 2, color=palette(1)),
            p.points(s, [(2, 4)], color=palette(2)),
            p.polyline(s, [(0, 0), (2, 4)], color=palette(3)),
        ),
    ),
]


@pytest.mark.parametrize(
    ("graph_scene", "plane_scene"), [s[1:] for s in SCENES], ids=[s[0] for s in SCENES]
)
def test_drawing_a_graph_is_drawing_with_the_plane(
    graph_scene: Callable[[pgp.Graph], Any],
    plane_scene: Callable[[pgp.Plane, Any], None],
) -> None:
    """A graph draws exactly what the same plane calls would."""
    graph = pgp.Graph()
    graph_scene(graph)
    plane, surface = make_plane()
    graph.draw(plane, surface)
    expected_plane, expected = make_plane()
    plane_scene(expected_plane, expected)
    assert graph.errors == {}
    assert pixels(surface) == pixels(expected)


def test_hidden_items_are_not_drawn_but_keep_their_names() -> None:
    graph = pgp.Graph()
    f = graph.add("f(x) = x^2", visible=False)
    graph.add("y = f(x) + 1")
    plane, surface = make_plane()
    graph.draw(plane, surface)
    expected_plane, expected = make_plane()
    expected_plane.plot(expected, lambda x: x**2 + 1, color=palette(1))
    assert pixels(surface) == pixels(expected)
    f.visible = True
    frame(plane, surface, graph)
    assert pixels(surface) != pixels(expected)


def test_the_colours_follow_the_theme() -> None:
    graph = pgp.Graph()
    graph.add("y = x")
    plane, surface = make_plane(theme=pgp.DARK)
    graph.draw(plane, surface)
    expected_plane, expected = make_plane(theme=pgp.DARK)
    expected_plane.plot(expected, lambda x: x, color=pgp.DARK.palette[0])
    assert pixels(surface) == pixels(expected)


def test_changing_an_item_changes_the_drawing() -> None:
    graph = pgp.Graph()
    a = graph.add("a = 1")
    plot = graph.add("y = a x")
    plane, surface = make_plane()
    graph.draw(plane, surface)
    before = pixels(surface)
    assert isinstance(a, ParameterItem)
    a.value = 2
    frame(plane, surface, graph)
    assert pixels(surface) != before
    a.value = 1
    frame(plane, surface, graph)
    assert pixels(surface) == before
    assert isinstance(plot, PlotItem)
    plot.color = "red"
    frame(plane, surface, graph)
    assert pixels(surface) != before


def test_unchanged_plots_are_reused_between_frames() -> None:
    graph = pgp.Graph()
    graph.add("y = sin(x)")
    graph.add("x^2 + y^2 = 4")
    plane, surface = make_plane()
    frame(plane, surface, graph)
    hits = plane.cache_stats.hits
    frame(plane, surface, graph)
    assert plane.cache_stats.hits >= hits + 2


def test_a_graph_draws_on_several_planes() -> None:
    graph = pgp.Graph()
    graph.add("y = x^2")
    first, first_surface = make_plane()
    second, second_surface = make_plane()
    graph.draw(first, first_surface)
    graph.draw(second, second_surface)
    assert pixels(first_surface) == pixels(second_surface)


def test_items_can_be_made_directly() -> None:
    graph = pgp.Graph()
    point = graph.append(PointItem(("a", 1)))
    graph.append(ParameterItem("a", 3), index=0)
    assert graph.items[1] is point
    assert graph.evaluate("a") == 3.0
    assert isinstance(graph.items[0], ParameterItem)
    assert not isinstance(point, DraggableItem)


def test_item_reprs() -> None:
    graph = pgp.Graph()
    assert repr(graph.add("f(x) = x^2")) == "<PlotItem 'f': y = x^2>"
    assert repr(graph.add("x^2 + y^2 = 1")) == "<ImplicitItem: x^2 + y^2 = 1>"
    assert repr(graph.add("a = 2")) == "<ParameterItem 'a': a = 2>"
    assert repr(graph.plot(np.sin)) == "<PlotItem: y = sin>"
