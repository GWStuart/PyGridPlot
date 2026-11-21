"""Tests of reading one line of maths into an item (:meth:`Graph.add`)."""

from __future__ import annotations

import math
from typing import Any

import pytest

import pygridplot as pgp
from pygridplot import Function, ImplicitCurve, ParametricCurve, Point, PolarCurve
from pygridplot.graph import Definition, Parameter
from pygridplot.graph.lines import interpret


@pytest.mark.parametrize(
    ("line", "method", "arguments"),
    [
        ("y = x^2", "plot", {"expression": "x^2"}),
        ("x^2 = y", "plot", {"expression": "x^2"}),
        ("x^2", "plot", {"expression": "x^2"}),
        ("y = (x + 1)(x - 1)", "plot", {"expression": "(x + 1)(x - 1)"}),
        ("f(x) = x^2 / 4", "plot", {"expression": "x^2 / 4", "name": "f"}),
        ("r = 1 + cos(θ)", "polar", {"r": "1 + cos(θ)"}),
        ("r = 2", "polar", {"r": "2"}),
        ("(cos(t), sin(t))", "parametric", {"x": "cos(t)", "y": "sin(t)"}),
        (
            "C = (2cos(t), sin(t))",
            "parametric",
            {"name": "C", "x": "2cos(t)", "y": "sin(t)"},
        ),
        ("(1, -2.5)", "point", {"position": (1.0, -2.5)}),
        ("P = (a, 2a)", "point", {"name": "P", "position": ("a", "2a")}),
        ("a = 2", "parameter", {"name": "a", "value": 2.0}),
        ("a = -1.5", "parameter", {"name": "a", "value": -1.5}),
        ("b = 2a + 1", "parameter", {"name": "b", "value": "2a + 1"}),
        ("g(u, v) = u v", "define", {"signature": "g(u, v)", "expression": "u v"}),
        ("h(t) = t^2", "define", {"signature": "h(t)", "expression": "t^2"}),
        ("x^2 + y^2 = 9", "implicit", {"expression": "x^2 + y^2", "level": 9.0}),
        ("x = 3", "implicit", {"expression": "x", "level": 3.0}),
        ("x = y^2", "implicit", {"expression": "x - (y^2)"}),
        ("y^2 = x^3 - x", "implicit", {"expression": "y^2 - (x^3 - x)"}),
        ("a = x", "implicit", {"expression": "a - (x)"}),
    ],
)
def test_lines(line: str, method: str, arguments: dict[str, Any]) -> None:
    assert interpret(line) == (method, arguments)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("y < x^2", "inequalities"),
        ("x^2 + y", "needs an equation"),
        ("(1, 2, 3)", "two coordinates"),
        ("(1, 2) = x", "one side of an equation"),
        ("f'(x) = 2", "name with parameters"),
        ("f(2) = 3", "name with parameters"),
        ("y = ", "expected"),
    ],
)
def test_bad_lines(line: str, message: str) -> None:
    with pytest.raises(pgp.ExpressionError, match=message):
        interpret(line)


def test_add_makes_the_item_of_each_kind() -> None:
    graph = pgp.Graph()
    expected = [
        ("y = x^2", Function),
        ("f(x) = x", Function),
        ("r = 1", PolarCurve),
        ("(cos(t), sin(t))", ParametricCurve),
        ("(1, 2)", Point),
        ("a = 2", Parameter),
        ("g(u, v) = u + v", Definition),
        ("x^2 + y^2 = 1", ImplicitCurve),
    ]
    for line, kind in expected:
        assert type(graph.add(line)) is kind
    assert graph.names == ("f", "a", "g")
    assert graph["g"].parameters == ("u", "v")  # type: ignore[attr-defined]


def test_add_passes_options_on() -> None:
    graph = pgp.Graph()
    item = graph.add("y = sin(x)", color="red", width=4, domain=(0, 5))
    assert isinstance(item, Function)
    assert (item.color, item.width, item.domain) == ("red", 4, (0.0, 5.0))
    curve = graph.add("(cos(t), sin(t))", t_range=(0, math.pi))
    assert isinstance(curve, ParametricCurve)
    assert curve.t_range == (0.0, math.pi)
    named = graph.add("y = x", name="line")
    assert named.name == "line"


def test_an_option_cannot_contradict_the_line() -> None:
    graph = pgp.Graph()
    with pytest.raises(pgp.GraphError, match="name is given by the line"):
        graph.add("f(x) = x", name="g")
    graph.add("f(x) = x", name="f")  # the same is fine


def test_a_point_can_be_made_draggable() -> None:
    graph = pgp.Graph()
    point = graph.add("P = (1, 2)", draggable=True, lock="x")
    assert isinstance(point, Point)
    assert (point.name, point.position, point.lock) == ("P", (1.0, 2.0), "x")
    with pytest.raises(pgp.GraphError, match="only a point"):
        graph.add("y = x", draggable=True)
    with pytest.raises(pgp.GraphError, match="must be numbers"):
        graph.add("(a, 1)", draggable=True)
