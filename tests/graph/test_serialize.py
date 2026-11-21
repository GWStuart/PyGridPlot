"""Tests of saving graphs as JSON and reading them back."""

from __future__ import annotations

import json
import math
from dataclasses import astuple
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pygridplot as pgp
from pygridplot.graph import GraphError
from pygridplot.graph.serialize import FORMAT, VERSION

from .conftest import frame, make_plane, pixels


def every_kind() -> pgp.Graph:
    """Return a graph with an item of every kind, most options set."""
    graph = pgp.Graph(view=pgp.GraphView((-6, 6), (-4, 4), equal_aspect=False))
    graph.add("a = 1.5")
    graph.add("b = 2a")
    graph.add("f(x) = a sin(x)", color=(200, 40, 40), width=3.5)
    graph.plot("x^2 / 8", domain=(-3, 2), opacity=0.6, visible=False)
    graph.plot("b x", sampling=pgp.SamplingConfig(samples_per_px=2.0))
    graph.add("C = (2cos(t), sin(t))", t_range=(0, math.pi))
    graph.polar("1 + cos(θ)", theta_range=(0, 4 * math.pi), name="R")
    graph.implicit(
        "x^2 + y^2", [1, 4, 9], sampling=pgp.ImplicitSampling(refine_steps=2)
    )
    graph.add("Q = (a, f(a))", marker="square", radius=6)
    graph.points([(0, 0), (1, float("nan")), ("b", 1)], edge_color="white")
    graph.polyline([(-3, 0), (-2, 1), (-1, 0)], closed=True)
    graph.polygon([(1, -1), (3, -1), (2, "-b")], fill_opacity=0.5, fill_rule="evenodd")
    graph.draggable(
        (1, 0.5), name="P", x_range=(0, math.inf), lock="y", coordinates="always"
    )
    graph.draggable(curve="f", t=1.0, name="D", enabled=False)
    graph.draggable(curve=("cos(t)", "sin(t)", (0, math.pi)), t=0.5)
    graph.define("g(u, v)", "u v + 1")
    return graph


def test_a_saved_graph_reads_back_the_same() -> None:
    graph = every_kind()
    text = graph.to_json()
    loaded = pgp.Graph.from_json(text)
    assert loaded.to_json() == text
    assert loaded.to_dict() == graph.to_dict()
    assert loaded.names == graph.names
    assert loaded.view == graph.view
    assert loaded.errors == {}


def test_a_saved_graph_draws_the_same() -> None:
    graph = every_kind()
    loaded = pgp.Graph.from_json(graph.to_json())
    plane, surface = make_plane()
    other_plane, other = make_plane()
    graph.draw(plane, surface)
    loaded.draw(other_plane, other)
    assert pixels(surface) == pixels(other)


def test_the_output_is_tidy() -> None:
    graph = pgp.Graph()
    graph.add("a = 2")
    graph.add("y = a x", width=3)
    data = json.loads(graph.to_json())
    assert data == {
        "format": FORMAT,
        "version": VERSION,
        "items": [
            {"kind": "parameter", "name": "a", "value": 2},
            {"kind": "plot", "expression": "a x", "palette": 0, "style": {"width": 3}},
        ],
    }
    assert '"t_range": [0, 3.14' not in graph.to_json()
    one_line = graph.to_json(indent=None)
    assert "\n" not in one_line
    assert json.loads(one_line) == data


def test_short_lists_stay_on_one_line() -> None:
    graph = pgp.Graph()
    graph.add("(cos(t), sin(t))", t_range=(0, 3), color=(10, 20, 30))
    text = graph.to_json()
    assert '"t_range": [0, 3]' in text
    assert '"color": [10, 20, 30]' in text


def test_dragged_points_are_saved_where_they_are() -> None:
    graph = pgp.Graph()
    point = graph.draggable((1, 1), name="P")
    point.position = (2.5, -1)
    assert pgp.Graph.from_json(graph.to_json())["P"].position == (2.5, -1.0)  # type: ignore[attr-defined]


def test_gaps_and_open_bounds_are_saved_as_null() -> None:
    graph = pgp.Graph()
    graph.polyline([(0, 0), (float("nan"), 1), (2, 2)])
    graph.draggable((1, 1), y_range=(-math.inf, 3))
    data = graph.to_dict()
    assert data["items"][0]["points"][1] == [None, 1]
    assert data["items"][1]["y_range"] == [None, 3]
    loaded = pgp.Graph.from_dict(json.loads(json.dumps(data)))
    gap = loaded.items[0].points[1]  # type: ignore[attr-defined]
    assert math.isnan(gap[0])
    assert loaded.items[1].y_range == (-math.inf, 3.0)  # type: ignore[attr-defined]


def test_files(tmp_path: Path) -> None:
    graph = every_kind()
    path = tmp_path / "graph.json"
    graph.save(path)
    assert path.read_text(encoding="utf-8").endswith("}\n")
    assert pgp.Graph.load(path).to_json() == graph.to_json()
    assert "θ" in path.read_text(encoding="utf-8")  # written as UTF-8, not escaped


# --- Python functions -----------------------------------------------------------


def test_registered_functions_are_saved_by_name() -> None:
    graph = pgp.Graph(functions={"bump": lambda x: np.exp(-(x**2))})
    graph.add("y = bump(x)")
    graph.plot(np.sinc, name="s")
    with pytest.raises(
        GraphError, match=r"items\[1\]\.expression: uses the Python function sinc"
    ):
        graph.to_json()
    graph.register("sinc", np.sinc)
    data = graph.to_dict()
    assert data["functions"] == ["bump", "sinc"]
    assert data["items"][1]["expression"] == {"function": "sinc"}
    with pytest.raises(
        GraphError, match=r"needs the Python functions \['bump', 'sinc'\]"
    ):
        pgp.Graph.from_dict(data)
    loaded = pgp.Graph.from_dict(data, functions=dict(graph.functions))
    xs = np.linspace(-1, 1, 5)
    np.testing.assert_allclose(loaded.function("s")(xs), np.sinc(xs))
    assert loaded.errors == {}


def wave(t: Any) -> Any:
    """A Python function to register."""
    return np.sin(3 * t)


def test_python_curve_components_are_saved_by_name() -> None:
    graph = pgp.Graph(functions={"wave": wave})
    curve = graph.parametric(lambda t: t, "t", (0, 1))
    with pytest.raises(
        GraphError, match=r"items\[0\]\.x: uses the Python function <lambda>"
    ):
        graph.to_dict()
    curve.x = wave
    data = graph.to_dict()
    assert data["items"][0]["x"] == {"function": "wave"}
    assert pgp.Graph.from_dict(data, functions={"wave": wave}).to_dict() == data


# --- Views ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [{}, {"equal_aspect": False}, {"log_x": True}, {"log_x": True, "log_y": True}],
)
def test_a_saved_view_is_restored_exactly(options: dict[str, Any]) -> None:
    plane, surface = make_plane(**options)
    plane.set_view((1, 10), (0.5, 2))
    graph = pgp.Graph()
    view = graph.capture_view(plane)
    assert graph.view is view
    other, _ = make_plane()
    for _ in range(10):  # saving and loading again and again changes nothing
        loaded = pgp.Graph.from_json(graph.to_json())
        loaded.apply_view(other)
        graph = loaded
        graph.capture_view(other)
    expected = astuple(plane.view.state())
    assert astuple(other.view.state()) == pytest.approx(expected, rel=1e-12)
    assert (other.log_x, other.log_y) == (plane.log_x, plane.log_y)
    pgp.Graph().apply_view(other)  # no view: nothing happens
    assert astuple(other.view.state()) == pytest.approx(expected, rel=1e-12)
    frame(plane, surface, graph)


def test_log_views() -> None:
    view = pgp.GraphView((1, 1000), (0.1, 10), log_x=True, log_y=True)
    plane, _ = make_plane()
    view.apply(plane)
    assert plane.log_x
    assert plane.log_y
    assert pgp.GraphView.of(plane).x_range == pytest.approx((1, 1000))
    with pytest.raises(ValueError, match="positive on a log axis"):
        pgp.GraphView((0, 10), (1, 2), log_x=True)
    with pytest.raises(ValueError, match="increasing"):
        pgp.GraphView((2, 1), (1, 2))


# --- Reading is strict -------------------------------------------------------------


def saved(*items: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Return saved data with the given items."""
    return {"format": FORMAT, "version": VERSION, "items": list(items), **extra}


def test_hand_written_files_need_only_what_differs() -> None:
    graph = pgp.Graph.from_dict(
        saved(
            {"kind": "plot", "expression": "x^2"},
            {"kind": "parametric", "x": "cos(t)", "y": 0},
            {"kind": "draggable", "curve": "f"},
            {"kind": "plot", "name": "f", "expression": "x", "style": {"color": "red"}},
        )
    )
    assert len(graph) == 4
    assert graph.items[1].t_range == (0.0, 2 * math.pi)  # type: ignore[attr-defined]
    assert graph["f"].color == "red"  # type: ignore[attr-defined]
    assert graph.errors == {}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"format": "other", "version": 1, "items": []}, "not a saved graph"),
        (saved(version=99), "saved in version 99"),
        (saved(version="1"), "version: expected a whole number"),
        ({"format": FORMAT, "version": 1}, "items: missing"),
        (saved(colour="red"), "colour: unknown key"),
        (saved({"kind": "spline"}), r"items\[0\]\.kind: unknown kind of item 'spline'"),
        (saved({"expression": "x"}), r"items\[0\]\.kind: missing"),
        (saved({"kind": "plot"}), r"items\[0\]\.expression: missing"),
        (
            saved({"kind": "plot", "expression": 2}),
            r"items\[0\]\.expression: expected an expression",
        ),
        (saved({"kind": "plot", "expression": "x^"}), r"items\[0\]: expected a number"),
        (
            saved({"kind": "plot", "expression": "x", "width": 2}),
            r"items\[0\]\.width: unknown key",
        ),
        (
            saved({"kind": "plot", "expression": "x", "style": {"width": -1}}),
            r"items\[0\]\.style: .*width",
        ),
        (
            saved({"kind": "plot", "expression": "x", "style": {"widht": 1}}),
            r"items\[0\]\.style\.widht: unknown key",
        ),
        (
            saved({"kind": "plot", "expression": "x", "style": {"width": "thick"}}),
            r"items\[0\]\.style\.width: expected a number",
        ),
        (
            saved({"kind": "polygon", "points": [], "style": {"fill": 1}}),
            r"items\[0\]\.style\.fill: expected true or false",
        ),
        (
            saved({"kind": "point", "position": [0, 0], "style": {"marker": 3}}),
            r"items\[0\]\.style\.marker: expected text",
        ),
        (
            saved({"kind": "plot", "expression": "x", "sampling": {"max_rounds": 2.5}}),
            r"items\[0\]\.sampling\.max_rounds: expected a whole number",
        ),
        (
            saved({"kind": "plot", "expression": "x", "domain": [1]}),
            r"items\[0\]\.domain: expected two numbers",
        ),
        (
            saved({"kind": "plot", "expression": "x", "palette": -1}),
            r"items\[0\]\.palette: expected a non-negative",
        ),
        (
            saved({"kind": "plot", "expression": "x", "visible": "yes"}),
            "expected true or false",
        ),
        (
            saved({"kind": "plot", "expression": {"function": "nope"}}),
            r"items\[0\]\.expression: unknown function 'nope'",
        ),
        (
            saved({"kind": "point", "position": [1, 2, 3]}),
            r"items\[0\]\.position: expected",
        ),
        (
            saved({"kind": "points", "points": [[1]]}),
            r"items\[0\]\.points\[0\]: expected",
        ),
        (saved({"kind": "parameter", "value": 1}), r"items\[0\]\.name: missing"),
        (saved({"kind": "parameter", "name": "sin", "value": 1}), "built-in name"),
        (saved({"kind": "draggable", "curve": 3}), r"items\[0\]\.curve: expected"),
        (
            saved({"kind": "draggable", "curve": {"x": "t", "y": "t"}}),
            r"items\[0\]\.curve\.t_range: missing",
        ),
        (
            saved({"kind": "draggable", "position": [0, 0], "lock": "z"}),
            r"items\[0\]: lock must be",
        ),
        (
            saved(
                {"kind": "parameter", "name": "a", "value": 1},
                {"kind": "parameter", "name": "a", "value": 2},
            ),
            r"items\[1\]: the name 'a' is already taken",
        ),
        (
            saved(view={"x_range": [1, 0], "y_range": [0, 1]}),
            "view: x_range must be increasing",
        ),
        (saved(view={"x_range": [0, 1]}), r"view\.y_range: missing"),
        (saved(functions=[1]), r"functions\[0\]: expected"),
        (saved(items={}), "items: expected a list"),
        (saved("plot"), r"items\[0\]: expected an object"),
    ],
)
def test_invalid_data_is_explained(data: dict[str, Any], message: str) -> None:
    with pytest.raises(GraphError, match=message):
        pgp.Graph.from_dict(data)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{", "not valid JSON"),
        ("[]", "a saved graph is a JSON object"),
        ("", "not valid JSON"),
    ],
)
def test_invalid_json(text: str, message: str) -> None:
    with pytest.raises(GraphError, match=message):
        pgp.Graph.from_json(text)


# --- A property test: any graph reads back the same -------------------------------

EXPRESSIONS = st.sampled_from(["x", "x^2 - 1", "a sin(x)", "f(x) + 1", "sqrt(x)", "2"])
NUMBERS = st.floats(-50, 50, allow_nan=False).map(lambda v: round(v, 3))
RANGES = st.tuples(NUMBERS, st.floats(0.5, 20)).map(
    lambda p: (p[0], round(p[0] + p[1], 3))
)
COLOURS = st.one_of(
    st.none(),
    st.sampled_from(["red", "#123456"]),
    st.tuples(st.integers(0, 255), st.integers(0, 255), st.integers(0, 255)),
)
COORDINATES = st.one_of(NUMBERS, st.sampled_from(["a", "2a", "f(1)"]))


@st.composite
def graphs(draw: st.DrawFn) -> pgp.Graph:
    """Draw a graph of random items with random (valid) options."""
    graph = pgp.Graph()
    graph.add("a = 2")
    graph.add("f(x) = x / 2")
    for _ in range(draw(st.integers(0, 6))):
        kind = draw(
            st.sampled_from(
                [
                    "plot",
                    "parametric",
                    "polar",
                    "implicit",
                    "point",
                    "path",
                    "drag",
                    "parameter",
                ]
            )
        )
        visible = draw(st.booleans())
        color = draw(COLOURS)
        if kind == "plot":
            graph.plot(
                draw(EXPRESSIONS),
                domain=draw(st.none() | RANGES),
                color=color,
                width=draw(st.none() | st.floats(0.5, 8)),
                visible=visible,
            )
        elif kind == "parametric":
            graph.parametric(
                draw(EXPRESSIONS).replace("x", "t"), "t", draw(RANGES), color=color
            )
        elif kind == "polar":
            graph.polar(
                draw(st.sampled_from(["1", "θ / 3", "a + cos(theta)"])), visible=visible
            )
        elif kind == "implicit":
            level = draw(st.one_of(NUMBERS, st.lists(NUMBERS, min_size=1, max_size=3)))
            graph.implicit("x^2 + a y^2", level, color=color)
        elif kind == "point":
            graph.point(
                (draw(COORDINATES), draw(COORDINATES)),
                radius=draw(st.none() | st.floats(1, 9)),
            )
        elif kind == "path":
            points = draw(st.lists(st.tuples(COORDINATES, COORDINATES), max_size=5))
            maker = draw(st.sampled_from([graph.points, graph.polyline, graph.polygon]))
            maker(points, color=color, visible=visible)
        elif kind == "drag":
            lock = draw(st.sampled_from([None, "x", "y"]))
            low = draw(st.none() | st.just(-math.inf) | NUMBERS)
            graph.draggable(
                (draw(NUMBERS), draw(NUMBERS)),
                lock=lock,
                x_range=None if low is None else (low, math.inf),
                coordinates=draw(st.sampled_from(["hover", "always", "never"])),
            )
        else:
            graph.parameter(
                f"p{len(graph)}", draw(st.one_of(NUMBERS, st.just("a + 1")))
            )
    return graph


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graphs())
def test_any_graph_reads_back_the_same(graph: pgp.Graph) -> None:
    text = graph.to_json()
    loaded = pgp.Graph.from_json(text)
    assert loaded.to_json() == text
    assert pgp.Graph.from_dict(graph.to_dict()).to_dict() == graph.to_dict()
    assert len(loaded) == len(graph)


JSON = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(-5, 5)
    | st.floats(-1e6, 1e6)
    | st.text(max_size=6),
    lambda children: (
        st.lists(children, max_size=3)
        | st.dictionaries(st.text(max_size=6), children, max_size=3)
    ),
    max_leaves=6,
)


def _paths(data: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Return the path of every value inside nested lists and dicts."""
    found = [prefix]
    if isinstance(data, dict):
        for key, value in data.items():
            found += _paths(value, (*prefix, key))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            found += _paths(value, (*prefix, index))
    return found


@settings(max_examples=300, deadline=None)
@given(st.data())
def test_damaged_files_raise_only_graph_errors(data: st.DataObject) -> None:
    """Replacing any value of a saved graph by any JSON never crashes loading."""
    saved_data = json.loads(every_kind().to_json())
    path = data.draw(st.sampled_from(_paths(saved_data)[1:]))
    target = saved_data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = data.draw(JSON)
    try:
        graph = pgp.Graph.from_dict(saved_data)
    except GraphError:
        return
    graph.to_dict()  # whatever loads can be saved again
