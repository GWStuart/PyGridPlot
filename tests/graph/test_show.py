"""Tests of :func:`pygridplot.show` with lines of maths and graphs."""

from __future__ import annotations

import numpy as np
import pygame
import pytest

import pygridplot as pgp


def quit_soon() -> None:
    """Make the next :func:`pygridplot.show` window close after one frame."""
    pygame.event.post(pygame.event.Event(pygame.QUIT))


def test_show_draws_lines_functions_and_graphs() -> None:
    graph = pgp.Graph(view=pgp.GraphView((-2, 2), (-1, 1)))
    graph.draggable((0.5, 0.5), name="P")
    graph.add("y = P.y sin(x / P.x)")
    quit_soon()
    pgp.show("y = x^2", "x^2 + y^2 = 9", np.cos, graph, size=(200, 150), width=3)
    assert graph.errors == {}
    assert graph["P"].position == (0.5, 0.5)  # type: ignore[attr-defined]


def test_show_reports_a_bad_line_before_opening_a_window() -> None:
    with pytest.raises(pgp.ExpressionError, match="expected a number"):
        pgp.show("y = x +* 2")


def test_the_view_of_a_graph_is_used_unless_another_is_given() -> None:
    graph = pgp.Graph(view=pgp.GraphView((0, 1), (0, 1)))
    quit_soon()
    pgp.show(graph, size=(200, 150))
    quit_soon()
    pgp.show(graph, x_range=(-5, 5), size=(200, 150))
