"""Tests of :func:`pygridplot.expression`: text to a function in one call."""

from __future__ import annotations

import numpy as np
import pygame
import pytest

import pygridplot as pgp


def test_the_variables_are_found_in_the_text() -> None:
    assert pgp.expression("x^2").variables == ("x",)
    assert pgp.expression("x y + 1").variables == ("x", "y")
    assert pgp.expression("y - x").variables == ("x", "y")  # always in this order
    assert pgp.expression("cos(t)").variables == ("t",)
    assert pgp.expression("1 + cos(θ)").variables == ("θ",)
    assert pgp.expression("2").variables == ("x",)  # a constant function of x


def test_variables_can_be_given() -> None:
    f = pgp.expression("u v", "u", "v")
    assert f(2.0, 3.0) == 6.0
    g = pgp.expression("x + y", "y", "x")
    assert g(1.0, 10.0) == 11.0


def test_names_can_be_numbers_functions_or_points() -> None:
    f = pgp.expression("a h(x) + P.y", a=2, h=np.sinc, P=(0.0, 1.0))
    xs = np.linspace(-2, 2, 9)
    np.testing.assert_allclose(f(xs), 2 * np.sinc(xs) + 1)
    assert f.variables == ("x",)  # names given are not variables


def test_bad_names() -> None:
    with pytest.raises(TypeError, match="number, a function or an"):
        pgp.expression("a x", a="two")
    with pytest.raises(pgp.ExpressionError, match="unknown name 'b'"):
        pgp.expression("a x + b", a=1)


def test_an_expression_can_be_plotted_and_differentiated() -> None:
    pygame.init()
    try:
        surface = pygame.Surface((200, 150))
        plane = pgp.Plane(pygame.Rect(0, 0, 200, 150))
        plane.draw_grid(surface)
        f = pgp.expression("x^3 / 3")
        plane.plot(surface, f)
        plane.plot(surface, pgp.derivative(f))
        np.testing.assert_allclose(pgp.derivative(f)(np.array([2.0])), [4.0])
    finally:
        pygame.quit()
