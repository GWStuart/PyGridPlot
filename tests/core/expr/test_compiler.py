"""Tests of compiling expressions: values, builtins, bindings, derivatives, errors."""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pygridplot import derivative
from pygridplot.core.expr import (
    FUNCTIONS,
    RESERVED,
    ConstantBinding,
    ExpressionError,
    FunctionBinding,
    PointBinding,
    ValueBinding,
    compile_expression,
    expression,
)

XS = np.array([-3.0, -1.0, -0.25, 0.0, 0.5, 1.0, 2.5, 7.0])


def run(text: str, *args: Any, **bindings: Any) -> Any:
    """Compile ``text`` in ``x`` and evaluate it on ``args`` (default: XS)."""
    compiled = compile_expression(text, ("x",), bindings)
    return compiled(*(args or (XS,)))


# --- Values -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "numpy"),
    [
        ("x^2 + 2x - 1", lambda x: x**2 + 2 * x - 1),
        ("-x^2", lambda x: -(x**2)),
        ("2^-x", lambda x: 2.0 ** (-x)),
        ("x/2/3", lambda x: x / 2 / 3),
        ("3(x + 1)(x - 1)", lambda x: 3 * (x + 1) * (x - 1)),
        ("a sin(x)", lambda x: 1.5 * np.sin(x)),
        ("sin(x)^2 + cos(x)^2", lambda x: np.sin(x) ** 2 + np.cos(x) ** 2),
        ("e^x", np.exp),
        ("2 pi x", lambda x: 2 * np.pi * x),
        ("τ x - π", lambda x: 2 * np.pi * x - np.pi),
        ("x²", lambda x: x**2),
        ("x × 2 − 1", lambda x: x * 2 - 1),
    ],
)
def test_values_match_numpy(text: str, numpy: Callable[[Any], Any]) -> None:
    np.testing.assert_allclose(run(text, a=ConstantBinding(1.5)), numpy(XS), rtol=1e-14)


def test_undefined_values_are_nan_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        values = run("sqrt(x) + ln(x) + 1/(x - 1)")
    assert np.isnan(values[XS < 0]).all()
    assert np.isinf(values[XS == 0]).all()  # ln(0) = -inf


def test_constants_are_spread_over_the_arguments() -> None:
    assert run("2").shape == XS.shape
    assert run("2 + pi", np.zeros((3, 4))).shape == (3, 4)
    assert float(run("2", 5.0)) == 2.0
    assert compile_expression("1 + 2")() == 3.0
    two = compile_expression("x - x + y", ("x", "y"))
    assert two(np.zeros(3), 1.0).shape == (3,)


def test_constant_expressions_are_marked_constant() -> None:
    assert compile_expression("2 sin(pi / 6)", ("x",)).constant
    assert not compile_expression("2x", ("x",)).constant
    live = compile_expression("a", ("x",), {"a": ValueBinding(lambda: 1.0)})
    assert not live.constant


def test_wrong_number_of_arguments() -> None:
    compiled = compile_expression("x + y", ("x", "y"))
    with pytest.raises(TypeError, match="takes 2"):
        compiled(1.0)


# --- Builtins -----------------------------------------------------------------


with np.errstate(all="ignore"):  # the expected values include poles
    BUILTIN_CASES = [
        ("sec(x)", 1 / np.cos(XS)),
        ("csc(x)", 1 / np.sin(XS)),
        ("cot(x)", 1 / np.tan(XS)),
        ("arcsin(x / 8)", np.arcsin(XS / 8)),
        ("asin(x / 8)", np.arcsin(XS / 8)),
        ("acos(x / 8)", np.arccos(XS / 8)),
        ("atan(x)", np.arctan(XS)),
        ("atan(x, -1)", np.arctan2(XS, -1)),
        ("arctan(1, x)", np.arctan2(1, XS)),
        ("sinh(x) + cosh(x) + tanh(x)", np.sinh(XS) + np.cosh(XS) + np.tanh(XS)),
        (
            "asinh(x) + acosh(x) + atanh(x / 8)",
            np.arcsinh(XS) + np.arccosh(XS) + np.arctanh(XS / 8),
        ),
        ("exp(x)", np.exp(XS)),
        ("ln(x)", np.log(XS)),
        ("log(x)", np.log10(XS)),
        ("log(x, 2)", np.log2(XS)),
        ("log2(x) + log10(x)", np.log2(XS) + np.log10(XS)),
        ("sqrt(x) + cbrt(x)", np.sqrt(XS) + np.cbrt(XS)),
        ("hypot(x, 2)", np.hypot(XS, 2)),
        ("abs(x) + sign(x)", np.abs(XS) + np.sign(XS)),
        ("floor(x) + ceil(x)", np.floor(XS) + np.ceil(XS)),
        ("mod(x, 2)", np.mod(XS, 2)),
        ("min(x, 0, 1)", np.minimum(np.minimum(XS, 0), 1)),
        ("max(x, 0)", np.maximum(XS, 0)),
        ("clamp(x, -1, 1)", np.clip(XS, -1, 1)),
    ]


@pytest.mark.parametrize(("text", "expected"), BUILTIN_CASES)
def test_builtins(text: str, expected: Any) -> None:
    np.testing.assert_allclose(run(text), expected, rtol=1e-14, equal_nan=True)


def test_round_rounds_halves_up() -> None:
    np.testing.assert_array_equal(
        run("round(x)", np.array([-1.5, -0.5, 0.5, 1.5, 2.5, 2.4])),
        [-1.0, 0.0, 1.0, 2.0, 3.0, 2.0],
    )


def test_gamma_and_factorial() -> None:
    values = np.array([-2.0, -0.5, 0.0, 0.5, 5.0, 200.0, np.nan])
    gamma = run("gamma(x)", values)
    np.testing.assert_allclose(
        gamma[[1, 3, 4]], [math.gamma(-0.5), math.sqrt(math.pi), 24.0]
    )
    assert np.isnan(gamma[[0, 2, 6]]).all()  # poles, and NaN in
    assert gamma[5] == math.inf
    assert compile_expression("factorial(5)")() == 120.0


def test_every_builtin_function_is_reserved() -> None:
    assert set(FUNCTIONS) <= RESERVED
    assert {"pi", "π", "tau", "τ", "e"} <= RESERVED


def test_builtins_take_the_documented_number_of_arguments() -> None:
    for name, builtin in FUNCTIONS.items():
        args = ", ".join(["x"] * builtin.min_args)
        run(f"{name}({args})")
        too_few = ", ".join(["x"] * (builtin.min_args - 1))
        if builtin.min_args > 1:
            with pytest.raises(ExpressionError, match="takes"):
                run(f"{name}({too_few})")
        if builtin.max_args is not None:
            too_many = ", ".join(["x"] * (builtin.max_args + 1))
            with pytest.raises(ExpressionError, match="takes"):
                run(f"{name}({too_many})")


# --- Logic ------------------------------------------------------------------


def test_comparisons_give_one_zero_or_nan() -> None:
    values = np.array([-1.0, 0.0, 1.0, np.nan])
    np.testing.assert_array_equal(run("x > 0", values), [0.0, 0.0, 1.0, np.nan])
    np.testing.assert_array_equal(run("x <= 0", values), [1.0, 1.0, 0.0, np.nan])
    np.testing.assert_array_equal(run("x == 0", values), [0.0, 1.0, 0.0, np.nan])
    np.testing.assert_array_equal(run("x != 0", values), [1.0, 0.0, 1.0, np.nan])


def test_if_and_logic() -> None:
    values = np.array([-2.0, -0.5, 0.5, 2.0, np.nan])
    np.testing.assert_array_equal(
        run("if(x > 0, x, -x)", values), [2.0, 0.5, 0.5, 2.0, np.nan]
    )
    np.testing.assert_array_equal(
        run("if(x > 0, 1)", values), [np.nan, np.nan, 1.0, 1.0, np.nan]
    )
    np.testing.assert_array_equal(
        run("and(x > -1, x < 1)", values), [0.0, 1.0, 1.0, 0.0, np.nan]
    )
    np.testing.assert_array_equal(
        run("or(x < -1, x > 1)", values), [1.0, 0.0, 0.0, 1.0, np.nan]
    )
    np.testing.assert_array_equal(
        run("not(x > 0)", values), [1.0, 1.0, 0.0, 0.0, np.nan]
    )


def test_an_undefined_condition_gives_an_undefined_value() -> None:
    np.testing.assert_array_equal(
        run("if(sqrt(x) > 1, 1, 0)", np.array([-4.0, 0.25, 4.0])), [np.nan, 0.0, 1.0]
    )


# --- Bindings -----------------------------------------------------------------


def test_value_bindings_are_read_at_every_call() -> None:
    value = [2.0]
    compiled = compile_expression("a x", ("x",), {"a": ValueBinding(lambda: value[0])})
    assert compiled(3.0) == 6.0
    value[0] = -1.0
    assert compiled(3.0) == -3.0


def test_a_number_before_parentheses_multiplies() -> None:
    np.testing.assert_allclose(run("a(x + 1)", a=ConstantBinding(2.0)), 2 * (XS + 1))
    np.testing.assert_allclose(
        run("a(x + 1)", a=ValueBinding(lambda: 3.0)), 3 * (XS + 1)
    )


def test_function_bindings() -> None:
    f = FunctionBinding(np.sinc, 1)
    g = FunctionBinding(lambda u, v: u * v, 2)
    np.testing.assert_allclose(run("f(x) + g(x, 2)", f=f, g=g), np.sinc(XS) + 2 * XS)
    with pytest.raises(ExpressionError, match="f takes 1 argument, not 2"):
        run("f(x, x)", f=f)
    with pytest.raises(ExpressionError, match="call it"):
        run("f + 1", f=f)


def test_function_bindings_find_how_many_arguments_python_functions_take() -> None:
    assert FunctionBinding.of(np.cos).arity == 1
    assert FunctionBinding.of(np.hypot).arity == 2
    assert FunctionBinding.of(lambda u, v, w: u).arity == 3
    assert FunctionBinding.of(lambda u, v=1: u).arity is None
    assert FunctionBinding.of(lambda *a: 0).arity is None
    with pytest.raises(ExpressionError, match="h takes 1 argument"):
        expression("h(x, 1)", h=np.cos)


def test_point_bindings() -> None:
    point = PointBinding(lambda: (1.0, 3.0))
    np.testing.assert_allclose(run("P.x x + P.y", P=point), XS + 3)
    with pytest.raises(ExpressionError, match=re.escape("P.x and P.y")):
        run("P + 1", P=point)
    with pytest.raises(ExpressionError, match="not a function"):
        run("P(x)", P=point)
    with pytest.raises(ExpressionError, match=r"\.x and \.y, not \.z"):
        run("P.z", P=point)


def test_a_point_cannot_be_used_in_arithmetic() -> None:
    with pytest.raises(ExpressionError, match="point"):
        run("(1, 2) + x")


def test_variables_come_before_bindings() -> None:
    assert run("a", 5.0, a=ConstantBinding(1.0)) == 1.0
    compiled = compile_expression("a", ("a",), {"a": ConstantBinding(1.0)})
    assert compiled(5.0) == 5.0


def test_builtin_names_cannot_be_variables() -> None:
    with pytest.raises(ExpressionError, match="built-in"):
        compile_expression("pi + 1", ("pi",))


# --- Derivatives -------------------------------------------------------------


def test_primes_differentiate_exactly() -> None:
    f = FunctionBinding(compile_expression("x^3 - 2x", ("x",)), 1)
    np.testing.assert_allclose(run("f'(x)", f=f), 3 * XS**2 - 2, rtol=1e-14)
    np.testing.assert_allclose(run("f''(x)", f=f), 6 * XS, rtol=1e-14)
    np.testing.assert_allclose(run("f'''(x)", f=f), np.full_like(XS, 6.0))
    np.testing.assert_allclose(run("sin'(x)"), np.cos(XS), rtol=1e-14)
    np.testing.assert_allclose(run("exp''(2x)"), np.exp(2 * XS), rtol=1e-14)


def test_primes_on_a_python_function() -> None:
    f = FunctionBinding(np.sin, 1)
    np.testing.assert_allclose(run("f''(x)", f=f), -np.sin(XS), rtol=1e-12)


def test_compiled_expressions_differentiate_exactly() -> None:
    f = expression("x^2 sin(x) + if(x > 0, x, 0)")
    exact = 2 * XS * np.sin(XS) + XS**2 * np.cos(XS) + (XS > 0)
    np.testing.assert_allclose(derivative(f)(XS[XS != 0]), exact[XS != 0], rtol=1e-12)


def test_derivatives_of_undefined_values_are_undefined() -> None:
    values = np.array([-1.0, 4.0])
    np.testing.assert_array_equal(
        derivative(expression("sqrt(x)"))(values), [np.nan, 0.25]
    )
    np.testing.assert_array_equal(
        derivative(expression("if(x > 0, x^2)"))(values), [np.nan, 8.0]
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("a'(x)", "no derivative"),
        ("g'(x, x)", "one variable"),
        ("f" + "'" * 11 + "(x)", "at most 10 primes"),
    ],
)
def test_prime_errors(text: str, message: str) -> None:
    with pytest.raises(ExpressionError, match=message):
        run(
            text,
            a=ConstantBinding(1.0),
            f=FunctionBinding(np.sin, 1),
            g=FunctionBinding(np.hypot, 2),
        )


# --- Errors -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "suggestion"),
    [
        ("sinx", "sin(x)"),
        ("2cosx", "cos(x)"),
        ("pix", "pi*x"),
        ("xa", "x*a"),
        ("sqr(x)", "sqrt"),
        ("lg(x)", "log"),
    ],
)
def test_unknown_names_get_suggestions(text: str, suggestion: str) -> None:
    with pytest.raises(ExpressionError, match=re.escape(suggestion)) as caught:
        run(text, a=ConstantBinding(1.0))
    assert "unknown name" in caught.value.message


def test_unknown_name_without_a_suggestion() -> None:
    with pytest.raises(ExpressionError) as caught:
        run("1 + qqq")
    assert caught.value.message == "unknown name 'qqq'"
    assert caught.value.position == 4


# --- A property test against NumPy ---------------------------------------------

UNARY: dict[str, Callable[[Any], Any]] = {
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "ln": np.log,
    "tanh": np.tanh,
    "floor": np.floor,
}
BINARY: dict[str, Callable[[Any, Any], Any]] = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.true_divide,
    "^": np.power,
}

Tree = tuple[str, Callable[[Any], Any]]
"""An expression's text and the same computation in NumPy."""


def _leaf() -> st.SearchStrategy[Tree]:
    numbers = st.sampled_from([0.5, 1.0, 2.0, 3.0, 10.0]).map(
        lambda v: (f"{v:g}", lambda x, v=v: np.full_like(x, v))
    )
    return st.one_of(st.just(("x", lambda x: x)), numbers)


def _extend(children: st.SearchStrategy[Tree]) -> st.SearchStrategy[Tree]:
    def unary(name: str, child: Tree) -> Tree:
        text, f = child
        return f"{name}({text})", lambda x: UNARY[name](f(x))

    def negate(child: Tree) -> Tree:
        text, f = child
        return f"-({text})", lambda x: np.negative(f(x))

    def binary(op: str, left: Tree, right: Tree) -> Tree:
        (lt, lf), (rt, rf) = left, right
        return f"({lt}) {op} ({rt})", lambda x: BINARY[op](lf(x), rf(x))

    def implicit(left: Tree, right: Tree) -> Tree:
        (lt, lf), (rt, rf) = left, right
        return f"({lt})({rt})", lambda x: np.multiply(lf(x), rf(x))

    return st.one_of(
        st.builds(unary, st.sampled_from(sorted(UNARY)), children),
        st.builds(negate, children),
        st.builds(binary, st.sampled_from(sorted(BINARY)), children, children),
        st.builds(implicit, children, children),
    )


TREES = st.recursive(_leaf(), _extend, max_leaves=12)


@settings(max_examples=200, deadline=None)
@given(TREES)
def test_compiled_expressions_agree_with_numpy(tree: Tree) -> None:
    text, numpy = tree
    compiled = expression(text, "x")
    with np.errstate(all="ignore"):
        expected = numpy(XS)
    np.testing.assert_allclose(
        compiled(XS), expected, rtol=1e-12, atol=0, equal_nan=True
    )


@settings(max_examples=100, deadline=None)
@given(TREES)
def test_compiled_derivatives_agree_with_numpy(tree: Tree) -> None:
    """Compiled expressions differentiate exactly like the NumPy code they mirror.

    Constant parts are folded when compiling, so their derivative is exactly
    zero even where the chain rule on the NumPy code meets a singularity
    (``sqrt`` of a constant 0 gives ``0 · ∞ = NaN``): where the NumPy code's
    derivative is undefined, the compiled one may be defined.
    """
    text, numpy = tree
    with np.errstate(all="ignore"):
        expected = derivative(numpy)(XS)
    actual = derivative(expression(text, "x"))(XS)
    defined = ~np.isnan(expected)
    assert np.isnan(expected[np.isnan(actual)]).all()
    np.testing.assert_allclose(
        actual[defined], expected[defined], rtol=1e-10, atol=1e-300
    )
