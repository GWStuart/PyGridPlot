"""Tests of the expression lexer and parser: tokens, grammar, spans and errors."""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pygridplot.core.expr import ExpressionError, parse, parse_equation, references
from pygridplot.core.expr.lexer import tokenize
from pygridplot.core.expr.parser import (
    MAX_DEPTH,
    Attribute,
    Binary,
    Call,
    Compare,
    Name,
    Node,
    Number,
    Tuple,
    Unary,
)

# --- Tokens -----------------------------------------------------------------


def kinds(text: str) -> list[tuple[str, str]]:
    """Return the (kind, text) of every token but the end."""
    return [(t.kind, t.text) for t in tokenize(text)[:-1]]


@pytest.mark.parametrize(
    ("text", "value"),
    [("2", 2.0), ("0.5", 0.5), (".5", 0.5), ("3.", 3.0), ("1e-3", 1e-3), ("2E4", 2e4)],
)
def test_numbers(text: str, value: float) -> None:
    (token, end) = tokenize(text)
    assert token.kind == "number"
    assert token.value == value
    assert end.kind == "end"


def test_an_e_without_digits_is_a_name() -> None:
    assert kinds("2e") == [("number", "2"), ("name", "e")]
    assert kinds("2ex") == [("number", "2"), ("name", "ex")]
    assert kinds("2e-x") == [
        ("number", "2"),
        ("name", "e"),
        ("operator", "-"),
        ("name", "x"),
    ]


def test_names_take_letters_digits_and_underscores() -> None:
    assert kinds("a_2 θ π x1") == [
        ("name", "a_2"),
        ("name", "θ"),
        ("name", "π"),
        ("name", "x1"),
    ]


def test_primes_are_counted_after_a_name() -> None:
    (token, *_) = tokenize("f''(x)")
    assert (token.text, token.primes, token.start, token.end) == ("f", 2, 0, 3)


def test_unicode_operators_are_read_as_ascii() -> None:
    tokens = [t for _, t in kinds("a − b × c ÷ d · e ≤ f ≥ g ≠ h")]
    assert tokens == [*"a-b*c/d*e", "<=", "f", ">=", "g", "!=", "h"]


def test_double_star_is_a_power() -> None:
    assert kinds("x**2") == [("name", "x"), ("operator", "^"), ("number", "2")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("x²", [("name", "x"), ("operator", "^"), ("number", "2")]),
        ("x¹⁰", [("name", "x"), ("operator", "^"), ("number", "10")]),
        ("x⁻¹", [("name", "x"), ("operator", "^"), ("operator", "-"), ("number", "1")]),
    ],
)
def test_superscripts_are_powers(text: str, expected: list[tuple[str, str]]) -> None:
    assert kinds(text) == expected


@pytest.mark.parametrize("text", ["x @ 2", "a $", "x⁻", "#"])
def test_unexpected_characters(text: str) -> None:
    with pytest.raises(ExpressionError):
        tokenize(text)


# --- Grammar ------------------------------------------------------------------


def show(node: Node) -> str:
    """Write a tree fully parenthesised, to compare structures."""
    if isinstance(node, Number):
        return f"{node.value:g}"
    if isinstance(node, Name):
        return node.name
    if isinstance(node, Attribute):
        return f"{node.target}.{node.attribute}"
    if isinstance(node, Call):
        args = ", ".join(show(a) for a in node.args)
        return f"{node.name}{chr(39) * node.primes}({args})"
    if isinstance(node, Unary):
        return f"({node.op}{show(node.operand)})"
    if isinstance(node, (Binary, Compare)):
        return f"({show(node.left)} {node.op} {show(node.right)})"
    assert isinstance(node, Tuple)
    return "<" + ", ".join(show(i) for i in node.items) + ">"


@pytest.mark.parametrize(
    ("text", "tree"),
    [
        ("1 + 2 * 3", "(1 + (2 * 3))"),
        ("a / b * c", "((a / b) * c)"),
        ("a - b - c", "((a - b) - c)"),
        ("-x^2", "(-(x ^ 2))"),
        ("2^-x", "(2 ^ (-x))"),
        ("2^3^2", "(2 ^ (3 ^ 2))"),
        ("x^2^-1", "(x ^ (2 ^ (-1)))"),
        ("--x", "(-(-x))"),
        ("2x", "(2 * x)"),
        ("2 pi r", "((2 * pi) * r)"),
        ("a sin(x)", "(a * sin(x))"),
        ("3(x + 1)", "(3 * (x + 1))"),
        ("(x + 1)(x - 1)", "((x + 1) * (x - 1))"),
        ("2x^2", "(2 * (x ^ 2))"),
        ("x/2y", "((x / 2) * y)"),
        ("2 -x", "(2 - x)"),
        ("f'(x)", "f'(x)"),
        ("log(x, 2)", "log(x, 2)"),
        ("P.x + P.y", "(P.x + P.y)"),
        ("(1, 2)", "<1, 2>"),
        ("(cos(t), sin(t))", "<cos(t), sin(t)>"),
        ("x < 1 + 2", "(x < (1 + 2))"),
        ("x^2 + y^2 <= 4", "(((x ^ 2) + (y ^ 2)) <= 4)"),
    ],
)
def test_precedence(text: str, tree: str) -> None:
    assert show(parse(text)) == tree


def test_equations_split_at_the_equals_sign() -> None:
    equation = parse_equation("x^2 + y^2 = 9")
    assert show(equation.lhs) == "((x ^ 2) + (y ^ 2))"
    assert equation.rhs is not None
    assert show(equation.rhs) == "9"
    assert equation.equals == 10
    single = parse_equation("x + 1")
    assert single.rhs is None
    assert single.equals is None


def test_parse_refuses_an_equation() -> None:
    with pytest.raises(ExpressionError, match="="):
        parse("y = x")


@pytest.mark.parametrize(
    "text",
    [
        "y = (x+1)(x-1)",
        "(x + 1)^2 = 3",
        "f(x) = -(x)",
        "((x)) + ((1, 2))",
        "2(x - 1)^2 + y",
    ],
)
def test_spans_give_back_the_source(text: str) -> None:
    """Slicing the text by a node's span gives its source, parentheses included."""
    equation = parse_equation(text)
    for node in (equation.lhs, equation.rhs):
        if node is None:
            continue
        part = text[node.start : node.end]
        assert show(parse(part)) == show(node)


def test_references_lists_every_name() -> None:
    assert references(parse("a sin(x - P.x) + f'(b) + pi")) == {
        "a",
        "sin",
        "x",
        "P",
        "f",
        "b",
        "pi",
    }


# --- Errors -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "message", "position"),
    [
        ("", "empty", None),
        ("2 +", "expected a number, a name or '('", 3),
        ("x 2", "unexpected number 2", 2),
        ("(x", "expected ')'", 2),
        ("x)", "unexpected ')'", 1),
        ("sin()", "needs at least one argument", 4),
        ("f'", "needs an argument", 2),
        ("x < 1 < 2", "chained comparisons", 6),
        ("a = b = c", "unexpected '='", 6),
        ("P.x'", "cannot have primes", 2),
        ("f(x).y", "only points have coordinates", 4),
    ],
)
def test_syntax_errors(text: str, message: str, position: int | None) -> None:
    with pytest.raises(ExpressionError, match=re.escape(message)) as caught:
        parse_equation(text)
    error = caught.value
    assert error.text == text
    if position is not None:
        assert error.position == position


def test_errors_show_where_they_are() -> None:
    with pytest.raises(ExpressionError) as caught:
        parse("1 + * 2")
    lines = str(caught.value).splitlines()
    assert lines[1].strip() == "1 + * 2"
    assert lines[2].index("^") - lines[1].index("1") == 4


def test_deep_nesting_is_an_error_not_a_crash() -> None:
    parse("(" * (MAX_DEPTH - 2) + "x" + ")" * (MAX_DEPTH - 2))
    for text in (
        "(" * 10_000 + "x" + ")" * 10_000,
        "-" * 10_000 + "x",
        "2^" * 10_000 + "x",
    ):
        with pytest.raises(ExpressionError, match="nested too deeply"):
            parse(text)


@settings(max_examples=300, deadline=None)
@given(st.text(alphabet="x2.+-*/^()=,'<>ePπ² ", max_size=24))
def test_any_text_parses_or_fails_cleanly(text: str) -> None:
    """The parser never raises anything but an ExpressionError."""
    failure: ExpressionError | None = None
    try:
        parse_equation(text)
    except ExpressionError as error:
        failure = error
    if failure is not None:
        assert failure.text == text
        assert failure.position is None or 0 <= failure.position <= len(text)
