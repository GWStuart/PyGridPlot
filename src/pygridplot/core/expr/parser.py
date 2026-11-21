"""Parsing expressions into syntax trees.

The grammar, from the loosest binding to the tightest::

    equation    := expression ("=" expression)?
    expression  := additive (comparison additive)?       < <= > >= == !=
    additive    := term (("+" | "-") term)*
    term        := unary (("*" | "/") unary | implicit)*
    implicit    := power                                 (no operator, see below)
    unary       := ("-" | "+") unary | power
    power       := postfix (("^" | "**") unary)?          (right-associative)
    postfix     := primary ("." name)?                   P.x, P.y
    primary     := number | name primes? call? | "(" expression ("," expression)* ")"
    call        := "(" expression ("," expression)* ")"

So ``-x^2`` is ``-(x^2)``, ``2^-x`` is ``2^(-x)``, ``2^3^2`` is
``2^(3^2)``, and ``a/b*c`` is ``(a/b)*c``.

**Implicit multiplication**: a term followed directly by a name or an
opening parenthesis is multiplied by it, at the same precedence as ``*``:
``2x``, ``3(x + 1)``, ``(x + 1)(x - 1)``, ``a sin(x)``, ``2pi r``. A
number after a term is not (``x 2`` is an error), and neither is a minus
sign (``2 -x`` is a subtraction). ``name(...)`` is parsed as a call; when
the name turns out to be a number rather than a function (``a(x + 1)``),
the compiler multiplies instead.

A parenthesised list of two or more expressions is a tuple, the notation
for points and parametric curves: ``(cos(t), sin(t))``.

Every node records where it is in the text (``start`` and ``end``, so that
``text[node.start:node.end]`` is its source); the span of a parenthesised
expression includes its parentheses, so slicing a product such as
``(x + 1)(x - 1)`` gives it back whole.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final, TypeAlias

from pygridplot.core.expr.errors import ExpressionError
from pygridplot.core.expr.lexer import Token, tokenize

MAX_DEPTH: Final = 120
"""Deepest nesting accepted (parentheses, powers, signs), so that evaluating
a compiled expression can never exhaust Python's recursion limit."""

COMPARISONS: Final = frozenset({"<", "<=", ">", ">=", "==", "!="})


@dataclass(frozen=True, slots=True)
class Number:
    """A number literal."""

    value: float
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Name:
    """A name: a variable, a constant, a parameter."""

    name: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Attribute:
    """A coordinate of a point: ``P.x`` or ``P.y``."""

    target: str
    attribute: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Call:
    """A function call, ``f(a, b)``, or its derivative, ``f'(x)``."""

    name: str
    primes: int
    args: tuple[Node, ...]
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Unary:
    """A sign: ``-a`` or ``+a``."""

    op: str
    operand: Node
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Binary:
    """An arithmetic operation: ``+ - * / ^``."""

    op: str
    left: Node
    right: Node
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Compare:
    """A comparison: ``< <= > >= == !=`` (true is 1, false is 0)."""

    op: str
    left: Node
    right: Node
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Tuple:
    """A parenthesised list of two or more expressions: a point or a curve."""

    items: tuple[Node, ...]
    start: int
    end: int


Node: TypeAlias = Number | Name | Attribute | Call | Unary | Binary | Compare | Tuple
"""Any node of a syntax tree."""


@dataclass(frozen=True, slots=True)
class Equation:
    """A parsed line: an expression, or two sides of an equation.

    Attributes:
        lhs: The expression, or the left-hand side.
        rhs: The right-hand side, or ``None`` for a plain expression.
        equals: Index of the ``=`` in the text, or ``None``.
    """

    lhs: Node
    rhs: Node | None
    equals: int | None


def parse(text: str) -> Node:
    """Parse an expression (no ``=``).

    Args:
        text: The expression.

    Returns:
        Its syntax tree.

    Raises:
        ExpressionError: For a syntax error.
    """
    equation = parse_equation(text)
    if equation.rhs is not None:
        assert equation.equals is not None
        msg = "unexpected '=' (write an expression, not an equation)"
        raise ExpressionError(msg, text, equation.equals)
    return equation.lhs


def parse_equation(text: str) -> Equation:
    """Parse an expression or an equation ``lhs = rhs``.

    Args:
        text: The line.

    Returns:
        The parsed line.

    Raises:
        ExpressionError: For a syntax error.
    """
    parser = _Parser(text)
    lhs = parser.expression()
    rhs: Node | None = None
    equals: int | None = None
    if parser.peek_operator("="):
        equals = parser.advance().start
        rhs = parser.expression()
    parser.expect_end()
    return Equation(lhs, rhs, equals)


class _Parser:
    """A recursive-descent parser over the tokens of one text."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = tokenize(text)
        self.index = 0
        self.depth = 0
        if self.tokens[0].kind == "end":
            msg = "the expression is empty"
            raise ExpressionError(msg, text, 0)

    # --- Tokens --------------------------------------------------------
    def peek(self) -> Token:
        return self.tokens[self.index]

    def peek_operator(self, *ops: str) -> bool:
        token = self.tokens[self.index]
        return token.kind == "operator" and token.text in ops

    def advance(self) -> Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def expect_operator(self, op: str, context: str) -> Token:
        if not self.peek_operator(op):
            self.fail(f"expected '{op}' {context}")
        return self.advance()

    def expect_end(self) -> None:
        if self.peek().kind != "end":
            self.fail(f"unexpected {_describe(self.peek())}")

    def fail(self, message: str, position: int | None = None) -> None:
        where = self.peek().start if position is None else position
        raise ExpressionError(message, self.text, where)

    def enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            self.fail("the expression is nested too deeply")

    # --- Grammar -------------------------------------------------------
    def expression(self) -> Node:
        left = self.additive()
        if self.peek_operator(*COMPARISONS):
            op = self.advance().text
            right = self.additive()
            if self.peek_operator(*COMPARISONS):
                self.fail("chained comparisons are not supported (use and(a, b))")
            return Compare(op, left, right, left.start, right.end)
        return left

    def additive(self) -> Node:
        node = self.term()
        while self.peek_operator("+", "-"):
            op = self.advance().text
            right = self.term()
            node = Binary(op, node, right, node.start, right.end)
        return node

    def term(self) -> Node:
        node = self.unary()
        while True:
            if self.peek_operator("*", "/"):
                op = self.advance().text
                right = self.unary()
                node = Binary(op, node, right, node.start, right.end)
            elif self.peek().kind == "name" or self.peek_operator("("):
                right = self.power()  # implicit multiplication
                node = Binary("*", node, right, node.start, right.end)
            else:
                return node

    def unary(self) -> Node:
        if self.peek_operator("-", "+"):
            token = self.advance()
            self.enter()
            operand = self.unary()
            self.depth -= 1
            return Unary(token.text, operand, token.start, operand.end)
        return self.power()

    def power(self) -> Node:
        base = self.postfix()
        if self.peek_operator("^"):
            self.advance()
            self.enter()
            exponent = self.unary()
            self.depth -= 1
            return Binary("^", base, exponent, base.start, exponent.end)
        return base

    def postfix(self) -> Node:
        node = self.primary()
        if self.peek_operator("."):
            dot = self.advance()
            if not isinstance(node, Name):
                self.fail("only points have coordinates (.x and .y)", dot.start)
            assert isinstance(node, Name)
            if self.peek().kind != "name":
                self.fail("expected x or y after '.'")
            attribute = self.advance()
            if attribute.primes:
                self.fail("a coordinate cannot have primes", attribute.start)
            return Attribute(node.name, attribute.text, node.start, attribute.end)
        return node

    def primary(self) -> Node:
        token = self.peek()
        if token.kind == "number":
            self.advance()
            return Number(token.value, token.start, token.end)
        if token.kind == "name":
            self.advance()
            if self.peek_operator("("):
                args, end = self.arguments(token.text)
                return Call(token.text, token.primes, args, token.start, end)
            if token.primes:
                name = token.text + "'" * token.primes
                self.fail(f"{name} needs an argument, as in {name}(x)", token.end)
            return Name(token.text, token.start, token.end)
        if self.peek_operator("("):
            opening = self.advance()
            self.enter()
            items = [self.expression()]
            while self.peek_operator(","):
                self.advance()
                items.append(self.expression())
            closing = self.expect_operator(")", "to close the parenthesis")
            self.depth -= 1
            if len(items) == 1:  # a group: its text includes the parentheses
                return replace(items[0], start=opening.start, end=closing.end)
            return Tuple(tuple(items), opening.start, closing.end)
        self.fail(f"expected a number, a name or '(', not {_describe(token)}")
        raise AssertionError  # unreachable: fail always raises

    def arguments(self, name: str) -> tuple[tuple[Node, ...], int]:
        self.advance()  # "("
        self.enter()
        if self.peek_operator(")"):
            self.fail(f"{name}() needs at least one argument")
        args = [self.expression()]
        while self.peek_operator(","):
            self.advance()
            args.append(self.expression())
        closing = self.expect_operator(")", f"to close the arguments of {name}")
        self.depth -= 1
        return tuple(args), closing.end


def _describe(token: Token) -> str:
    """Name a token for an error message."""
    if token.kind == "end":
        return "end of expression"
    if token.kind == "number":
        return f"number {token.text}"
    if token.kind == "name":
        return f"name {token.text!r}"
    return f"'{token.text}'"


def references(node: Node) -> set[str]:
    """Return every name a syntax tree refers to (variables included).

    Args:
        node: The tree.

    Returns:
        The names of values, functions and points it uses.
    """
    found: set[str] = set()
    stack: list[Node] = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, Name):
            found.add(current.name)
        elif isinstance(current, Attribute):
            found.add(current.target)
        elif isinstance(current, Call):
            found.add(current.name)
            stack.extend(current.args)
        elif isinstance(current, Unary):
            stack.append(current.operand)
        elif isinstance(current, (Binary, Compare)):
            stack.extend((current.left, current.right))
        elif isinstance(current, Tuple):
            stack.extend(current.items)
    return found
