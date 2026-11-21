"""Splitting expression text into tokens.

The tokens are numbers (``2``, ``0.5``, ``.5``, ``1e-3``), names (``x``,
``a2``, ``theta``, ``π``, with any primes after them counted, as in
``f''``), and operators. A few Unicode symbols that maths text often
contains are read as their ASCII equivalents: ``−`` as ``-``, ``×`` and
``·`` as ``*``, ``÷`` as ``/``, ``≤ ≥ ≠`` as ``<= >= !=``, and a
superscript number as a power: ``x²`` as ``x^2``, ``x¹⁰`` as ``x^10`` and
``x⁻¹`` as ``x^-1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pygridplot.core.expr.errors import ExpressionError

TokenKind: TypeAlias = Literal["number", "name", "operator", "end"]
"""The kinds of token."""

DIGITS: Final = "0123456789"

OPERATORS: Final = frozenset(
    {"+", "-", "*", "/", "^", "(", ")", ",", ".", "=", "<", ">"}
)
"""Single-character operators."""

DOUBLE_OPERATORS: Final = frozenset({"**", "<=", ">=", "==", "!="})
"""Two-character operators (``**`` is a synonym of ``^``)."""

ALIASES: Final = {
    "−": "-",
    "×": "*",
    "·": "*",
    "⋅": "*",
    "÷": "/",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
}
"""Unicode operators and their ASCII equivalents."""

SUPERSCRIPTS: Final = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹", DIGITS, strict=True))
"""Superscript digits: a run of them (after an optional ``⁻`` or ``⁺``) is a power."""

SUPERSCRIPT_SIGNS: Final = {"⁻": "-", "⁺": "+"}
"""Superscript signs, allowed at the start of a superscript power."""


@dataclass(frozen=True, slots=True)
class Token:
    """One token of an expression.

    Attributes:
        kind: ``"number"``, ``"name"``, ``"operator"`` or ``"end"``.
        text: The number's text, the name, or the operator in ASCII
            (``"^"`` for ``**``, ``"*"`` for ``×``).
        start: Index of the token's first character.
        end: Index just past its last character.
        value: The value of a number.
        primes: How many primes (``'``) follow a name.
    """

    kind: TokenKind
    text: str
    start: int
    end: int
    value: float = 0.0
    primes: int = 0


def tokenize(text: str) -> list[Token]:
    """Split an expression into tokens, ending with an ``"end"`` token.

    Args:
        text: The expression.

    Returns:
        The tokens.

    Raises:
        ExpressionError: For a character that cannot start a token.
    """
    tokens: list[Token] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c in DIGITS or (c == "." and i + 1 < n and text[i + 1] in DIGITS):
            j = _scan_number(text, i)
            tokens.append(Token("number", text[i:j], i, j, float(text[i:j])))
            i = j
        elif c.isalpha() or c == "_":
            j = i + 1
            while j < n and _is_name_char(text[j]):
                j += 1
            k = j
            while k < n and text[k] == "'":
                k += 1
            tokens.append(Token("name", text[i:j], i, k, primes=k - j))
            i = k
        elif c in SUPERSCRIPTS or c in SUPERSCRIPT_SIGNS:
            i = _scan_superscript(text, i, tokens)
        elif c in ALIASES:
            tokens.append(Token("operator", ALIASES[c], i, i + 1))
            i += 1
        elif text[i : i + 2] in DOUBLE_OPERATORS:
            pair = text[i : i + 2]
            tokens.append(Token("operator", "^" if pair == "**" else pair, i, i + 2))
            i += 2
        elif c in OPERATORS:
            tokens.append(Token("operator", c, i, i + 1))
            i += 1
        else:
            msg = f"unexpected character {c!r}"
            raise ExpressionError(msg, text, i)
    tokens.append(Token("end", "", n, n))
    return tokens


def _scan_superscript(text: str, i: int, tokens: list[Token]) -> int:
    """Add the tokens of a superscript power (``²``, ``¹⁰``, ``⁻¹``) at ``i``.

    Returns:
        The index just past the power.

    Raises:
        ExpressionError: For a superscript sign without digits after it.
    """
    n = len(text)
    tokens.append(Token("operator", "^", i, i))
    j = i
    if text[j] in SUPERSCRIPT_SIGNS:
        tokens.append(Token("operator", SUPERSCRIPT_SIGNS[text[j]], j, j + 1))
        j += 1
    k = j
    while k < n and text[k] in SUPERSCRIPTS:
        k += 1
    if k == j:
        msg = "a superscript sign needs superscript digits after it, as in x⁻¹"
        raise ExpressionError(msg, text, i)
    digits = "".join(SUPERSCRIPTS[c] for c in text[j:k])
    tokens.append(Token("number", digits, j, k, float(digits)))
    return k


def _is_name_char(c: str) -> bool:
    """Return whether a character can continue a name (letters, digits, ``_``).

    Only ASCII digits count, so that ``x²`` is ``x`` squared, not a name.
    """
    return c.isalpha() or c in DIGITS or c == "_"


def _scan_number(text: str, i: int) -> int:
    """Return the index just past the number starting at ``i``.

    A number is digits with an optional fraction and an optional exponent
    (``e`` or ``E``, an optional sign and digits). An ``e`` not followed by
    digits is not part of the number, so ``2e`` reads as ``2`` times ``e``.
    """
    n = len(text)
    j = i
    while j < n and text[j] in DIGITS:
        j += 1
    if j < n and text[j] == ".":
        j += 1
        while j < n and text[j] in DIGITS:
            j += 1
    if j < n and text[j] in "eE":
        k = j + 1
        if k < n and text[k] in "+-":
            k += 1
        if k < n and text[k] in DIGITS:
            while k < n and text[k] in DIGITS:
                k += 1
            j = k
    return j
