"""Exact, clean number formatting for tick labels.

Labels are built from *exact* decimal values rather than floats. A grid line
is the ``n``-th multiple of a ladder step ``m × 10^k``, so its value is
exactly ``n·m × 10^k``. Using :class:`~decimal.Decimal` keeps it that way, so
labels such as ``0.30000000000000004`` or ``-0`` can never appear.

A formatted label is a :class:`LabelText`: a short sequence of
:class:`TextRun` pieces. Scientific notation keeps its exponent in a separate
superscript run (``"2×10"`` + superscript ``"6"``), and the renderer draws it
smaller and raised. This looks like typeset maths and does not depend on the
font having Unicode superscript glyphs.

Grids choose one :data:`Notation` per axis with :func:`choose_notation`,
so the labels along an axis never mix ``500000`` with ``1×10⁶``. Scientific
notation is only used for very large or very small magnitudes, and only when
it is actually shorter. Near ``10⁶`` with a fine spacing, ``1000000.1228``
beats ``1.0000001228×10⁶``. Within one frame a label's text depends only
on its value and the axis' notation. That matters when cross-fading
between two grid spacings: a label present in both keeps exactly the same
text and never flickers.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Final, Literal, TypeAlias

from pygridplot.core.nice import NiceStep

Notation: TypeAlias = Literal["fixed", "scientific"]
"""How a number is written: ``"fixed"`` (``1500``) or ``"scientific"`` (``1.5×10³``)."""

RunRole: TypeAlias = Literal["text", "sign", "superscript"]
"""How a :class:`TextRun` should be typeset.

* ``"text"``: normal text on the baseline.
* ``"sign"``: a leading minus sign. It is drawn like ``"text"`` but kept
  separate so layout can let it hang outside the digits.
* ``"superscript"``: a smaller, raised exponent.
"""

MINUS_SIGN = "−"
"""The typographic minus sign (U+2212), wider than a hyphen."""

TIMES_SIGN = "×"
"""The multiplication sign (U+00D7) used in scientific notation."""


@dataclass(frozen=True, slots=True)
class TextRun:
    """A piece of a label with a single typesetting role.

    Attributes:
        text: The characters in this run.
        role: How the run is typeset (see :data:`RunRole`).
    """

    text: str
    role: RunRole = "text"


@dataclass(frozen=True, slots=True)
class LabelText:
    """A formatted label made up of one or more :class:`TextRun` pieces.

    Attributes:
        runs: The pieces of the label in reading order.
    """

    runs: tuple[TextRun, ...]

    @property
    def plain(self) -> str:
        """The label as plain text, with ``^`` before superscripts.

        Useful for debugging and tests, e.g. ``"2×10^6"``.
        """
        return "".join(
            f"^{run.text}" if run.role == "superscript" else run.text
            for run in self.runs
        )

    @property
    def has_sign(self) -> bool:
        """Whether the label starts with a separate minus-sign run."""
        return bool(self.runs) and self.runs[0].role == "sign"

    @property
    def has_superscript(self) -> bool:
        """Whether any run of the label is a superscript."""
        return any(run.role == "superscript" for run in self.runs)


@dataclass(frozen=True, slots=True)
class NumberFormat:
    """Options controlling how tick values are written.

    Scientific notation is considered when the largest magnitude involved is
    at least ``10**sci_large_exponent``, or below ``10**sci_small_exponent``.
    In between, numbers are always written in fixed notation. With the
    defaults ``100000`` and ``0.0001`` stay as they are, while ``1000000``
    can become ``1×10⁶`` and ``0.00005`` can become ``5×10⁻⁵``.

    Attributes:
        sci_large_exponent: Magnitudes of ``10**sci_large_exponent`` and above
            may use scientific notation.
        sci_small_exponent: Magnitudes below ``10**sci_small_exponent`` may use
            scientific notation.
        minus: The character used for negative signs.
        times: The character used between mantissa and power of ten.
    """

    sci_large_exponent: int = 6
    sci_small_exponent: int = -4
    minus: str = MINUS_SIGN
    times: str = TIMES_SIGN

    def __post_init__(self) -> None:
        """Validate that the scientific-notation thresholds are ordered."""
        if self.sci_small_exponent > self.sci_large_exponent:
            msg = "sci_small_exponent must not exceed sci_large_exponent"
            raise ValueError(msg)


ZERO_LABEL = LabelText((TextRun("0"),))
"""The label for zero, shared by every format."""

SUPERSCRIPT_WIDTH: Final = 0.7
"""Approximate width of a superscript character relative to a normal one."""


def magnitude_exponent(value: Decimal) -> int:
    """Return ``floor(log10(|value|))`` exactly, for a non-zero finite value.

    Args:
        value: A non-zero, finite decimal.

    Returns:
        The power of ten of the value's leading digit.
    """
    _sign, digits, exponent = value.normalize().as_tuple()
    assert isinstance(exponent, int)  # finite values always have an int exponent
    return exponent + len(digits) - 1


def choose_notation(
    largest: Decimal, resolution: NiceStep, fmt: NumberFormat
) -> Notation:
    """Choose one notation for all the labels along an axis.

    Scientific notation is only considered when ``largest`` is outside the
    fixed range of ``fmt``. It is then chosen if it is no longer than fixed
    notation at the axis' resolution, so it is used to avoid long runs of
    zeros (``2000000``, ``0.00000003``), never to rewrite a value full of
    significant digits.

    Args:
        largest: The largest magnitude among the axis' labels.
        resolution: The spacing between labelled values (it sets how many
            digits each label needs).
        fmt: Formatting options.

    Returns:
        The notation to use for every non-zero label of the axis.
    """
    if largest.is_zero() or not largest.is_finite():
        return "fixed"
    power = magnitude_exponent(abs(largest))
    if fmt.sci_small_exponent <= power < fmt.sci_large_exponent:
        return "fixed"
    k = resolution.exponent
    significant = max(1, power - k + 1)
    fixed_length = power + 1 if power >= 0 else 1
    if k < 0:
        fixed_length += 1 - k  # the decimal point and the fractional digits
    exponent_chars = len(str(abs(power))) + (1 if power < 0 else 0)
    sci_length = (
        significant
        + (1 if significant > 1 else 0)
        + 3
        + SUPERSCRIPT_WIDTH * exponent_chars
    )
    return "scientific" if sci_length <= fixed_length else "fixed"


def tick_value(index: int, step: NiceStep) -> Decimal:
    """Return the exact value of the ``index``-th multiple of ``step``.

    Args:
        index: The integer multiplier.
        step: The ladder step.

    Returns:
        ``index × step`` as an exact :class:`~decimal.Decimal`.
    """
    return Decimal(index * step.mantissa).scaleb(step.exponent)


def format_tick(
    index: int, step: NiceStep, fmt: NumberFormat, notation: Notation | None = None
) -> LabelText:
    """Format the label of the ``index``-th multiple of ``step``.

    Args:
        index: The integer multiplier.
        step: The ladder step.
        fmt: Formatting options.
        notation: Force a notation (as chosen by :func:`choose_notation`), or
            ``None`` to decide from the value alone.

    Returns:
        The formatted label.
    """
    return format_value(tick_value(index, step), fmt, notation)


def format_value(
    value: Decimal, fmt: NumberFormat, notation: Notation | None = None
) -> LabelText:
    """Format an exact decimal value as a clean tick label.

    Trailing zeros are dropped (``1.50`` becomes ``1.5``), and zero is always
    plain ``0`` (never ``-0``).

    Args:
        value: The exact value to format. It must be finite.
        fmt: Formatting options.
        notation: Force a notation, or ``None`` to decide from the value
            alone: scientific outside ``fmt``'s fixed range, fixed inside it.

    Returns:
        The formatted label.

    Raises:
        ValueError: If ``value`` is not finite.
    """
    if not value.is_finite():
        msg = f"cannot format non-finite value {value!r}"
        raise ValueError(msg)
    if value.is_zero():
        return ZERO_LABEL
    magnitude = abs(value).normalize()
    _sign, digits, _exponent = magnitude.as_tuple()
    power = magnitude_exponent(magnitude)
    if notation is None:
        in_range = fmt.sci_small_exponent <= power < fmt.sci_large_exponent
        notation = "fixed" if in_range else "scientific"
    runs: list[TextRun] = []
    if value < 0:
        runs.append(TextRun(fmt.minus, "sign"))
    if notation == "fixed":
        runs.append(TextRun(format(magnitude, "f")))
    else:
        runs.extend(_scientific_runs(digits, power, fmt))
    return LabelText(tuple(runs))


def _scientific_runs(
    digits: tuple[int, ...], power: int, fmt: NumberFormat
) -> list[TextRun]:
    """Build the runs for ``d.ddd × 10^power``.

    Args:
        digits: Significant digits of the magnitude with no trailing zeros.
        power: The power of ten of the leading digit.
        fmt: Formatting options (for the minus and times signs).

    Returns:
        A mantissa run followed by a superscript exponent run.
    """
    leading = str(digits[0])
    rest = "".join(str(d) for d in digits[1:])
    mantissa = f"{leading}.{rest}" if rest else leading
    exponent = f"{fmt.minus}{-power}" if power < 0 else str(power)
    return [TextRun(f"{mantissa}{fmt.times}10"), TextRun(exponent, "superscript")]
