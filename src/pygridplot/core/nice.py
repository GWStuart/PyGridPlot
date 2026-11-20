"""Exact "nice" grid spacings: the 1-2-5 ladder.

Grid spacings are always ``{1, 2, 5} × 10^k``. Rather than storing them as
lossy floats, each spacing is identified by an integer **ladder level**::

    level  ...  -3    -2    -1    0    1    2    3    4    5  ...
    step   ...  0.1   0.2   0.5   1    2    5    10   20   50 ...

``mantissa = (1, 2, 5)[level % 3]`` and ``exponent = level // 3``. Because the
level is an integer, all the questions the grid needs to ask can be answered
exactly with integer arithmetic: "one step coarser", "the minor step", "is
this step a whole multiple of that one", and so on.

A useful fact: the conventional minor subdivision of every ladder step is
exactly two levels finer (``1 → 0.2``, ``2 → 0.5``, ``5 → 1``), giving 5, 4
and 5 subdivisions respectively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Final, Literal, TypeAlias

MANTISSAS: Final = (1, 2, 5)
"""The mantissas of the nice-number ladder, in increasing order."""

SpacingRule: TypeAlias = Literal["nearest", "ceil"]
"""How a raw spacing is rounded to the ladder.

* ``"nearest"``: the ladder step closest to the raw spacing in log space,
  so on-screen spacing stays within a factor of about 1.58 of the target.
* ``"ceil"``: the smallest ladder step that is at least the raw spacing, so
  on-screen spacing is between 1 and 2.5 times the target.
"""

_LOG10_MANTISSAS: Final = tuple(math.log10(m) for m in MANTISSAS)


@dataclass(frozen=True, order=True, slots=True)
class NiceStep:
    """One step of the 1-2-5 ladder, identified exactly by its level.

    Attributes:
        level: Integer position on the ladder (``0`` is a spacing of 1).
    """

    level: int

    @classmethod
    def from_parts(cls, mantissa: int, exponent: int) -> NiceStep:
        """Build a step from its mantissa and power of ten.

        Args:
            mantissa: 1, 2 or 5.
            exponent: Power of ten.

        Returns:
            The step ``mantissa × 10^exponent``.

        Raises:
            ValueError: If ``mantissa`` is not 1, 2 or 5.
        """
        if mantissa not in MANTISSAS:
            msg = f"mantissa must be one of {MANTISSAS}, got {mantissa}"
            raise ValueError(msg)
        return cls(3 * exponent + MANTISSAS.index(mantissa))

    @property
    def mantissa(self) -> int:
        """The leading digit: 1, 2 or 5."""
        return MANTISSAS[self.level % 3]

    @property
    def exponent(self) -> int:
        """The power of ten: the step is ``mantissa × 10^exponent``."""
        return self.level // 3

    @property
    def decimal(self) -> Decimal:
        """The exact value of the step as a :class:`~decimal.Decimal`."""
        return Decimal(self.mantissa).scaleb(self.exponent)

    @property
    def value(self) -> float:
        """The step as the float nearest to its exact value."""
        return float(self.decimal)

    @property
    def log10(self) -> float:
        """``log10`` of the step's value."""
        return _LOG10_MANTISSAS[self.level % 3] + self.exponent

    @property
    def subdivisions(self) -> int:
        """How many minor intervals one major interval of this step contains.

        5 for mantissa 1, 4 for mantissa 2 and 5 for mantissa 5.
        """
        return self.multiple_of(self.minor)

    @property
    def minor(self) -> NiceStep:
        """The conventional minor step, always two ladder levels finer."""
        return NiceStep(self.level - 2)

    def coarser(self, levels: int = 1) -> NiceStep:
        """Return the step ``levels`` positions further up the ladder.

        Args:
            levels: How many ladder positions to move (may be negative).

        Returns:
            The coarser step.
        """
        return NiceStep(self.level + levels)

    def finer(self, levels: int = 1) -> NiceStep:
        """Return the step ``levels`` positions further down the ladder.

        Args:
            levels: How many ladder positions to move (may be negative).

        Returns:
            The finer step.
        """
        return NiceStep(self.level - levels)

    def multiple_of(self, base: NiceStep) -> int:
        """Return the exact integer ``q`` with ``self == q × base``.

        Args:
            base: The (finer or equal) step to divide by.

        Returns:
            The integer ratio between the two steps.

        Raises:
            ValueError: If ``self`` is not a whole multiple of ``base``.
        """
        if not self.is_multiple_of(base):
            msg = f"{self} is not a whole multiple of {base}"
            raise ValueError(msg)
        return self._units_of(base) // base.mantissa

    def is_multiple_of(self, base: NiceStep) -> bool:
        """Return whether this step is a whole multiple of ``base``.

        Args:
            base: The step to compare against.

        Returns:
            ``True`` if ``self / base`` is a positive integer.
        """
        if self.exponent < base.exponent:
            return False
        return self._units_of(base) % base.mantissa == 0

    def _units_of(self, base: NiceStep) -> int:
        """Return this step's mantissa scaled to ``base``'s power of ten.

        Args:
            base: A step with an exponent no larger than this step's.

        Returns:
            ``mantissa × 10^(exponent - base.exponent)`` as an exact integer.
        """
        shift: int = self.exponent - base.exponent
        scale: int = pow(10, shift)
        return self.mantissa * scale

    def __str__(self) -> str:
        return format(self.decimal.normalize(), "f")


def common_step(a: NiceStep, b: NiceStep) -> NiceStep:
    """Return the largest ladder step that both ``a`` and ``b`` are multiples of.

    This is the greatest common divisor of the two values. For any two
    ladder steps it is itself a ladder step, for example
    ``common_step(0.2, 0.5) == 0.1`` and ``common_step(5, 10) == 5``.

    Args:
        a: First step.
        b: Second step.

    Returns:
        The greatest common ladder divisor of ``a`` and ``b``.
    """
    low = min(a.exponent, b.exponent)
    units_a = a.mantissa * 10 ** (a.exponent - low)
    units_b = b.mantissa * 10 ** (b.exponent - low)
    divisor = math.gcd(units_a, units_b)
    # `divisor` is a 1-2-5 number times a power of ten: peel off the tens.
    exponent = low
    while divisor % 10 == 0:
        divisor //= 10
        exponent += 1
    return NiceStep.from_parts(divisor, exponent)


def level_boundary(level: int, rule: SpacingRule) -> float:
    """Return the ``log10`` raw spacing at which ``level`` gives way to ``level + 1``.

    A raw spacing ``r`` selects ``level`` when
    ``level_boundary(level - 1) < log10(r) <= level_boundary(level)``.

    Args:
        level: Ladder level.
        rule: The rounding rule (see :data:`SpacingRule`).

    Returns:
        The boundary in ``log10`` units.
    """
    here = NiceStep(level).log10
    if rule == "ceil":
        return here
    return (here + NiceStep(level + 1).log10) / 2


def select_level(raw_spacing: float, rule: SpacingRule = "nearest") -> int:
    """Round a raw world spacing to a ladder level.

    Args:
        raw_spacing: Desired spacing in world units (must be positive).
        rule: The rounding rule (see :data:`SpacingRule`).

    Returns:
        The selected ladder level.

    Raises:
        ValueError: If ``raw_spacing`` is not a positive finite number.
    """
    if not (raw_spacing > 0 and math.isfinite(raw_spacing)):
        msg = f"raw spacing must be positive and finite, got {raw_spacing!r}"
        raise ValueError(msg)
    target = math.log10(raw_spacing)
    level = 3 * math.floor(target)
    while target > level_boundary(level, rule):
        level += 1
    while target <= level_boundary(level - 1, rule):
        level -= 1
    return level
