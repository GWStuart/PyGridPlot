"""Tests for exact tick label formatting (:mod:`pygridplot.core.grid.formatting`)."""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pygridplot.core.grid.formatting import (
    MINUS_SIGN,
    NumberFormat,
    choose_notation,
    format_log_value,
    format_tick,
    format_value,
    magnitude_exponent,
    tick_value,
)
from pygridplot.core.grid.nice import NiceStep

FMT = NumberFormat()
M = MINUS_SIGN


@pytest.mark.parametrize(
    ("index", "level", "expected"),
    [
        (1, -3, "0.1"),
        (2, -3, "0.2"),
        (3, -3, "0.3"),  # never 0.30000000000000004
        (10, -3, "1"),  # trailing zeros dropped
        (15, -3, "1.5"),
        (-3, -3, f"{M}0.3"),
        (0, -3, "0"),
        (7, 0, "7"),
        (-12, 1, f"{M}24"),
        (3, 3, "30"),
        (5, 15, "500000"),
        (1, -12, "0.0001"),
    ],
)
def test_fixed_notation(index: int, level: int, expected: str) -> None:
    assert format_tick(index, NiceStep(level), FMT).plain == expected


def test_zero_is_never_negative() -> None:
    for level in range(-20, 20):
        assert format_tick(0, NiceStep(level), FMT).plain == "0"
    assert format_value(Decimal("-0"), FMT).plain == "0"
    assert format_value(Decimal("-0.000"), FMT).plain == "0"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1000000", "1×10^6"),
        ("999999", "999999"),
        ("2500000", "2.5×10^6"),
        ("-1000000", f"{M}1×10^6"),
        ("0.00005", f"5×10^{M}5"),
        ("0.0001", "0.0001"),
        ("0.000125", "0.000125"),  # 10^-4 <= |v|: still fixed
        ("0.0000125", f"1.25×10^{M}5"),
        ("-0.00002", f"{M}2×10^{M}5"),
        ("1.23E+15", "1.23×10^15"),
    ],
)
def test_scientific_notation_thresholds(value: str, expected: str) -> None:
    assert format_value(Decimal(value), FMT).plain == expected


def test_runs_have_roles() -> None:
    label = format_value(Decimal("-2E+6"), FMT)
    roles = [run.role for run in label.runs]
    assert roles == ["sign", "text", "superscript"]
    assert label.has_sign
    assert label.has_superscript
    assert not format_value(Decimal(3), FMT).has_sign


def test_custom_format() -> None:
    fmt = NumberFormat(
        sci_large_exponent=3, sci_small_exponent=-1, minus="-", times="x"
    )
    assert format_value(Decimal(1500), fmt).plain == "1.5x10^3"
    assert format_value(Decimal("-0.05"), fmt).plain == "-5x10^-2"
    assert format_value(Decimal("0.5"), fmt).plain == "0.5"
    with pytest.raises(ValueError, match="sci_small_exponent"):
        NumberFormat(sci_large_exponent=-5, sci_small_exponent=0)


def test_non_finite_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        format_value(Decimal("NaN"), FMT)


@given(st.integers(-(10**6), 10**6), st.integers(-30, 30))
def test_tick_values_are_exact(index: int, level: int) -> None:
    step = NiceStep(level)
    assert tick_value(index, step) == index * step.decimal


@given(st.integers(-(10**6), 10**6), st.integers(-30, 30))
def test_labels_contain_no_float_junk(index: int, level: int) -> None:
    text = format_tick(index, NiceStep(level), FMT).plain
    assert "e" not in text.lower()
    assert "000000000" not in text
    assert text != f"{M}0"
    # Fixed-notation labels parse back to the exact value.
    if "×" not in text:
        assert Decimal(text.replace(M, "-")) == tick_value(index, NiceStep(level))


@pytest.mark.parametrize(
    ("largest", "level", "expected"),
    [
        ("1000000", 15, "scientific"),  # step 1e5: "1.1×10⁶" beats "1100000"
        ("1000000", 12, "fixed"),  # step 1e4: "1010000" is shorter
        ("1000000.1228", -12, "fixed"),  # never "1.0000001228×10⁶"
        ("500000000000000", 42, "scientific"),
        ("0.00002", -15, "scientific"),  # step 1e-5
        ("0.0002", -14, "fixed"),  # inside the fixed range
        ("123", 0, "fixed"),
        ("0", 0, "fixed"),
    ],
)
def test_choose_notation(largest: str, level: int, expected: str) -> None:
    assert choose_notation(Decimal(largest), NiceStep(level), FMT) == expected


def test_forced_notation() -> None:
    assert format_value(Decimal(500000), FMT, "scientific").plain == "5×10^5"
    assert format_value(Decimal(2000000), FMT, "fixed").plain == "2000000"
    assert format_value(Decimal(0), FMT, "scientific").plain == "0"


def test_magnitude_exponent() -> None:
    assert magnitude_exponent(Decimal("123.4")) == 2
    assert magnitude_exponent(Decimal("0.00123")) == -3
    assert magnitude_exponent(Decimal("1E+6")) == 6


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", "1"),
        ("0.01", "0.01"),
        ("100000", "100000"),
        ("0.0001", "0.0001"),
        ("1E+6", "10^6"),
        ("1E-5", f"10^{MINUS_SIGN}5"),
        ("2E+6", "2×10^6"),
        ("1.5E+7", "1.5×10^7"),
        ("3E-12", f"3×10^{MINUS_SIGN}12"),
        ("1E+300", "10^300"),
        ("0.5", "0.5"),
    ],
)
def test_log_labels_choose_their_notation_one_by_one(value: str, expected: str) -> None:
    assert format_log_value(Decimal(value), NumberFormat()).plain == expected


def test_log_labels_follow_the_number_format() -> None:
    fmt = NumberFormat(sci_large_exponent=4, sci_small_exponent=-3)
    assert format_log_value(Decimal(10000), fmt).plain == "10^4"
    assert format_log_value(Decimal(1000), fmt).plain == "1000"
    assert format_log_value(Decimal("0.001"), fmt).plain == "0.001"
    assert format_log_value(Decimal("0.0001"), fmt).plain == f"10^{MINUS_SIGN}4"
