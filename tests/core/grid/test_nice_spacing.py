"""Tests for the 1-2-5 ladder (:mod:`nice`) and spacing selection (:mod:`spacing`)."""

from __future__ import annotations

import itertools
import math
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pygridplot.core.grid.nice import (
    NiceStep,
    common_step,
    level_boundary,
    select_level,
)
from pygridplot.core.grid.spacing import (
    SnapTracker,
    SpacingChoice,
    choose_faded,
    smoothstep,
)


@pytest.mark.parametrize(
    ("level", "mantissa", "exponent", "value"),
    [
        (0, 1, 0, Decimal(1)),
        (1, 2, 0, Decimal(2)),
        (2, 5, 0, Decimal(5)),
        (3, 1, 1, Decimal(10)),
        (-1, 5, -1, Decimal("0.5")),
        (-2, 2, -1, Decimal("0.2")),
        (-3, 1, -1, Decimal("0.1")),
        (-16, 5, -6, Decimal("0.000005")),
    ],
)
def test_ladder_levels(
    level: int, mantissa: int, exponent: int, value: Decimal
) -> None:
    step = NiceStep(level)
    assert (step.mantissa, step.exponent) == (mantissa, exponent)
    assert step.decimal == value
    assert step.value == float(value)
    assert NiceStep.from_parts(mantissa, exponent) == step


def test_from_parts_rejects_non_nice_mantissas() -> None:
    with pytest.raises(ValueError, match="mantissa"):
        NiceStep.from_parts(3, 0)


@pytest.mark.parametrize(
    ("level", "minor", "subdivisions"),
    [(0, "0.2", 5), (1, "0.5", 4), (2, "1", 5), (-3, "0.02", 5)],
)
def test_minor_steps_and_subdivisions(
    level: int, minor: str, subdivisions: int
) -> None:
    step = NiceStep(level)
    assert step.minor.decimal == Decimal(minor)
    assert step.subdivisions == subdivisions


def test_multiples() -> None:
    assert NiceStep.from_parts(1, 1).multiple_of(NiceStep.from_parts(2, 0)) == 5
    assert NiceStep.from_parts(5, 0).multiple_of(NiceStep.from_parts(1, -1)) == 50
    assert not NiceStep.from_parts(5, 0).is_multiple_of(NiceStep.from_parts(2, 0))
    assert not NiceStep(0).is_multiple_of(NiceStep(1))
    with pytest.raises(ValueError, match="multiple"):
        NiceStep.from_parts(5, 0).multiple_of(NiceStep.from_parts(2, 0))


@given(st.integers(-40, 40), st.integers(-40, 40))
def test_common_step_is_the_greatest_common_ladder_divisor(a: int, b: int) -> None:
    sa, sb = NiceStep(a), NiceStep(b)
    g = common_step(sa, sb)
    assert sa.is_multiple_of(g)
    assert sb.is_multiple_of(g)
    # No coarser ladder step divides both.
    coarser = g.coarser()
    assert not (sa.is_multiple_of(coarser) and sb.is_multiple_of(coarser))


def test_common_step_examples() -> None:
    assert str(common_step(NiceStep(-2), NiceStep(-1))) == "0.1"  # 0.2, 0.5
    assert str(common_step(NiceStep(2), NiceStep(3))) == "5"  # 5, 10
    assert str(common_step(NiceStep(1), NiceStep(3))) == "2"  # 2, 10


@pytest.mark.parametrize(
    ("raw", "nearest", "ceil"),
    [
        (1.0, "1", "1"),
        (1.4, "1", "2"),
        (1.42, "2", "2"),
        (3.1, "2", "5"),
        (3.2, "5", "5"),
        (7.0, "5", "10"),
        (7.1, "10", "10"),
        (0.03, "0.02", "0.05"),
        (2e7, "20000000", "20000000"),
    ],
)
def test_select_level_rules(raw: float, nearest: str, ceil: str) -> None:
    assert str(NiceStep(select_level(raw, "nearest"))) == nearest
    assert str(NiceStep(select_level(raw, "ceil"))) == ceil


def test_select_level_rejects_bad_input() -> None:
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="positive"):
            select_level(bad)


@given(st.floats(1e-12, 1e12))
def test_nearest_rule_keeps_spacing_within_bounds(raw: float) -> None:
    value = NiceStep(select_level(raw, "nearest")).value
    ratio = value / raw
    assert 1 / math.sqrt(2.5) - 1e-9 <= ratio <= math.sqrt(2.5) + 1e-9


@given(st.floats(1e-12, 1e12))
def test_ceil_rule_never_rounds_down(raw: float) -> None:
    value = NiceStep(select_level(raw, "ceil")).value
    assert raw * (1 - 1e-12) <= value <= raw * 2.5 * (1 + 1e-12)


def test_level_boundaries_are_increasing() -> None:
    for rule in ("nearest", "ceil"):
        bounds = [level_boundary(level, rule) for level in range(-10, 10)]
        assert all(b2 > b1 for b1, b2 in itertools.pairwise(bounds))


def test_smoothstep() -> None:
    assert smoothstep(-1) == 0.0
    assert smoothstep(0) == 0.0
    assert smoothstep(0.5) == 0.5
    assert smoothstep(1) == 1.0
    assert smoothstep(2) == 1.0


def test_spacing_choice_validation_and_weights() -> None:
    assert SpacingChoice(NiceStep(0)).weighted_steps() == ((NiceStep(0), 1.0),)
    blended = SpacingChoice(NiceStep(0), NiceStep(1), 0.25)
    assert blended.weighted_steps() == ((NiceStep(0), 0.75), (NiceStep(1), 0.25))
    assert blended.dominant == NiceStep(0)
    assert SpacingChoice(NiceStep(0), NiceStep(1), 0.5).dominant == NiceStep(1)
    with pytest.raises(ValueError, match="blend"):
        SpacingChoice(NiceStep(0), None, 0.5)
    with pytest.raises(ValueError, match="blend"):
        SpacingChoice(NiceStep(0), NiceStep(1), 1.5)


def test_choose_faded_outside_bands_is_a_single_level() -> None:
    # raw = 100 / 50 = 2 sits exactly on a ladder value: far from any boundary.
    choice = choose_faded(50.0, 100.0, "nearest", 0.05)
    assert choice == SpacingChoice(NiceStep(1))


def test_choose_faded_is_continuous_across_a_boundary() -> None:
    boundary = level_boundary(0, "nearest")  # between 1 and 2
    band = 0.05
    samples = [boundary + band * t for t in (-1.2, -1.0, -0.5, 0.0, 0.5, 1.0, 1.2)]
    weights = []
    for u in samples:
        scale = 100.0 / 10**u
        choice = choose_faded(scale, 100.0, "nearest", band)
        weights_by_level = {step.level: w for step, w in choice.weighted_steps()}
        coarse_weight = weights_by_level.get(1, 0.0)
        weights.append(coarse_weight)
    assert weights[0] == 0.0
    assert weights[1] == pytest.approx(0.0, abs=1e-9)
    assert weights[3] == pytest.approx(0.5)
    assert weights[-2] == pytest.approx(1.0, abs=1e-9)
    assert weights[-1] == 1.0
    assert all(b >= a for a, b in itertools.pairwise(weights))


def test_choose_faded_band_validation() -> None:
    with pytest.raises(ValueError, match="band"):
        choose_faded(50.0, 100.0, "nearest", 0.2)
    assert choose_faded(50.0, 100.0, "nearest", 0.0).coarse is None


def test_snap_tracker_hysteresis() -> None:
    tracker = SnapTracker()
    boundary = level_boundary(0, "nearest")  # between 1 and 2
    h = 0.04

    def choose(u: float) -> int:
        return tracker.choose(100.0 / 10**u, 100.0, "nearest", h).fine.level

    assert choose(boundary - 0.01) == 0
    assert choose(boundary + 0.01) == 0  # inside the hysteresis band: stays
    assert choose(boundary + 0.03) == 0
    assert choose(boundary + 0.05) == 1  # beyond it: switches
    assert choose(boundary - 0.01) == 1  # and stays switched on the way back
    assert choose(boundary - 0.05) == 0
    tracker.reset()
    assert tracker.level is None
