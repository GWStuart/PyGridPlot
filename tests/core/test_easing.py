"""Tests for the critically damped spring used for eased pan and zoom."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pygridplot.core.easing import Spring

SETTLE = {"settle_offset": 1e-6, "settle_velocity": 1e-5}


def run(spring: Spring, seconds: float, dt: float, tau: float) -> list[float]:
    """Advance a spring for ``seconds`` and return the per-step deltas."""
    return [spring.advance(dt, tau, **SETTLE) for _ in range(round(seconds / dt))]


def test_spring_reaches_the_target_exactly_without_overshoot() -> None:
    spring = Spring()
    spring.push(2.0)
    deltas = run(spring, 1.0, 1 / 60, 0.05)
    assert sum(deltas) == pytest.approx(2.0, abs=1e-12)
    assert all(d >= 0 for d in deltas)  # critically damped: never overshoots
    assert not spring.active


def test_spring_starts_gently() -> None:
    spring = Spring()
    spring.push(1.0)
    first, second, third = run(spring, 3 / 60, 1 / 60, 0.05)
    assert 0 < first < second  # accelerates: no instant jump to full speed
    assert third > 0


def test_ninety_percent_after_about_four_time_constants() -> None:
    spring = Spring()
    spring.push(1.0)
    travelled = list(itertools.accumulate(run(spring, 1.0, 0.001, 0.05)))
    t90 = next(i for i, v in enumerate(travelled) if v >= 0.9) * 0.001
    assert t90 == pytest.approx(3.89 * 0.05, rel=0.02)


@given(dt=st.floats(1e-4, 0.5))
def test_spring_is_stable_for_any_time_step(dt: float) -> None:
    spring = Spring()
    spring.push(1.0)
    total = 0.0
    for _ in range(round(3.0 / dt) + 2):
        total += spring.advance(dt, 0.05, **SETTLE)
        assert -1e-9 <= total <= 1.0 + 1e-9
    assert total == pytest.approx(1.0, abs=1e-6)


def test_a_stream_of_pushes_becomes_an_even_glide() -> None:
    spring = Spring()
    deltas = []
    for frame in range(90):
        if frame % 6 == 0:  # a wheel notch every 0.1 s
            spring.push(0.14)
        deltas.append(spring.advance(1 / 60, 0.05, **SETTLE))
    steady = deltas[30:84]
    mean = sum(steady) / len(steady)
    assert max(steady) / mean < 1.35  # far smoother than per-notch pulses


def test_zero_time_constant_jumps_immediately() -> None:
    spring = Spring()
    spring.push(5.0)
    assert spring.advance(1 / 60, 0.0, **SETTLE) == 5.0
    assert not spring.active
    spring.push(1.0)
    spring.stop()
    assert spring.advance(1 / 60, 0.05, **SETTLE) == 0.0
