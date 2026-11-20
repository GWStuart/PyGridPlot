"""Tests for robust evaluation and adaptive sampling (:mod:`core.sampling`).

The function test set from PROJECT_PLAN section 9 (``1/x``, ``tan``,
``sqrt``, ``log``, ``floor``, ``x**2``) is checked for the properties that
make each one look right.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pygridplot.core.evaluation import evaluate
from pygridplot.core.sampling import SamplingConfig, sample_function

SCALE = (60.0, 60.0)
X_RANGE = (-7.6, 7.6)
BAND = (-5.1, 5.1)


def sample(
    func: object,
    x_range: tuple[float, float] = X_RANGE,
    config: SamplingConfig | None = None,
) -> list[np.ndarray]:
    """Sample with the default test view (60 px per unit, about 900x600 px)."""
    return sample_function(
        func,  # type: ignore[arg-type]
        x_range,
        SCALE,
        BAND,
        config or SamplingConfig(),
    )


# ----------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------
def test_vectorised_functions_are_called_once() -> None:
    calls = []

    def f(x: np.ndarray) -> np.ndarray:
        calls.append(np.shape(x))
        return x * 2

    np.testing.assert_array_equal(evaluate(f, np.array([1.0, 2.0])), [2.0, 4.0])
    assert calls == [(2,)]


def test_scalar_only_functions_fall_back_to_a_loop() -> None:
    values = evaluate(math.sin, np.array([0.0, math.pi / 2]))
    np.testing.assert_allclose(values, [0.0, 1.0])


def test_errors_become_nan_and_never_escape() -> None:
    values = evaluate(math.sqrt, np.array([-1.0, 4.0]))  # ValueError for -1
    assert math.isnan(values[0])
    assert values[1] == 2.0

    def always_fails(x: object) -> float:
        raise RuntimeError("boom")

    assert np.isnan(evaluate(always_fails, np.array([1.0, 2.0]))).all()


def test_complex_results_keep_only_real_values() -> None:
    values = evaluate(lambda x: np.sqrt(x + 0j), np.array([-4.0, 9.0]))
    assert math.isnan(values[0])
    assert values[1] == 3.0
    scalar = evaluate(lambda x: complex(x, 0.0), np.array([1.5]))
    assert scalar[0] == 1.5


def test_awkward_return_types() -> None:
    xs = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(evaluate(lambda x: 5, xs), [5, 5, 5])  # constant
    np.testing.assert_array_equal(evaluate(lambda x: x > 1.5, xs), [0, 1, 1])
    np.testing.assert_array_equal(evaluate(lambda x: x.astype(int), xs), [1, 2, 3])
    masked = evaluate(lambda x: np.ma.masked_greater(x, 2.5), xs)
    assert math.isnan(masked[2])
    assert masked[0] == 1.0
    nones = evaluate(lambda x: None, xs)
    assert np.isnan(nones).all()


def test_the_input_array_is_protected_from_mutation() -> None:
    xs = np.array([1.0, 2.0])

    def mutating(x: np.ndarray) -> np.ndarray:
        x *= 100
        return x

    evaluate(mutating, xs)
    np.testing.assert_array_equal(xs, [1.0, 2.0])


def test_empty_input() -> None:
    assert evaluate(np.sin, np.zeros(0)).shape == (0,)


# ----------------------------------------------------------------------
# sample_function: the shared test set
# ----------------------------------------------------------------------
def test_parabola_is_one_accurate_polyline() -> None:
    (line,) = sample(lambda x: x**2)
    assert line[0, 0] == pytest.approx(X_RANGE[0])
    assert line[-1, 0] == pytest.approx(X_RANGE[1])
    np.testing.assert_allclose(line[:, 1], line[:, 0] ** 2)
    # The polyline never strays more than the tolerance from the curve.
    mids = 0.5 * (line[1:, 0] + line[:-1, 0])
    chord = 0.5 * (line[1:, 1] + line[:-1, 1])
    assert np.max(np.abs(chord - mids**2)) * SCALE[1] <= 0.3


def test_reciprocal_is_broken_at_its_asymptote() -> None:
    lines = sample(lambda x: 1 / x)
    assert len(lines) == 2
    left, right = sorted(lines, key=lambda line: line[0, 0])
    assert left[-1, 0] < 0 < right[0, 0]
    assert left[-1, 1] < BAND[0]  # runs off the bottom...
    assert right[0, 1] > BAND[1]  # ...and comes back from the top


def test_tangent_is_broken_at_every_asymptote() -> None:
    lines = sample(np.tan)
    asymptotes = [k * math.pi + math.pi / 2 for k in range(-3, 3)]
    visible = [a for a in asymptotes if X_RANGE[0] < a < X_RANGE[1]]
    assert len(lines) == len(visible) + 1
    for line in lines:  # each branch increases monotonically: no joins
        assert np.all(np.diff(line[:, 1]) > 0)


def test_floor_has_no_vertical_risers() -> None:
    lines = sample(np.floor)
    assert len(lines) == 11  # the steps from -5 to 5; the others are off-screen
    for line in lines:
        assert np.ptp(line[:, 1]) == 0.0  # every piece is a flat step


def test_sqrt_reaches_its_domain_edge() -> None:
    (line,) = sample(np.sqrt)
    assert line[0, 0] >= 0.0
    assert line[0, 0] * SCALE[0] < 0.01  # starts within 1/100 px of x = 0
    assert line[0, 1] * SCALE[1] < 1.0


def test_log_runs_off_the_bottom_near_zero() -> None:
    (line,) = sample(np.log)
    assert line[0, 0] > 0.0
    assert line[0, 1] < BAND[0]


def test_removable_singularity_is_bridged() -> None:
    def sinc(x: np.ndarray) -> np.ndarray:
        return np.sin(x) / x  # NaN exactly at x = 0 (sampled!)

    lines = sample(sinc, (-5.0, 5.0))
    assert len(lines) == 2  # split only at the single undefined point
    left, right = sorted(lines, key=lambda line: line[0, 0])
    gap_px = (right[0, 0] - left[-1, 0]) * SCALE[0]
    assert gap_px < 0.05  # far too small to see
    assert left[-1, 1] == pytest.approx(1.0, abs=1e-3)


def test_steep_but_continuous_functions_are_not_broken() -> None:
    (line,) = sample(lambda x: np.tanh(20 * x))
    # Accurate everywhere: check the curve at the middle of every segment.
    mids = 0.5 * (line[1:, 0] + line[:-1, 0])
    chord = 0.5 * (line[1:, 1] + line[:-1, 1])
    assert np.max(np.abs(chord - np.tanh(20 * mids))) * SCALE[1] <= 0.3


def test_functions_without_visible_values_give_nothing() -> None:
    assert sample(lambda x: np.full_like(x, np.nan)) == []
    assert sample(lambda x: x + 100) == []  # entirely above the screen
    assert sample(np.sin, (3.0, 3.0)) == []


def test_huge_values_do_not_break_sampling() -> None:
    lines = sample(lambda x: np.where(x > 0, 1e300, 0.5))
    assert lines
    assert all(np.isfinite(line).all() for line in lines)


def test_the_evaluation_budget_is_respected() -> None:
    count = 0

    def counted(x: np.ndarray) -> np.ndarray:
        nonlocal count
        count += np.size(x)
        return np.sin(1 / x)  # infinitely wiggly near 0

    config = SamplingConfig(max_samples=3000)
    sample(counted, config=config)
    assert count <= 3000


def test_without_refinement_only_huge_rises_are_broken() -> None:
    no_rounds = SamplingConfig(max_rounds=0)
    assert len(sample(np.floor, config=no_rounds)) == 1  # small jumps stay joined
    assert len(sample(np.tan, config=no_rounds)) == 5  # asymptotes still break


@settings(max_examples=60, deadline=None)
@given(shift=st.integers(-300, 300))
def test_samples_are_anchored_to_the_world(shift: int) -> None:
    """Panning by whole pixels reuses exactly the same world samples."""
    step = 1 / SCALE[0]
    moved = (X_RANGE[0] + shift * step, X_RANGE[1] + shift * step)
    (base,) = sample(np.cos)
    (other,) = sample(np.cos, moved)
    # Away from both ranges' ends (where the end segments differ), every
    # sample position of one is also a sample position of the other.
    low = max(X_RANGE[0], moved[0]) + 0.5
    high = min(X_RANGE[1], moved[1]) - 0.5
    common = base[(base[:, 0] > low) & (base[:, 0] < high), 0]
    np.testing.assert_array_equal(np.intersect1d(common, other[:, 0]), common)


def test_branches_reach_the_edge_even_at_very_narrow_poles() -> None:
    """The gamma function's poles get extremely narrow for large negative x."""
    lines = sample(math.gamma, (-9.2, 9.2))
    near_poles = [line for line in lines if line[-1, 0] < -4.5]
    assert len(near_poles) == 5  # the branches between -9.2 and -5
    for line in near_poles:
        assert np.max(np.abs(line[:, 1])) > BAND[1]  # every branch leaves the screen


def test_domain_edges_are_found_exactly() -> None:
    (line,) = sample(np.sqrt)
    assert line[0, 0] * SCALE[0] < 1e-5  # within a hundred-thousandth of a pixel
    (arc,) = sample(lambda x: np.sqrt(1 - x**2), (-1.5, 1.5))
    assert arc[0, 0] == pytest.approx(-1.0, abs=1e-6)
    assert arc[-1, 0] == pytest.approx(1.0, abs=1e-6)


def test_steps_are_not_mistaken_for_poles() -> None:
    left, right = sorted(sample(np.sign), key=lambda line: line[0, 0])
    assert set(np.unique(left[:, 1])) == {-1.0}
    assert set(np.unique(right[:, 1])) <= {0.0, 1.0}


def test_chasing_respects_the_budget() -> None:
    count = 0

    def counted(x: object) -> float:
        nonlocal count
        if isinstance(x, float):  # array attempts fail at once and do no work
            count += 1
        return math.gamma(x)  # type: ignore[arg-type]

    sample(counted, (-9.2, 9.2), SamplingConfig(max_samples=1500))
    assert count <= 1500
