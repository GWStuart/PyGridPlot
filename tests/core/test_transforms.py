"""Tests for axis transforms (:mod:`pygridplot.core.transforms`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pygridplot.core.transforms import LINEAR, LOG, AxisTransform, transform_for


def test_linear_transform_is_the_identity() -> None:
    assert LINEAR.forward(-3.5) == -3.5
    assert LINEAR.inverse(2.0) == 2.0
    values = np.array([-1.0, 0.0, 7.25])
    assert np.array_equal(LINEAR.forward(values), values)
    assert not LINEAR.is_log
    assert LINEAR.default_center == 0.0


def test_log_transform_uses_decades() -> None:
    assert LOG.forward(1000.0) == pytest.approx(3.0)
    assert LOG.inverse(-2.0) == pytest.approx(0.01)
    assert LOG.is_log
    assert LOG.default_center == 1.0


def test_log_of_zero_is_minus_infinity_and_negatives_are_nan() -> None:
    assert LOG.forward(0.0) == -math.inf
    assert math.isnan(LOG.forward(-1.0))
    with np.errstate(all="raise"):  # no warnings leak out
        result = LOG.forward(np.array([0.0, -2.0, 10.0]))
    assert result[0] == -math.inf
    assert math.isnan(result[1])
    assert result[2] == pytest.approx(1.0)


def test_inverse_overflows_to_infinity_quietly() -> None:
    assert LOG.inverse(400.0) == math.inf
    with np.errstate(all="raise"):
        assert np.isinf(LOG.inverse(np.array([400.0])))[0]


def test_check_rejects_values_an_axis_cannot_show() -> None:
    assert LOG.check("x", 100.0) == pytest.approx(2.0)
    for bad in (0.0, -5.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="positive"):
            LOG.check("x", bad)
    with pytest.raises(ValueError, match="finite"):
        LINEAR.check("x", math.nan)


def test_transforms_are_validated_and_shared() -> None:
    with pytest.raises(ValueError, match="linear"):
        AxisTransform("ln")  # type: ignore[arg-type]
    assert transform_for(log=True) is LOG
    assert transform_for(log=False) is LINEAR


@given(st.floats(1e-300, 1e300))
def test_log_round_trip(value: float) -> None:
    assert LOG.inverse(LOG.forward(value)) == pytest.approx(value, rel=1e-12)


def test_derivatives_map_directions_to_transformed_coordinates() -> None:
    xs = np.array([0.5, 1.0, 10.0, 250.0])
    assert np.array_equal(LINEAR.derivative(xs), np.ones(4))
    np.testing.assert_allclose(LOG.derivative(xs), 1 / (xs * math.log(10)))
    # Away from zero, it is the slope of forward().
    h = 1e-6
    numeric = (LOG.forward(xs * (1 + h)) - LOG.forward(xs * (1 - h))) / (2 * h * xs)
    np.testing.assert_allclose(LOG.derivative(xs), numeric, rtol=1e-6)
    assert np.isnan(LOG.derivative(np.array([0.0, -3.0]))).all()
