"""Tests for evaluating vector fields robustly (:mod:`core.plot.evaluation`).

Functions of one and two variables are tested with the samplers that use
them (``test_sampling.py``, ``test_implicit.py``); these tests cover
:func:`evaluate_vectors`, for functions returning a pair ``(u, v)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pygridplot.core.plot.evaluation import evaluate_vectors

XS = np.array([1.0, 2.0, 3.0])
YS = np.array([-1.0, 0.0, 4.0])


@pytest.mark.parametrize(
    "field",
    [
        lambda x, y: (-y, x),  # a tuple of arrays
        lambda x, y: [-y, x],  # a list
        lambda x, y: np.array([-y, x]),  # an array of shape (2, N)
        lambda x, y: (-y, math.cos(x) * 0 + x),  # scalar-only: point by point
    ],
)
def test_fields_return_one_row_per_point(field: object) -> None:
    vectors = evaluate_vectors(field, XS, YS)  # type: ignore[arg-type]
    assert vectors.shape == (3, 2)
    np.testing.assert_array_equal(vectors, np.column_stack((-YS, XS)))


def test_constant_components_are_spread_over_the_points() -> None:
    vectors = evaluate_vectors(lambda x, y: (1, y), XS, YS)
    np.testing.assert_array_equal(vectors, np.column_stack((np.ones(3), YS)))
    both = evaluate_vectors(lambda x, y: (2.0, -1.0), XS, YS)  # a constant field
    np.testing.assert_array_equal(both, np.tile([2.0, -1.0], (3, 1)))


def test_undefined_vectors_are_nan() -> None:
    def partly(x: float, y: float) -> tuple[float, float]:
        if x == 2.0:
            raise ZeroDivisionError
        u = math.sqrt(y) if y >= 0 else math.nan  # scalar-only
        return (u, x)

    vectors = evaluate_vectors(partly, XS, YS)
    assert np.isnan(vectors[0, 0])
    assert vectors[0, 1] == 1.0
    assert np.isnan(vectors[1]).all()
    np.testing.assert_array_equal(vectors[2], [2.0, 3.0])


@pytest.mark.parametrize(
    "result",
    [
        "not a pair",
        (1.0, 2.0, 3.0),  # three components
        None,
        1j,
    ],
)
def test_results_that_are_not_pairs_are_nan(result: object) -> None:
    vectors = evaluate_vectors(lambda x, y: result, XS, YS)
    assert np.isnan(vectors).all()


def test_complex_components_keep_only_real_values() -> None:
    vectors = evaluate_vectors(lambda x, y: (np.sqrt(y + 0j), x + 0j), XS, YS)
    assert np.isnan(vectors[0, 0])  # sqrt(-1) is not real
    np.testing.assert_array_equal(vectors[1:, 0], [0.0, 2.0])
    np.testing.assert_array_equal(vectors[:, 1], XS)


def test_no_points_give_no_vectors() -> None:
    assert evaluate_vectors(lambda x, y: (x, y), XS[:0], YS[:0]).shape == (0, 2)
