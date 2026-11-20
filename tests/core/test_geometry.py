"""Tests for point normalisation and clipping (:mod:`pygridplot.core.geometry`)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pygridplot.core.geometry import (
    as_points,
    clip_path,
    clip_paths,
    clip_polygon,
    clip_segments,
    polygon_area,
)

BOX = (0.0, 0.0, 100.0, 50.0)


def test_as_points_accepts_the_documented_formats() -> None:
    assert as_points((1, 2)).tolist() == [[1.0, 2.0]]
    assert as_points([(1, 2), (3, 4)]).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert as_points(np.zeros((5, 2))).shape == (5, 2)
    assert as_points([]).shape == (0, 2)


@pytest.mark.parametrize("bad", [[1, 2, 3], [[1, 2, 3]], "abc", [(1, "x")]])
def test_as_points_rejects_other_shapes(bad: object) -> None:
    with pytest.raises(ValueError, match=r"points|expected"):
        as_points(bad)  # type: ignore[arg-type]


def test_clip_segments_cases() -> None:
    a = np.array([[10.0, 10.0], [-50.0, 25.0], [200.0, 0.0], [50.0, -10.0]])
    b = np.array([[20.0, 20.0], [150.0, 25.0], [300.0, 50.0], [50.0, -5.0]])
    keep, t0, t1 = clip_segments(a, b, BOX)
    assert keep.tolist() == [True, True, False, False]
    assert (t0[0], t1[0]) == (0.0, 1.0)  # entirely inside: untouched
    assert t0[1] == pytest.approx(50 / 200)  # enters at x = 0
    assert t1[1] == pytest.approx(150 / 200)  # leaves at x = 100


def test_non_finite_segments_are_dropped() -> None:
    a = np.array([[np.nan, 1.0]])
    b = np.array([[5.0, 5.0]])
    keep, _, _ = clip_segments(a, b, BOX)
    assert not keep[0]


def test_clip_path_splits_where_the_path_leaves_the_box() -> None:
    path = np.array([[10.0, 10], [50, 10], [50, 80], [60, 80], [60, 10], [90, 10]])
    pieces = clip_path(path, BOX)
    assert len(pieces) == 2
    first, second = pieces
    np.testing.assert_allclose(first, [[10, 10], [50, 10], [50, 50]])
    np.testing.assert_allclose(second, [[60, 50], [60, 10], [90, 10]])


def test_clip_path_nan_breaks_and_single_points() -> None:
    path = np.array([[10.0, 10], [20, 10], [np.nan, np.nan], [30, 30], [40, 30]])
    assert len(clip_path(path, BOX)) == 2
    assert len(clip_path(np.array([[5.0, 5.0]]), BOX)) == 1
    assert clip_path(np.array([[500.0, 5.0]]), BOX) == []
    assert clip_path(np.zeros((0, 2)), BOX) == []


def test_closed_paths_include_the_closing_edge() -> None:
    triangle = np.array([[10.0, 10], [40, 10], [25, 40]])
    (piece,) = clip_path(triangle, BOX, closed=True)
    assert piece.shape == (4, 2)
    np.testing.assert_allclose(piece[0], piece[-1])


def test_clip_paths_handles_several_paths() -> None:
    paths = [np.array([[10.0, 10], [20, 10]]), np.array([[30.0, 30], [40, 30]])]
    assert len(clip_paths(paths, BOX)) == 2
    assert clip_paths([], BOX) == []


@settings(max_examples=150, deadline=None)
@given(
    st.lists(
        st.tuples(st.floats(-1e4, 1e4), st.floats(-1e4, 1e4)), min_size=2, max_size=12
    )
)
def test_clipped_paths_stay_inside_the_box(points: list[tuple[float, float]]) -> None:
    for piece in clip_path(np.array(points), BOX):
        assert np.all(piece[:, 0] >= BOX[0] - 1e-6)
        assert np.all(piece[:, 0] <= BOX[2] + 1e-6)
        assert np.all(piece[:, 1] >= BOX[1] - 1e-6)
        assert np.all(piece[:, 1] <= BOX[3] + 1e-6)


def test_clip_polygon_inside_outside_and_partial() -> None:
    inside = np.array([[10.0, 10], [30, 10], [30, 30], [10, 30]])
    np.testing.assert_allclose(abs(polygon_area(clip_polygon(inside, BOX))), 400)
    outside = inside + 1000
    assert clip_polygon(outside, BOX).shape == (0, 2)
    straddling = np.array([[-50.0, 10], [50, 10], [50, 100], [-50, 100]])
    clipped = clip_polygon(straddling, BOX)
    assert abs(polygon_area(clipped)) == pytest.approx(50 * 40)


def test_clip_polygon_handles_concave_shapes() -> None:
    # A "U" shape whose arms poke out of the top of the box.
    u_shape = np.array(
        [
            [10.0, 10],
            [90, 10],
            [90, 80],
            [70, 80],
            [70, 30],
            [30, 30],
            [30, 80],
            [10, 80],
        ]
    )
    clipped = clip_polygon(u_shape, BOX)
    expected = 80 * 20 + 2 * 20 * 20  # the base plus the two arms up to y = 50
    assert abs(polygon_area(clipped)) == pytest.approx(expected)


@settings(max_examples=150, deadline=None)
@given(
    st.lists(
        st.tuples(st.floats(-300, 400), st.floats(-300, 400)), min_size=3, max_size=10
    )
)
def test_clipped_polygons_stay_inside(points: list[tuple[float, float]]) -> None:
    clipped = clip_polygon(np.array(points), BOX)
    if clipped.shape[0]:
        assert np.all(clipped[:, 0] >= BOX[0] - 1e-6)
        assert np.all(clipped[:, 0] <= BOX[2] + 1e-6)
        assert np.all(clipped[:, 1] >= BOX[1] - 1e-6)
        assert np.all(clipped[:, 1] <= BOX[3] + 1e-6)


def test_polygon_area_sign_and_degenerate_rings() -> None:
    square = np.array([[0.0, 0], [1, 0], [1, 1], [0, 1]])
    assert polygon_area(square) == 1.0
    assert polygon_area(square[::-1]) == -1.0
    assert polygon_area(square[:2]) == 0.0
