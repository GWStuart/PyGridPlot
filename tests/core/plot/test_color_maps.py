"""Tests for colouring by value (``core.plot.color_maps`` and ``pgp.color_map``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import pygridplot as pgp
from pygridplot.core.plot.color_maps import (
    COLOR_MAPS,
    check_range,
    check_stops,
    map_values,
    value_range,
)

GRAY = check_stops("gray")


def test_values_map_linearly_between_the_ends() -> None:
    colors = map_values([0.0, 0.5, 1.0], GRAY, (0.0, 1.0))
    assert colors.tolist() == [
        [0, 0, 0, 255],
        [128, 128, 128, 255],
        [255, 255, 255, 255],
    ]
    beyond = map_values([-5.0, 7.0], GRAY, (0.0, 1.0))
    assert beyond.tolist() == [[0, 0, 0, 255], [255, 255, 255, 255]]  # clamped


def test_unknown_values_are_transparent() -> None:
    colors = map_values([math.nan, 1.0, -1.0], GRAY, (0.1, 10.0), log=True)
    assert colors[0].tolist() == [0, 0, 0, 0]
    assert colors[1].tolist() == [128, 128, 128, 255]  # the middle, in decades
    assert colors[2].tolist() == [0, 0, 0, 0]  # not positive on a log scale


def test_ranges_default_to_the_values_own() -> None:
    colors = map_values([2.0, 4.0, 3.0], GRAY)
    assert colors[:, 0].tolist() == [0, 255, 128]
    assert map_values([5.0, 5.0], GRAY)[:, 0].tolist() == [128, 128]  # all equal
    assert map_values([math.nan], GRAY)[0].tolist() == [0, 0, 0, 0]
    assert value_range(np.array([3.0, np.nan, -1.0])) == (-1.0, 3.0)
    assert value_range(np.array([3.0, -1.0]), log=True) == (3.0, 3.0)
    assert value_range(np.array([math.nan])) is None


def test_shapes_are_kept() -> None:
    grid = np.linspace(0, 1, 12).reshape(3, 4)
    assert map_values(grid, GRAY, (0, 1)).shape == (3, 4, 4)
    assert map_values(0.5, GRAY, (0, 1)).shape == (4,)


def test_built_in_maps_run_from_dark_to_light() -> None:
    for name in ("viridis", "plasma", "inferno", "magma"):
        low, high = map_values([0.0, 1.0], check_stops(name), (0, 1))
        assert sum(low[:3].tolist()) < sum(high[:3].tolist()), name
    middle = map_values([0.5], check_stops("coolwarm"), (0, 1))[0]
    assert middle[0] == middle[1] == middle[2]  # grey in the middle
    assert set(COLOR_MAPS) >= {"viridis", "coolwarm", "gray"}


def test_maps_and_ranges_are_checked() -> None:
    assert check_stops([(255, 0, 0), (0, 0, 255, 128)]).tolist() == [
        [255, 0, 0, 255],
        [0, 0, 255, 128],
    ]
    for bad in ("rainbow", [(1, 2, 3)], [(1, 2)], [(300, 0, 0), (0, 0, 0)], 5):
        with pytest.raises(ValueError, match="colour map"):
            check_stops(bad)
    assert check_range(None) is None
    assert check_range((1, 2)) == (1.0, 2.0)
    for bad_range, log in (
        ((2, 1), False),
        ((0, 1), True),
        ((1,), False),
        ("ab", False),
    ):
        with pytest.raises(ValueError, match="range"):
            check_range(bad_range, log=log)


def test_the_public_color_map_resolves_names() -> None:
    colors = pgp.color_map([0, 1], ["navy", "#ff0000"], (0, 1))
    assert colors.tolist() == [[0, 0, 128, 255], [255, 0, 0, 255]]
    assert pgp.color_map([1, 2, 3]).shape == (3, 4)  # viridis, own range
    with pytest.raises(ValueError, match="unknown colour"):
        pgp.color_map([1], ["notacolour", "red"])
    with pytest.raises(ValueError, match="numbers"):
        pgp.color_map(["a"])
    # Ready for points: one RGBA row per value.
    assert pgp.color_map(np.arange(5.0), "plasma", (0, 4)).dtype == np.uint8
