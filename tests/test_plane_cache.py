"""Reusing plots between frames (:mod:`pygridplot.core.plot.memo`).

The guarantee is that the cache only changes the speed: a plane with the
cache draws exactly what a plane without it draws, through any sequence of
frames. The property test drives both planes through random sequences of
changes and compares every frame pixel for pixel.
"""

from __future__ import annotations

import math

import numpy as np
import pygame
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

import pygridplot as pgp
from pygridplot.core.plot.memo import PlotCache, Recorder, same_inputs

pytestmark = pytest.mark.usefixtures("pygame_display")

SIZE = (240, 180)


class Scene:
    """A frame's worth of plots, controlled by a few parameters."""

    def __init__(self) -> None:
        self.shift = 0.0
        self.spike = 0.5
        self.data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
        self.show_extra = True
        self.color = (200, 40, 40)

    def draw(self, plane: pgp.Plane, surface: pygame.Surface) -> None:
        """Draw every kind of plot, some depending on the parameters."""
        plane.draw_grid(surface)
        shift = self.shift
        plane.plot(surface, lambda x: np.sin(x - shift))  # a new lambda each frame
        if self.show_extra:  # a plot that comes and goes
            plane.plot(surface, np.cos, color=self.color)
        spike = self.spike
        # A narrow spike, between the first samples: moving it must be seen.
        plane.plot(surface, lambda x: 2 * np.exp(-(((x - spike) / 0.002) ** 2)))
        plane.plot(surface, pgp.derivative(lambda x: x**3 / 9 - shift))
        plane.parametric(surface, np.cos, lambda t: np.sin(t) + shift, (0, 2 * math.pi))
        plane.implicit(surface, lambda x, y: x * y - shift, 1)
        plane.polyline(surface, self.data)  # the same array, changed in place
        plane.polygon(surface, self.data - np.array([0.0, 2.0]))
        plane.points(surface, self.data)


def pixels(surface: pygame.Surface) -> NDArray[np.uint8]:
    """Return a copy of a surface's pixels."""
    return pygame.surfarray.array3d(surface).copy()


def planes() -> tuple[pgp.Plane, pgp.Plane, pygame.Surface, pygame.Surface]:
    """Two identical planes, with and without the cache, and their surfaces."""
    config = pgp.InteractionConfig(smoothing=0)
    cached = pgp.Plane(pygame.Rect(0, 0, *SIZE), scale=40.0, interaction=config)
    fresh = pgp.Plane(
        pygame.Rect(0, 0, *SIZE), scale=40.0, interaction=config, cache=False
    )
    return cached, fresh, pygame.Surface(SIZE), pygame.Surface(SIZE)


CHANGES = st.sampled_from(
    ["nothing", "nothing", "shift", "spike", "data", "toggle", "colour", "pan", "zoom"]
)


@settings(max_examples=25, deadline=None)
@given(changes=st.lists(CHANGES, min_size=2, max_size=8))
def test_the_cache_never_changes_what_is_drawn(changes: list[str]) -> None:
    cached, fresh, a, b = planes()
    scene = Scene()
    for change in ["nothing", *changes]:
        if change == "shift":
            scene.shift += 0.37
        elif change == "spike":
            scene.spike += 0.003  # less than a pixel: only the recorded points see it
        elif change == "data":
            scene.data[1, 1] += 0.25  # in place
        elif change == "toggle":
            scene.show_extra = not scene.show_extra
        elif change == "colour":
            scene.color = (40, 40, 200) if scene.color[0] > 100 else (200, 40, 40)
        elif change in ("pan", "zoom"):
            for plane in (cached, fresh):
                if change == "pan":
                    plane.view.pan_pixels(7, -3)
                else:
                    plane.view.zoom(1.1)
        scene.draw(cached, a)
        scene.draw(fresh, b)
        assert np.array_equal(pixels(a), pixels(b)), change
    assert cached.cache_stats.hits > 0 or all(c != "nothing" for c in changes)


def test_unchanged_frames_are_reused() -> None:
    cached, _, surface, _ = planes()
    scene = Scene()
    scene.draw(cached, surface)
    assert cached.cache_stats == pgp.CacheStats(hits=0, misses=9)
    scene.draw(cached, surface)
    assert cached.cache_stats == pgp.CacheStats(hits=9, misses=9)
    # Three plots change with the shift. The derivative of x³/9 - shift does
    # not: its values are the same, so it is reused, which is exactly right.
    scene.shift += 1.0
    scene.draw(cached, surface)
    assert cached.cache_stats == pgp.CacheStats(hits=15, misses=12)


def test_changes_of_style_and_inputs_are_seen() -> None:
    cached, _, surface, _ = planes()
    for kwargs in ({}, {}, {"width": 4.0}, {"color": "black"}, {"opacity": 0.5}):
        cached.draw_grid(surface)
        cached.plot(surface, np.sin, **kwargs)
    assert cached.cache_stats == pgp.CacheStats(hits=1, misses=4)


def test_turning_the_cache_off_forgets_everything() -> None:
    cached, _, surface, _ = planes()
    cached.draw_grid(surface)
    cached.plot(surface, np.sin)
    cached.cache = False
    assert not cached.cache
    assert cached.cache_stats == pgp.CacheStats()
    cached.draw_grid(surface)
    cached.plot(surface, np.sin)
    assert cached.cache_stats == pgp.CacheStats()
    cached.cache = True
    cached.clear_cache()
    assert cached.cache_stats == pgp.CacheStats()


def test_the_grid_is_rebuilt_when_the_theme_or_view_changes() -> None:
    cached, fresh, a, b = planes()
    for step in range(4):
        if step == 2:
            cached.theme = fresh.theme = pgp.DARK
        if step == 3:
            cached.view.pan_pixels(10, 0)
            fresh.view.pan_pixels(10, 0)
        cached.draw_grid(a)
        fresh.draw_grid(b)
        assert np.array_equal(pixels(a), pixels(b))


def test_embedded_planes_rasterise_again_when_the_clip_changes() -> None:
    surface = pygame.Surface(SIZE)
    plane = pgp.Plane(pygame.Rect(0, 0, *SIZE), scale=40.0)
    plane.draw_grid(surface)
    plane.plot(surface, np.sin, color=(0, 0, 0))
    full = pixels(surface)
    plane.rect = pygame.Rect(0, 0, SIZE[0] // 2, SIZE[1])  # half the width
    plane.draw_grid(surface)
    plane.plot(surface, np.sin, color=(0, 0, 0))
    assert not np.array_equal(pixels(surface), full)


# --- The pure parts -------------------------------------------------------


def test_recorders_check_every_recorded_value() -> None:
    k = [2.0]
    recorder = Recorder(lambda x: k[0] * x)
    recorder(np.array([1.0, 2.0]))
    recorder(np.array([3.0]))
    assert recorder.evaluations == 3
    assert recorder.unchanged()
    k[0] = 2.000001
    assert not recorder.unchanged()


def test_recorders_handle_undefined_values_and_signed_zeros() -> None:
    recorder = Recorder(lambda x: np.sqrt(x))
    with np.errstate(invalid="ignore"):  # as the samplers evaluate
        recorder(np.array([-1.0, 0.0, 4.0]))
    assert recorder.unchanged()  # NaN equals NaN
    zero = Recorder(lambda x: 0.0 * x)
    zero(np.array([-1.0, 1.0]))  # -0.0 and 0.0
    zero.func = lambda x: 0.0 * np.abs(x)
    assert not zero.unchanged()  # the sign of zero counts


def test_two_variable_recorders() -> None:
    recorder = Recorder(lambda x, y: x * y, arity=2)
    recorder(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert recorder.unchanged()
    recorder.func = lambda x, y: x + y
    assert not recorder.unchanged()


def test_keys_compare_arrays_and_nested_values() -> None:
    a = np.array([[0.0, np.nan]])
    assert same_inputs((1, "x", a, (2.0, None)), (1, "x", a.copy(), (2.0, None)))
    assert not same_inputs((a,), (a + 1,))
    assert not same_inputs((a,), ([[0.0, np.nan]],))
    assert not same_inputs((1,), (1.0,))  # a different type may draw differently
    assert same_inputs(math.nan, math.nan)


def test_calls_are_matched_by_position_and_kind() -> None:
    cache = PlotCache()
    built: list[str] = []

    def make(name: str) -> object:
        def build(functions: list[object]) -> str:
            built.append(name)
            return name

        return build

    for _ in range(2):
        cache.begin_frame()
        assert cache.build("plot", 1, [np.sin], make("a")) == "a"
        assert cache.build("implicit", 1, [np.add], make("b"), arity=2) == "b"
    assert built == ["a", "b"]  # the second frame reused both
    cache.begin_frame()
    cache.build("polar", 1, [np.sin], make("c"))  # a different kind in the first slot
    assert built == ["a", "b", "c"]
    cache.begin_frame()
    cache.build("polar", 1, [3.0], make("d"))  # a constant in place of the function
    assert built[-1] == "d"
