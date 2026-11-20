"""Tests for the built-in pan/zoom controls, driven by synthetic events."""

from __future__ import annotations

import math
from collections.abc import Callable

import pygame
import pytest

from pygridplot.core.coords import ScreenRect
from pygridplot.core.view import View, ViewLimits
from pygridplot.interaction import InteractionConfig, InteractionController

pytestmark = pytest.mark.usefixtures("pygame_display")

RECT = ScreenRect(100, 50, 400, 300)
INSTANT = InteractionConfig(smoothing=0.0)


class FakeClock:
    """A controllable replacement for ``time.perf_counter``."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make(
    config: InteractionConfig = INSTANT,
    *,
    on_reset: Callable[[], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> InteractionController:
    """Return a controller for a fresh 400x300 view inside a larger window."""
    return InteractionController(
        View(RECT, scale=50.0),
        config,
        on_reset=on_reset,
        clock=clock if clock is not None else FakeClock(),
    )


def mouse(kind: int, pos: tuple[int, int], button: int = 1) -> pygame.event.Event:
    """Build a mouse button or motion event."""
    if kind == pygame.MOUSEMOTION:
        return pygame.event.Event(kind, pos=pos, rel=(0, 0), buttons=(1, 0, 0))
    return pygame.event.Event(kind, pos=pos, button=button)


def wheel(y: float) -> pygame.event.Event:
    """Build a wheel event scrolling ``y`` notches."""
    return pygame.event.Event(
        pygame.MOUSEWHEEL, x=0, y=int(y), precise_x=0.0, precise_y=y, flipped=False
    )


def key(code: int, mod: int = 0) -> pygame.event.Event:
    """Build a key-down event."""
    return pygame.event.Event(pygame.KEYDOWN, key=code, mod=mod, unicode="", scancode=0)


def test_drag_keeps_the_grabbed_point_under_the_cursor() -> None:
    controller = make()
    view = controller._view
    grabbed = view.to_world(200.0, 100.0)
    assert controller.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (200, 100)))
    assert controller.dragging
    for pos in [(230, 90), (260, 140), (311, 177)]:
        assert controller.handle_event(mouse(pygame.MOUSEMOTION, pos))
        assert view.to_world(*map(float, pos)) == pytest.approx(grabbed)
    assert controller.handle_event(mouse(pygame.MOUSEBUTTONUP, (311, 177)))
    assert not controller.dragging
    assert not controller.handle_event(mouse(pygame.MOUSEMOTION, (320, 180)))


def test_drag_only_starts_inside_the_plane() -> None:
    controller = make()
    assert not controller.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (10, 10)))
    assert not controller.dragging


def test_wheel_zooms_about_the_cursor() -> None:
    controller = make()
    view = controller._view
    controller.handle_event(mouse(pygame.MOUSEMOTION, (420, 120)))
    anchor = view.to_world(420.0, 120.0)
    assert controller.handle_event(wheel(2.0))
    assert view.scale == pytest.approx(50.0 * INSTANT.zoom_per_notch**2)
    assert view.to_world(420.0, 120.0) == pytest.approx(anchor)
    controller.handle_event(wheel(-0.5))  # trackpads send fractional notches
    assert view.scale == pytest.approx(50.0 * INSTANT.zoom_per_notch**1.5)


def test_wheel_outside_the_plane_is_ignored() -> None:
    controller = make()
    controller.handle_event(mouse(pygame.MOUSEMOTION, (5, 5)))
    assert not controller.handle_event(wheel(1.0))
    assert controller._view.scale == 50.0


def test_eased_zoom_glides_to_the_target() -> None:
    clock = FakeClock()
    controller = make(InteractionConfig(smoothing=0.05), clock=clock)
    view = controller._view
    controller.handle_event(mouse(pygame.MOUSEMOTION, (300, 200)))
    anchor = view.to_world(300.0, 200.0)
    controller.handle_event(wheel(3.0))
    assert view.scale == 50.0  # nothing happens until the next frame
    target = 50.0 * 1.15**3
    previous = view.scale
    for _ in range(60):
        clock.now += 1 / 60
        controller.update()
        assert previous <= view.scale <= target * (1 + 1e-12)
        previous = view.scale
    assert view.scale == pytest.approx(target)
    assert not controller.animating
    assert view.to_world(300.0, 200.0) == pytest.approx(anchor)


def test_eased_zoom_stops_at_the_limits_instead_of_piling_up() -> None:
    clock = FakeClock()
    view = View(RECT, scale=50.0, limits=ViewLimits(min_scale=10.0, max_scale=100.0))
    controller = InteractionController(
        view, InteractionConfig(smoothing=0.05), clock=clock
    )
    controller.handle_event(mouse(pygame.MOUSEMOTION, (300, 200)))
    for _ in range(50):  # far more zoom-out than the limits allow
        controller.handle_event(wheel(-10.0))
    for _ in range(30):
        clock.now += 1 / 60
        controller.update()
    assert view.scale == 10.0
    assert not controller.animating  # the excess was discarded at the limit
    controller.handle_event(wheel(1.0))  # zooming back in works immediately
    clock.now += 1 / 60
    controller.update()
    assert view.scale > 10.0


def test_arrow_keys_pan_by_a_fraction_of_the_plane() -> None:
    controller = make()
    view = controller._view
    assert controller.handle_event(key(pygame.K_RIGHT))
    assert view.center_x == pytest.approx(0.1 * 400 / 50)
    assert controller.handle_event(key(pygame.K_UP))
    assert view.center_y == pytest.approx(0.1 * 300 / 50)


def test_plus_and_minus_zoom_about_the_middle() -> None:
    controller = make()
    view = controller._view
    controller.handle_event(mouse(pygame.MOUSEMOTION, (450, 300)))  # off-centre cursor
    centre = view.center
    assert controller.handle_event(key(pygame.K_EQUALS))
    assert view.scale == pytest.approx(50.0 * 1.25)
    assert view.center == pytest.approx(centre)
    assert controller.handle_event(key(pygame.K_MINUS))
    assert view.scale == pytest.approx(50.0)


def test_reset_key_calls_back_and_modifiers_block_keys() -> None:
    calls: list[str] = []
    controller = make(on_reset=lambda: calls.append("reset"))
    assert controller.handle_event(key(pygame.K_HOME))
    assert calls == ["reset"]
    assert not controller.handle_event(key(pygame.K_0, pygame.KMOD_LCTRL))
    assert calls == ["reset"]


def test_keyboard_can_be_disabled() -> None:
    controller = make(InteractionConfig(smoothing=0.0, keyboard=False))
    assert not controller.handle_event(key(pygame.K_LEFT))


def test_eased_keyboard_pan_reaches_the_same_place() -> None:
    clock = FakeClock()
    controller = make(InteractionConfig(smoothing=0.05), clock=clock)
    controller.handle_event(key(pygame.K_LEFT))
    for _ in range(60):
        clock.now += 1 / 60
        controller.update()
    assert controller._view.center_x == pytest.approx(-0.1 * 400 / 50)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="greater than 1"):
        InteractionConfig(zoom_per_notch=1.0)
    with pytest.raises(ValueError, match="smoothing"):
        InteractionConfig(smoothing=-1.0)
    assert math.isclose(InteractionConfig().zoom_per_notch, 1.15)


def test_dragging_during_an_eased_zoom_keeps_the_grabbed_point() -> None:
    clock = FakeClock()
    controller = make(InteractionConfig(smoothing=0.05), clock=clock)
    view = controller._view
    controller.handle_event(mouse(pygame.MOUSEMOTION, (300, 200)))
    controller.handle_event(wheel(4.0))  # start a long eased zoom
    controller.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (300, 200)))
    grabbed = controller._drag_anchor
    assert grabbed is not None
    pos = (300, 200)
    for frame in range(40):
        pos = (300 + 3 * frame, 200 + frame)
        controller.handle_event(mouse(pygame.MOUSEMOTION, pos))
        clock.now += 1 / 60
        controller.update()
        assert view.to_world(*map(float, pos)) == pytest.approx(grabbed, rel=1e-9)
    assert view.scale == pytest.approx(50.0 * 1.15**4)
