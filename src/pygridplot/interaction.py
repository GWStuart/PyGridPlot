"""Turning pygame input events into pan and zoom operations on a view.

:class:`InteractionController` implements the plane's built-in controls:

========================  ==========================================
Input                     Action
========================  ==========================================
Left-drag                 Pan; the grabbed point stays under the cursor
Mouse wheel / trackpad    Zoom about the cursor
Shift / Ctrl + wheel      Stretch only x / only y (unequal aspect only)
Arrow keys                Pan by a fraction of the plane
``+`` / ``-``             Zoom about the middle of the plane
``Home`` / ``0``          Reset the view
========================  ==========================================

Dragging is *anchored*: the world point under the cursor when the button
went down is pinned to the cursor, so there is no drift however long the
drag lasts.

Wheel and keyboard movement is **eased** by a critically damped spring (see
:mod:`pygridplot.core.easing`). The view glides towards its target with
continuous velocity, so a notched mouse wheel feels as smooth as a trackpad.
Easing advances in :meth:`InteractionController.update`, which
:meth:`pygridplot.Plane.draw_grid` calls every frame. Set
``InteractionConfig(smoothing=0)`` for instant movement, for example in
programs that only redraw when an event arrives.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import pygame

from pygridplot.core.easing import Spring
from pygridplot.core.view import View, ZoomAxis

_MAX_PENDING_LOG_ZOOM: Final = math.log(1e4)
"""Cap on queued zoom, so spinning the wheel at a zoom limit cannot pile up."""

_MAX_FRAME_DT: Final = 0.1
"""Longest time step used for easing, so a stalled frame does not jump."""

_SETTLE_LOG_ZOOM: Final = 2e-4
"""Remaining zoom (log factor) below which easing snaps onto the target."""

_SETTLE_PAN_PX: Final = 0.05
"""Remaining pan in pixels below which easing snaps onto the target."""

_BLOCKING_MODIFIERS: Final = pygame.KMOD_CTRL | pygame.KMOD_ALT | pygame.KMOD_META
"""Modifier keys that stop the plane from reacting to key presses."""


@dataclass(frozen=True, slots=True)
class KeyBindings:
    """Which keys trigger each keyboard action.

    Each attribute is a tuple of pygame key codes. Pass an empty tuple to
    disable an action.

    Attributes:
        pan_left: Move the view left.
        pan_right: Move the view right.
        pan_up: Move the view up.
        pan_down: Move the view down.
        zoom_in: Zoom in about the middle of the plane.
        zoom_out: Zoom out about the middle of the plane.
        reset: Restore the initial view.
    """

    pan_left: tuple[int, ...] = (pygame.K_LEFT,)
    pan_right: tuple[int, ...] = (pygame.K_RIGHT,)
    pan_up: tuple[int, ...] = (pygame.K_UP,)
    pan_down: tuple[int, ...] = (pygame.K_DOWN,)
    zoom_in: tuple[int, ...] = (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS)
    zoom_out: tuple[int, ...] = (pygame.K_MINUS, pygame.K_KP_MINUS)
    reset: tuple[int, ...] = (pygame.K_HOME, pygame.K_0, pygame.K_KP0)


@dataclass(frozen=True, slots=True)
class InteractionConfig:
    """Settings for the built-in mouse and keyboard controls.

    Attributes:
        pan_button: Mouse button that drags the plane (1 is the left button).
        wheel_zoom: Whether the mouse wheel / trackpad zooms.
        zoom_per_notch: Zoom factor for one notch of a mouse wheel. Trackpads
            send fractional notches and zoom proportionally.
        invert_wheel: Swap the wheel direction (scrolling up zooms out).
        keyboard: Whether the keyboard controls are active.
        key_pan_fraction: How far one arrow-key press pans, as a fraction of
            the plane's width or height.
        key_zoom_factor: Zoom factor for one ``+``/``-`` key press.
        smoothing: Time constant in seconds of the easing spring. A single
            wheel notch is about 90% complete after ``3.9 * smoothing``
            (0.2 s with the default). ``0`` makes movement instant.
        bindings: Which keys trigger which actions.
    """

    pan_button: int = 1
    wheel_zoom: bool = True
    zoom_per_notch: float = 1.15
    invert_wheel: bool = False
    keyboard: bool = True
    key_pan_fraction: float = 0.1
    key_zoom_factor: float = 1.25
    smoothing: float = 0.05
    bindings: KeyBindings = field(default_factory=KeyBindings)

    def __post_init__(self) -> None:
        """Validate the numeric settings."""
        if not self.zoom_per_notch > 1.0 or not self.key_zoom_factor > 1.0:
            msg = "zoom_per_notch and key_zoom_factor must be greater than 1"
            raise ValueError(msg)
        if not 0.0 < self.key_pan_fraction <= 1.0:
            msg = "key_pan_fraction must be within (0, 1]"
            raise ValueError(msg)
        if self.smoothing < 0.0:
            msg = "smoothing must be non-negative"
            raise ValueError(msg)


class InteractionController:
    """Applies pan and zoom to a :class:`~pygridplot.core.view.View` from events.

    Args:
        view: The view to control. It is modified in place.
        config: Control settings.
        on_reset: Called when the reset key is pressed (the plane passes its
            :meth:`~pygridplot.Plane.reset_view`).
        clock: Returns the current time in seconds. Override for testing.
    """

    def __init__(
        self,
        view: View,
        config: InteractionConfig | None = None,
        *,
        on_reset: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._view = view
        self.config = config if config is not None else InteractionConfig()
        self._on_reset = on_reset
        self._clock = clock
        self._drag_anchor: tuple[float, float] | None = None
        self._mouse: tuple[int, int] | None = None
        self._zoom = Spring()  # in natural-log units of the scale
        self._zoom_about_cursor = True
        self._pan_x = Spring()  # in pixels of content movement
        self._pan_y = Spring()
        self._last_update: float | None = None

    @property
    def dragging(self) -> bool:
        """Whether a drag-pan is in progress."""
        return self._drag_anchor is not None

    @property
    def animating(self) -> bool:
        """Whether eased movement is still in progress."""
        return self._zoom.active or self._pan_x.active or self._pan_y.active

    def cancel(self) -> None:
        """Stop any drag and discard any eased movement still in progress."""
        self._drag_anchor = None
        self._zoom.stop()
        self._pan_x.stop()
        self._pan_y.stop()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Apply one pygame event to the view.

        Pass every event to this method (including mouse motion), because
        the controller tracks the cursor to know where to zoom.

        Args:
            event: The event to process.

        Returns:
            ``True`` if the event was used by the plane (so the caller can
            skip its own handling), ``False`` otherwise.
        """
        kind = event.type
        if kind == pygame.MOUSEMOTION:
            return self._on_motion(event)
        if kind == pygame.MOUSEBUTTONDOWN:
            return self._on_button_down(event)
        if kind == pygame.MOUSEBUTTONUP:
            return self._on_button_up(event)
        if kind == pygame.MOUSEWHEEL:
            return self._on_wheel(event)
        if kind == pygame.KEYDOWN:
            return self._on_key(event)
        if kind in (pygame.WINDOWFOCUSLOST, pygame.WINDOWLEAVE):
            self._drag_anchor = None
        return False

    def _on_motion(self, event: pygame.event.Event) -> bool:
        """Track the cursor and continue a drag.

        Args:
            event: A ``MOUSEMOTION`` event.

        Returns:
            ``True`` while dragging.
        """
        self._mouse = (int(event.pos[0]), int(event.pos[1]))
        if self._drag_anchor is None:
            return False
        self._view.pin(self._drag_anchor, self._mouse)
        return True

    def _on_button_down(self, event: pygame.event.Event) -> bool:
        """Start a drag when the pan button is pressed inside the plane.

        Args:
            event: A ``MOUSEBUTTONDOWN`` event.

        Returns:
            ``True`` if the press was inside the plane and used.
        """
        self._mouse = (int(event.pos[0]), int(event.pos[1]))
        if not self._view.rect.contains_point(*self._mouse):
            return False
        if event.button == self.config.pan_button:
            self._pan_x.stop()  # a drag takes over from keyboard panning
            self._pan_y.stop()
            self._drag_anchor = self._view.to_world(*map(float, self._mouse))
            return True
        # Buttons 4/5 are the legacy duplicates of wheel events; the wheel
        # itself is handled through MOUSEWHEEL.
        return event.button in (4, 5)

    def _on_button_up(self, event: pygame.event.Event) -> bool:
        """End a drag when the pan button is released.

        Args:
            event: A ``MOUSEBUTTONUP`` event.

        Returns:
            ``True`` if a drag ended.
        """
        self._mouse = (int(event.pos[0]), int(event.pos[1]))
        if event.button == self.config.pan_button and self._drag_anchor is not None:
            self._drag_anchor = None
            return True
        return False

    def _on_wheel(self, event: pygame.event.Event) -> bool:
        """Zoom about the cursor for wheel or trackpad scrolling.

        Args:
            event: A ``MOUSEWHEEL`` event.

        Returns:
            ``True`` if the cursor was over the plane and zooming is enabled.
        """
        if not self.config.wheel_zoom:
            return False
        mouse = self._mouse_position()
        if not self._view.rect.contains_point(*mouse):
            return False
        notches = float(getattr(event, "precise_y", event.y))
        if self.config.invert_wheel:
            notches = -notches
        if notches == 0.0:
            return True
        log_factor = notches * math.log(self.config.zoom_per_notch)
        axis = self._wheel_axis()
        if axis != "both":
            self._view.zoom_at(*mouse, math.exp(log_factor), axis=axis)
        elif self.config.smoothing > 0.0:
            self._queue_zoom(log_factor, about_cursor=True)
        else:
            self._view.zoom_at(*mouse, math.exp(log_factor))
        return True

    def _on_key(self, event: pygame.event.Event) -> bool:
        """Handle the keyboard shortcuts.

        Args:
            event: A ``KEYDOWN`` event.

        Returns:
            ``True`` if the key triggered an action.
        """
        if not self.config.keyboard or event.mod & _BLOCKING_MODIFIERS:
            return False
        keys = self.config.bindings
        key = event.key
        if key in keys.reset:
            self.cancel()
            if self._on_reset is not None:
                self._on_reset()
            return True
        if key in keys.zoom_in or key in keys.zoom_out:
            factor = self.config.key_zoom_factor
            self._zoom_center(math.log(factor if key in keys.zoom_in else 1 / factor))
            return True
        rect = self._view.rect
        step_x = self.config.key_pan_fraction * rect.width
        step_y = self.config.key_pan_fraction * rect.height
        pans = (
            (keys.pan_left, step_x, 0.0),
            (keys.pan_right, -step_x, 0.0),
            (keys.pan_up, 0.0, step_y),
            (keys.pan_down, 0.0, -step_y),
        )
        for bound, dx, dy in pans:
            if key in bound:
                self._pan(dx, dy)
                return True
        return False

    # ------------------------------------------------------------------
    # Easing
    # ------------------------------------------------------------------
    def update(self) -> bool:
        """Advance eased movement by the time since the previous call.

        Call this once per frame (:meth:`pygridplot.Plane.draw_grid` does it
        automatically).

        Returns:
            ``True`` if the view changed.
        """
        now = self._clock()
        last, self._last_update = self._last_update, now
        if not self.animating:
            return False
        dt = _MAX_FRAME_DT if last is None else min(max(now - last, 0.0), _MAX_FRAME_DT)
        self._advance_zoom(dt)
        self._advance_pan(dt)
        return True

    def _advance_zoom(self, dt: float) -> None:
        """Move the zoom spring forward and apply its change about the anchor.

        Args:
            dt: Elapsed time in seconds.
        """
        step = self._zoom.advance(
            dt,
            self.config.smoothing,
            settle_offset=_SETTLE_LOG_ZOOM,
            settle_velocity=_SETTLE_LOG_ZOOM * 10,
        )
        if step == 0.0:
            return
        applied = self._view.zoom_at(*self._zoom_anchor(), math.exp(step))
        if abs(math.log(applied) - step) > 1e-9:
            self._zoom.stop()  # hit a zoom limit: stop instead of pushing forever

    def _advance_pan(self, dt: float) -> None:
        """Move the pan springs forward and apply their change.

        Args:
            dt: Elapsed time in seconds.
        """
        settle = {
            "settle_offset": _SETTLE_PAN_PX,
            "settle_velocity": _SETTLE_PAN_PX * 10,
        }
        dx = self._pan_x.advance(dt, self.config.smoothing, **settle)
        dy = self._pan_y.advance(dt, self.config.smoothing, **settle)
        if dx or dy:
            self._view.pan_pixels(dx, dy)

    def _queue_zoom(self, log_factor: float, *, about_cursor: bool) -> None:
        """Add zoom for the spring to glide through.

        Args:
            log_factor: Natural log of the zoom factor to add.
            about_cursor: Whether the zoom should keep the point under the
                cursor fixed (wheel) rather than the middle of the plane (keys).
        """
        if not self.animating:
            self._last_update = self._clock()
        self._zoom_about_cursor = about_cursor
        self._zoom.push(log_factor)
        limit = _MAX_PENDING_LOG_ZOOM
        self._zoom.offset = min(max(self._zoom.offset, -limit), limit)

    def _zoom_center(self, log_factor: float) -> None:
        """Zoom about the middle of the plane, eased if enabled.

        Args:
            log_factor: Natural log of the zoom factor.
        """
        if self.config.smoothing > 0.0:
            self._queue_zoom(log_factor, about_cursor=False)
        else:
            self._view.zoom(math.exp(log_factor))

    def _pan(self, dx: float, dy: float) -> None:
        """Pan the content by ``(dx, dy)`` pixels, eased if enabled.

        Args:
            dx: Horizontal movement of the content in pixels.
            dy: Vertical movement of the content in pixels.
        """
        if self.dragging:
            return
        if self.config.smoothing <= 0.0:
            self._view.pan_pixels(dx, dy)
            return
        if not self.animating:
            self._last_update = self._clock()
        self._pan_x.push(dx)
        self._pan_y.push(dy)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _mouse_position(self) -> tuple[int, int]:
        """Return the latest known cursor position.

        Returns:
            The position from the most recent mouse event, or the current
            position reported by pygame if no mouse event has arrived yet.
        """
        if self._mouse is not None:
            return self._mouse
        x, y = pygame.mouse.get_pos()
        return (int(x), int(y))

    def _zoom_anchor(self) -> tuple[float, float]:
        """Return the screen point eased zooming should keep fixed.

        Returns:
            The cursor for wheel zooms while it is over the plane (and always
            while dragging, so the grabbed point stays under it), otherwise the
            middle of the plane.
        """
        mouse = self._mouse
        use_cursor = self._zoom_about_cursor or self.dragging
        if use_cursor and mouse is not None and self._view.rect.contains_point(*mouse):
            return (float(mouse[0]), float(mouse[1]))
        return self._view.origin

    def _wheel_axis(self) -> ZoomAxis:
        """Return which axes a wheel zoom should stretch, from held modifiers.

        Returns:
            ``"x"`` with Shift or ``"y"`` with Ctrl when the view allows unequal
            scales, otherwise ``"both"``.
        """
        if self._view.equal_aspect:
            return "both"
        mods = pygame.key.get_mods()
        if mods & pygame.KMOD_SHIFT:
            return "x"
        if mods & pygame.KMOD_CTRL:
            return "y"
        return "both"
