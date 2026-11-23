"""Turning pygame mouse events into drags of draggable points and sliders.

:class:`DragController` is the pygame side of
:class:`~pygridplot.core.handles.point.DraggablePoint`, and
:class:`SliderController` that of :class:`~pygridplot.core.handles.slider.Slider`.
The plane passes every event to the sliders first (their panels are drawn
on top), then to the points, then to its own pan and zoom controls.

For points:

* **Pressing** the drag button over a point grabs it (the event is used, so
  the plane does not start panning). The nearest point to the cursor wins,
  and among equally near ones the most recently created, which is drawn on
  top. A point can be grabbed ``grab_margin_px`` beyond its marker.
* **Moving** the mouse drags the grabbed point to the cursor, less the offset
  at which it was grabbed, so it never jumps. The target is kept inside the
  plane so a point cannot be dragged out of sight.
* **Releasing** the button, the cursor leaving the window or the window
  losing focus ends the drag.
* While no button is held, moving the mouse **hovers** points: the hovered
  point is highlighted when drawn, and the hand cursor is shown.

:meth:`DragController.update` (called every frame by the plane) drags the
grabbed point again to the last cursor position, so it stays under the
cursor while the view eases (zooming with the wheel in the middle of a drag)
or while its curve moves.

For sliders, which live in panels in the corners of the plane (see
:mod:`pygridplot.core.handles.panel`):

* **Pressing** on a slider's track moves the handle there and starts
  dragging; pressing on the handle itself grabs it where it is, so it does
  not jump. A press anywhere else in a panel is used up, so the plane never
  pans from under a panel.
* **Moving** the mouse drags the handle along the track (the value stays
  within the range, on the step).
* **The wheel** over a slider moves it by a step (or 1% of the track),
  instead of zooming; over the rest of a panel it does nothing.
* Hovering a slider's track highlights its handle and shows the hand.

The panels' layout comes from the last frame's drawing: what the user sees
is what a press hits, and a slider that was not drawn cannot be dragged.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Callable, Sequence
from typing import Any

import pygame

from pygridplot.core.handles.panel import Panel, SliderRow
from pygridplot.core.handles.point import DraggablePoint
from pygridplot.core.handles.slider import Slider
from pygridplot.core.view import View
from pygridplot.interaction import InteractionConfig


class DragController:
    """Lets the user drag a plane's draggable points with the mouse.

    Args:
        view: The plane's view.
        config: The control settings (``drag_button``, ``grab_margin_px``,
            ``hover_cursor``).
        points: Returns the plane's points, in creation order.
        blocked: Returns whether another control owns the mouse (a pan in
            progress), so nothing is hovered or grabbed meanwhile.
    """

    def __init__(
        self,
        view: View,
        config: InteractionConfig,
        points: Callable[[], Sequence[DraggablePoint]],
        *,
        blocked: Callable[[], bool] = lambda: False,
    ) -> None:
        self._view = view
        self.config = config
        self._points = points
        self._blocked = blocked
        self._mouse: tuple[int, int] | None = None
        self._dragged: DraggablePoint | None = None
        self._hovered: DraggablePoint | None = None
        self._offset = (0.0, 0.0)

    @property
    def dragged(self) -> DraggablePoint | None:
        """The point being dragged, if any."""
        return self._dragged

    @property
    def hovered(self) -> DraggablePoint | None:
        """The point under the cursor, if any."""
        return self._hovered

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Apply one pygame event to the points.

        Args:
            event: The event.

        Returns:
            ``True`` if the event grabbed, dragged or released a point (the
            plane's own controls should then ignore it).
        """
        kind = event.type
        if kind == pygame.MOUSEMOTION:
            self._mouse = (int(event.pos[0]), int(event.pos[1]))
            if self._dragged is not None:
                self._drag()
                return True
            self._refresh_hover()
            return False
        if kind == pygame.MOUSEBUTTONDOWN:
            self._mouse = (int(event.pos[0]), int(event.pos[1]))
            return event.button == self.config.drag_button and self._grab()
        if kind == pygame.MOUSEBUTTONUP:
            self._mouse = (int(event.pos[0]), int(event.pos[1]))
            if event.button == self.config.drag_button and self._dragged is not None:
                self.release()
                return True
            return False
        if kind in (pygame.WINDOWLEAVE, pygame.WINDOWFOCUSLOST):
            self.release()
            self._mouse = None
            self._set_hovered(None)
        return False

    def update(self) -> None:
        """Keep the dragged point under the cursor and the hover state current.

        Called every frame: the view or a point's curve may have moved under
        a cursor that did not.
        """
        if self._dragged is not None:
            if self._dragged not in self._points():  # removed while dragged
                self.release()
            else:
                self._drag()
                return
        self._refresh_hover()

    def release(self) -> None:
        """End any drag in progress."""
        point, self._dragged = self._dragged, None
        if point is not None:
            point.set_interaction(hovered=point is self._hovered, dragging=False)
            self._refresh_hover()

    def forget(self, point: DraggablePoint) -> None:
        """Stop tracking a point that was removed from the plane.

        Args:
            point: The removed point.
        """
        if point is self._dragged:
            self._dragged = None
        if point is self._hovered:
            self._hovered = None
        point.set_interaction(hovered=False, dragging=False)
        self._update_cursor()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def hit(self, px: float, py: float) -> DraggablePoint | None:
        """Return the point that a press at a screen position would grab.

        Args:
            px: Screen x.
            py: Screen y.

        Returns:
            The nearest enabled, visible point within its grab distance
            (ties go to the most recently created), or ``None``.
        """
        best: DraggablePoint | None = None
        best_distance = math.inf
        for point in self._points():
            if not (point.enabled and point.visible):
                continue
            sx, sy = point.screen_position
            distance = math.hypot(sx - px, sy - py)
            # The outline counts: by default it is drawn in the background colour.
            reach = point.style.radius + point.style.edge_width
            reach += self.config.grab_margin_px
            if distance <= reach and distance <= best_distance:
                best, best_distance = point, distance
        return best

    def _grab(self) -> bool:
        """Grab the point under the cursor, if any.

        Returns:
            Whether a point was grabbed.
        """
        mouse = self._mouse
        if (
            mouse is None
            or self._blocked()
            or not self._view.rect.contains_point(*mouse)
        ):
            return False
        point = self.hit(*mouse)
        if point is None:
            return False
        sx, sy = point.screen_position
        self._offset = (mouse[0] - sx, mouse[1] - sy)
        self._dragged = point
        self._set_hovered(point)
        point.set_interaction(hovered=True, dragging=True)
        return True

    def _drag(self) -> None:
        """Drag the grabbed point to the cursor, less the grab offset."""
        point, mouse = self._dragged, self._mouse
        if point is None or mouse is None:
            return
        rect = self._view.rect
        x = min(max(mouse[0] - self._offset[0], rect.left), rect.right - 1)
        y = min(max(mouse[1] - self._offset[1], rect.top), rect.bottom - 1)
        point.drag_to(x, y)

    def _refresh_hover(self) -> None:
        """Hover the point under the cursor (none while another control drags)."""
        mouse = self._mouse
        inside = mouse is not None and self._view.rect.contains_point(*mouse)
        found = None
        if inside and mouse is not None and not self._blocked():
            found = self.hit(*mouse)
        self._set_hovered(found)

    def _set_hovered(self, point: DraggablePoint | None) -> None:
        """Change which point is hovered, and the cursor with it.

        Args:
            point: The newly hovered point, or ``None``.
        """
        old = self._hovered
        if old is not point and old is not None:
            old.set_interaction(hovered=False, dragging=old is self._dragged)
        self._hovered = point
        if point is not None:
            point.set_interaction(hovered=True, dragging=point is self._dragged)
        self._update_cursor()

    def _update_cursor(self) -> None:
        """Ask for the hand cursor while a point is hovered or dragged."""
        want = self.config.hover_cursor and (
            self._hovered is not None or self._dragged is not None
        )
        HAND_CURSOR.request(self, want)


class SliderController:
    """Lets the user drag a plane's sliders, and turn them with the mouse wheel.

    Args:
        config: The control settings (``drag_button``, ``hover_cursor``).
        panels: Returns the panels as last drawn.
        blocked: Returns whether another control owns the mouse (a pan or a
            point being dragged), so no slider is hovered or grabbed meanwhile.
    """

    def __init__(
        self,
        config: InteractionConfig,
        panels: Callable[[], Sequence[Panel]],
        *,
        blocked: Callable[[], bool] = lambda: False,
    ) -> None:
        self.config = config
        self._panels = panels
        self._blocked = blocked
        self._mouse: tuple[int, int] | None = None
        self._dragged: Slider[Any] | None = None
        self._hovered: Slider[Any] | None = None
        self._offset = 0.0

    @property
    def dragged(self) -> Slider[Any] | None:
        """The slider being dragged, if any."""
        return self._dragged

    @property
    def hovered(self) -> Slider[Any] | None:
        """The slider under the cursor, if any."""
        return self._hovered

    @property
    def owns_cursor(self) -> bool:
        """Whether a slider is being dragged or the cursor is over a panel.

        Draggable points under a panel are then neither hovered nor grabbed.
        """
        return self._dragged is not None or self._panel_at(self._mouse) is not None

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Apply one pygame event to the sliders.

        Args:
            event: The event.

        Returns:
            ``True`` if the event was for the sliders or their panels (the
            plane's other controls should then ignore it).
        """
        kind = event.type
        if kind == pygame.MOUSEMOTION:
            self._mouse = (int(event.pos[0]), int(event.pos[1]))
            if self._dragged is not None:
                self._drag()
                return True
            self._refresh_hover()
            return False
        if kind == pygame.MOUSEBUTTONDOWN:
            self._mouse = (int(event.pos[0]), int(event.pos[1]))
            if self._blocked() or self._panel_at(self._mouse) is None:
                return False
            if event.button == self.config.drag_button:
                self._grab()
            return True  # a press on a panel never pans the plane
        if kind == pygame.MOUSEBUTTONUP:
            self._mouse = (int(event.pos[0]), int(event.pos[1]))
            if event.button == self.config.drag_button and self._dragged is not None:
                self.release()
                return True
            return False
        if kind == pygame.MOUSEWHEEL:
            if self._blocked() or self._panel_at(self._mouse) is None:
                return False
            row = self._row_at(self._mouse)
            if row is not None and row.slider.enabled and event.y:
                row.slider.nudge(int(event.y))
            return True
        if kind in (pygame.WINDOWLEAVE, pygame.WINDOWFOCUSLOST):
            self.release()
            self._mouse = None
            self._set_hovered(None)
        return False

    def update(self) -> None:
        """Keep the hover state current, and end a drag whose slider is gone.

        Called every frame: a slider may stop being drawn (or be removed)
        in the middle of a drag.
        """
        if self._dragged is not None and self._row_of(self._dragged) is None:
            self.release()
        if self._dragged is None:
            self._refresh_hover()

    def release(self) -> None:
        """End any drag in progress."""
        slider, self._dragged = self._dragged, None
        if slider is not None:
            slider.set_interaction(hovered=slider is self._hovered, dragging=False)
            self._refresh_hover()

    def forget(self, slider: Slider[Any]) -> None:
        """Stop tracking a slider that was removed from the plane.

        Args:
            slider: The removed slider.
        """
        if slider is self._dragged:
            self._dragged = None
        if slider is self._hovered:
            self._hovered = None
        slider.set_interaction(hovered=False, dragging=False)
        self._update_cursor()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _panel_at(self, mouse: tuple[int, int] | None) -> Panel | None:
        """The panel under a screen point, if any."""
        if mouse is None:
            return None
        for panel in self._panels():
            if panel.rect.contains_point(*mouse):
                return panel
        return None

    def _row_at(self, mouse: tuple[int, int] | None) -> SliderRow | None:
        """The row whose band is under a screen point, if any."""
        if mouse is None:
            return None
        for panel in self._panels():
            for row in panel.rows:
                if row.contains(*mouse):
                    return row
        return None

    def _row_of(self, slider: Slider[Any]) -> SliderRow | None:
        """A slider's row as last drawn, if it was drawn."""
        for panel in self._panels():
            for row in panel.rows:
                if row.slider is slider:
                    return row
        return None

    def _grab(self) -> None:
        """Grab the slider under the cursor, if any, moving it there off the handle."""
        mouse = self._mouse
        row = self._row_at(mouse)
        if mouse is None or row is None or not row.slider.enabled:
            return
        slider = row.slider
        if row.on_handle(*mouse):
            self._offset = mouse[0] - row.handle_x
        else:
            self._offset = 0.0
            slider.drag_to(row.fraction_at(mouse[0]))
        self._dragged = slider
        self._set_hovered(slider)
        slider.set_interaction(hovered=True, dragging=True)

    def _drag(self) -> None:
        """Drag the grabbed slider's handle to the cursor, less the grab offset."""
        slider, mouse = self._dragged, self._mouse
        if slider is None or mouse is None:
            return
        row = self._row_of(slider)
        if row is not None:
            slider.drag_to(row.fraction_at(mouse[0] - self._offset))

    def _refresh_hover(self) -> None:
        """Hover the slider under the cursor (none while another control drags)."""
        row = None if self._blocked() else self._row_at(self._mouse)
        self._set_hovered(
            row.slider if row is not None and row.slider.enabled else None
        )

    def _set_hovered(self, slider: Slider[Any] | None) -> None:
        """Change which slider is hovered, and the cursor with it."""
        old = self._hovered
        if old is not slider and old is not None:
            old.set_interaction(hovered=False, dragging=old is self._dragged)
        self._hovered = slider
        if slider is not None:
            slider.set_interaction(hovered=True, dragging=slider is self._dragged)
        self._update_cursor()

    def _update_cursor(self) -> None:
        """Ask for the hand cursor while a slider is hovered or dragged."""
        want = self.config.hover_cursor and (
            self._hovered is not None or self._dragged is not None
        )
        HAND_CURSOR.request(self, want)


class HandCursor:
    """The hand cursor, shared by every plane in the program.

    The mouse cursor belongs to the whole window, so the planes in it must
    agree on it: with two planes, one may still want the hand when the other
    stops wanting it. The hand is shown while at least one controller asks
    for it, and the cursor that was there before is restored when the last
    one stops. Controllers are held weakly, so one that is thrown away while
    asking does not keep the hand forever.
    """

    def __init__(self) -> None:
        self._wanting: weakref.WeakSet[object] = weakref.WeakSet()
        self._saved: pygame.cursors.Cursor | None = None

    def request(self, owner: object, want: bool) -> None:
        """Record whether ``owner`` wants the hand cursor, and update the cursor.

        Args:
            owner: Who is asking (a :class:`DragController`).
            want: Whether it wants the hand now.
        """
        if want:
            self._wanting.add(owner)
        else:
            self._wanting.discard(owner)
        try:
            if self._wanting and self._saved is None:
                self._saved = pygame.mouse.get_cursor()
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
            elif not self._wanting and self._saved is not None:
                saved, self._saved = self._saved, None
                pygame.mouse.set_cursor(saved)
        except pygame.error:  # no display, or a driver without cursors
            self._saved = None

    def reset(self) -> None:
        """Forget every request, without touching the cursor (for tests)."""
        self._wanting.clear()
        self._saved = None


HAND_CURSOR = HandCursor()
"""The program's one :class:`HandCursor`."""
