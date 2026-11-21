"""Turning pygame mouse events into drags of draggable points.

:class:`DragController` is the pygame side of
:class:`~pygridplot.core.handles.point.DraggablePoint`. The plane passes it
every event before its own pan and zoom controls:

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
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import pygame

from pygridplot.core.handles.point import DraggablePoint
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
        self._saved_cursor: pygame.cursors.Cursor | None = None

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
        """Show the hand cursor over a point, and restore the previous one after."""
        want = self.config.hover_cursor and (
            self._hovered is not None or self._dragged is not None
        )
        try:
            if want and self._saved_cursor is None:
                self._saved_cursor = pygame.mouse.get_cursor()
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
            elif not want and self._saved_cursor is not None:
                saved, self._saved_cursor = self._saved_cursor, None
                pygame.mouse.set_cursor(saved)
        except pygame.error:  # no display, or a driver without cursors
            self._saved_cursor = None
