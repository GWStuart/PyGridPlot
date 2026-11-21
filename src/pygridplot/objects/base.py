"""What every object shares: visibility, drawing, errors and styles.

* :class:`PlotObject`: the base of every object: visibility, drawing in two
  phases (:meth:`~PlotObject.update`, then :meth:`~PlotObject.render`), and
  the error that stopped it being drawn.
* :class:`StyledObject` and its kinds (:class:`LineObject`,
  :class:`MarkerObject`): objects drawn with a style, with a palette slot
  for their default colour.

A kind of object implements ``_render`` (draw itself with one of the
plane's methods) and ``_describe`` (a short description for ``repr``); the
base classes do the rest.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from pygridplot.core.handles.point import DraggablePoint
from pygridplot.core.plot.style import LineStyle, PointStyle, with_overrides
from pygridplot.objects.errors import warn

if TYPE_CHECKING:
    import pygame

    from pygridplot.plane import Plane


class PlotObject:
    """Something to draw, made once and changed through its attributes.

    Every attribute is read when the object is drawn. Draw objects by adding
    them to a plane (``plane.add(obj)``, then ``plane.draw(screen)`` every
    frame), or one at a time with :meth:`draw`.
    """

    def __init__(self, *, visible: bool = True) -> None:
        self._visible = bool(visible)
        self._draw_error: str | None = None
        self._reported: str | None = None

    @property
    def visible(self) -> bool:
        """Whether the object is drawn."""
        return self._visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        self._visible = bool(visible)

    @property
    def error(self) -> str | None:
        """Why the object could not be drawn the last time, or ``None``.

        For example an invalid setting for the plane (an object is never
        stopped by its own functions: as for the plane's methods, a function
        that raises or returns NaN leaves a gap).
        """
        return self._draw_error

    # --- Drawing ------------------------------------------------------------
    def update(self, plane: Plane) -> None:
        """Prepare to draw on a plane: the first half of :meth:`draw`.

        A plane updates every object before rendering any, so that an object
        reading a draggable point sees where it was dragged this frame.
        Programs normally call :meth:`draw`, or let the plane do it.

        Args:
            plane: The plane about to be drawn on.
        """
        self._draw_error = None

    def render(self, plane: Plane, surface: pygame.Surface) -> None:
        """Draw onto a plane: the second half of :meth:`draw`.

        A hidden object draws nothing. If the plane refuses to draw it, the
        error is kept in :attr:`error` and reported with a
        :class:`~pygridplot.PlotWarning` the first time it appears.

        Args:
            plane: The plane.
            surface: The surface (usually the display).
        """
        if not self._visible:
            return
        if self._draw_error is None:
            try:
                self._render(plane, surface)
            except (TypeError, ValueError) as failure:
                self._draw_error = str(failure)
        self._report(self._draw_error)

    def handles(self, plane: Plane) -> tuple[DraggablePoint, ...]:
        """Return the draggable handles the object shows on a plane (to draw last).

        Args:
            plane: The plane.

        Returns:
            The handles (none, except for draggable points).
        """
        return ()

    def detach(self, plane: Plane) -> None:
        """Take anything the object keeps on a plane off it (draggable handles).

        Args:
            plane: The plane the object will no longer be drawn on.
        """

    def draw(self, plane: Plane, surface: pygame.Surface) -> None:
        """Draw the object onto a plane now, with its draggable handle on top.

        Use it to place one object among immediate drawing calls; objects
        added to a plane are drawn by :meth:`Plane.draw_objects
        <pygridplot.Plane.draw_objects>`.

        Args:
            plane: The plane.
            surface: The surface (usually the display).
        """
        self.update(plane)
        self.render(plane, surface)
        shown = self.handles(plane)
        if shown:
            plane.draw_draggables(surface, *shown)

    def __repr__(self) -> str:
        """Return the class and a short description."""
        detail = self._describe()
        return f"<{type(self).__name__}{': ' if detail else ''}{detail}>"

    # --- For subclasses ---------------------------------------------------
    def _render(self, plane: Plane, surface: pygame.Surface) -> None:
        """Draw (the object is visible): one call of a plane method."""

    def _describe(self) -> str:
        """Return a short description of the object, for ``repr`` and warnings."""
        return ""

    def _report(self, message: str | None) -> None:
        """Warn about an error the first time it appears (or changes)."""
        if message == self._reported:
            return
        self._reported = message
        if message is not None:
            warn(f"{self!r}: {message}")


def style_property(field: str, annotation: str, doc: str) -> property:
    """Make a property for one field of a styled object's style.

    Reading it reads the style's field; setting it replaces the style with
    one where the field is changed (styles are immutable).

    Args:
        field: The style's field.
        annotation: Its type, for the documentation.
        doc: The property's documentation.

    Returns:
        The property.
    """

    def get(self: StyledObject) -> Any:
        return getattr(self._style, field)

    def put(self: StyledObject, value: Any) -> None:
        self._set_style(**{field: value})

    get.__annotations__["return"] = annotation
    return property(get, put, doc=doc)


class StyledObject(PlotObject):
    """An object drawn with a style: a line, a point or a polygon style.

    An object without a colour takes the colour of its *palette slot* from
    the plane's theme. A plane gives each object added to it the next slot,
    so colours stay with their objects when others are hidden or removed,
    and follow the theme; an object drawn without one takes the next colour
    of the frame, like the plane's drawing methods.
    """

    style_type: ClassVar[type] = LineStyle
    default_style: ClassVar[Any] = LineStyle()

    def __init__(
        self,
        *,
        style: Any = None,
        overrides: dict[str, Any] | None = None,
        palette: int | None = None,
        visible: bool = True,
        default: Any = None,
    ) -> None:
        super().__init__(visible=visible)
        base = self._check_style(style) if style is not None else default
        if base is None:
            base = self.default_style
        self._style = with_overrides(base, **(overrides or {}))
        self._palette = self._check_palette(palette)

    @property
    def style(self) -> Any:
        """The object's style: a :class:`~pygridplot.LineStyle`, ..."""
        return self._style

    @style.setter
    def style(self, style: Any) -> None:
        self._style = self._check_style(style)

    @property
    def palette(self) -> int | None:
        """The palette slot giving the object its colour when ``color`` is ``None``."""
        return self._palette

    @palette.setter
    def palette(self, slot: int | None) -> None:
        self._palette = self._check_palette(slot)

    color = style_property(
        "color",
        "ColorLike | None",
        "The colour (``None`` for the colour of the object's palette slot).",
    )
    opacity = style_property("opacity", "float", "The opacity, in ``[0, 1]``.")

    def _color(self, plane: Plane) -> Any:
        """Return the colour to draw with (``None``: the plane's next colour)."""
        if self.color is not None:
            return self.color
        if self._palette is None:
            return None
        palette = plane.theme.palette
        return palette[self._palette % len(palette)]

    def _set_style(self, **changes: Any) -> None:
        self._style = replace(self._style, **changes)

    def _check_style(self, style: Any) -> Any:
        if not isinstance(style, self.style_type):
            msg = f"style must be a {self.style_type.__name__}, got {style!r}"
            raise TypeError(msg)
        return style

    @staticmethod
    def _check_palette(slot: object) -> int | None:
        if slot is None:
            return None
        if (
            isinstance(slot, bool)
            or not isinstance(slot, (int, np.integer))
            or slot < 0
        ):
            msg = f"palette must be a non-negative whole number or None, got {slot!r}"
            raise ValueError(msg)
        return int(slot)


class LineObject(StyledObject):
    """An object drawn as lines, with a :class:`~pygridplot.LineStyle`."""

    style_type: ClassVar[type] = LineStyle
    default_style: ClassVar[Any] = LineStyle()

    width = style_property("width", "float", "The line width, in pixels.")


class MarkerObject(StyledObject):
    """An object drawn as markers, with a :class:`~pygridplot.PointStyle`."""

    style_type: ClassVar[type] = PointStyle
    default_style: ClassVar[Any] = PointStyle()

    radius = style_property("radius", "float", "The marker size, in pixels.")
    marker = style_property(
        "marker", "MarkerShape", 'The marker shape (``"circle"``, ``"square"``, ...).'
    )
    edge_color = style_property(
        "edge_color",
        "ColorLike | None",
        "The colour of the marker's outline (``None`` for none).",
    )
    edge_width = style_property(
        "edge_width", "float", "The outline's thickness, in pixels."
    )
